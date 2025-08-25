# This file implements `ProgressiveList` according to https://eips.ethereum.org/EIPS/eip-7916
# and `ProgressiveContainer` according to https://eips.ethereum.org/EIPS/eip-7495
# The EIP is still under review, functionality may change or go away without deprecation.

from typing import Any, BinaryIO, Dict, Iterator, List as PyList, Literal, Optional, Tuple, Type, TypeVar, cast
from types import GeneratorType
from textwrap import indent
from remerkleable.basic import boolean, uint8, uint256
from remerkleable.bitfields import BitsView, Bitvector, append_bit, deserialize_bits, pop_bit, serialize_bits
from remerkleable.core import BasicView, ObjType, View, ViewHook, ViewMeta, OFFSET_BYTE_LENGTH, pack_bits_to_chunks
from remerkleable.complex import Container, Fields, MonoSubtreeView, \
    append_view, create_readonly_iter, get_field_val_repr, pop_and_summarize
from remerkleable.readonly_iters import BitfieldIter, NodeIter
from remerkleable.tree import Gindex, Node, PairNode, subtree_fill_to_contents, zero_node, LEFT_GINDEX, RIGHT_GINDEX

V = TypeVar('V', bound=View)


def subtree_fill_progressive(nodes: PyList[Node], depth=0) -> Node:
    if len(nodes) == 0:
        return zero_node(0)
    base_size = 1 << depth
    return PairNode(
        subtree_fill_progressive(nodes[base_size:], depth + 2),
        subtree_fill_to_contents(nodes[:base_size], depth),
    )


def readonly_iter_progressive(backing: Node, length: int, elem_type: Type[View], is_packed: bool):
    yield from []
    tree_depth = 0
    while length > 0:
        base_size = 1 << tree_depth
        elems_per_chunk = 32 // elem_type.type_byte_length() if is_packed else 1
        subtree_len = min(base_size * elems_per_chunk, length)
        yield from create_readonly_iter(backing.get_right(), tree_depth, subtree_len, elem_type, is_packed)
        backing = backing.get_left()
        length -= subtree_len
        tree_depth += 2


def to_gindex_progressive(chunk_i: int) -> Tuple[Gindex, int, int]:
    depth = 0
    gindex = LEFT_GINDEX
    while True:
        base_size = 1 << depth
        if chunk_i < base_size:
            return (((gindex << 1) + 1) << depth) + chunk_i, depth, chunk_i
        chunk_i -= base_size
        depth += 2
        gindex <<= 1


def to_target_progressive(i: int, elems_per_chunk: int = 1) -> Tuple[Gindex, int, int]:
    chunk_i, offset_i = divmod(i, elems_per_chunk)

    _, depth, chunk_i = to_gindex_progressive(chunk_i)
    i = chunk_i * elems_per_chunk + offset_i

    target = LEFT_GINDEX
    d = 0
    while d < depth:
        target <<= 1
        d += 2

    return target, d, i


def to_target_progressive_elem(elem_type: Type[View], is_packed: bool, i: int) -> Tuple[Gindex, int, int]:
    if is_packed:
        return to_target_progressive(i, 32 // elem_type.type_byte_length())
    else:
        return to_target_progressive(i)


class ProgressiveList(MonoSubtreeView):
    __slots__ = ()

    def __new__(cls, *args, backing: Optional[Node] = None, hook: Optional[ViewHook] = None, **kwargs):
        if backing is not None:
            if len(args) != 0:
                raise Exception('cannot have both a backing and elements to init ProgressiveList')
            return super().__new__(cls, backing=backing, hook=hook, **kwargs)

        elem_cls = cls.element_cls()
        vals = list(args)
        if len(vals) == 1:
            val = vals[0]
            if isinstance(val, (GeneratorType, list, tuple)):
                vals = list(val)
            if issubclass(elem_cls, uint8):
                if isinstance(val, bytes):
                    vals = list(val)
                if isinstance(val, str):
                    if val[:2] == '0x':
                        val = val[2:]
                    vals = list(bytes.fromhex(val))
        input_views = []
        if len(vals) > 0:
            for el in vals:
                if isinstance(el, View):
                    input_views.append(el)
                else:
                    input_views.append(elem_cls.coerce_view(el))
            input_nodes = cls.views_into_chunks(input_views)
            contents = subtree_fill_progressive(input_nodes)
        else:
            contents = zero_node(0)
        backing = PairNode(contents, uint256(len(input_views)).get_backing())
        return super().__new__(cls, backing=backing, hook=hook, **kwargs)

    def __class_getitem__(cls, element_type) -> Type['ProgressiveList']:
        packed = isinstance(element_type, BasicView)

        class ProgressiveListView(ProgressiveList):
            @classmethod
            def is_packed(cls) -> bool:
                return packed

            @classmethod
            def element_cls(cls) -> Type[View]:
                return element_type

        ProgressiveListView.__name__ = ProgressiveListView.type_repr()
        return ProgressiveListView

    def length(self) -> int:
        return int(uint256.view_from_backing(self.get_backing().get_right()))

    def value_byte_length(self) -> int:
        elem_cls = self.__class__.element_cls()
        if elem_cls.is_fixed_byte_length():
            return elem_cls.type_byte_length() * self.length()
        else:
            return sum(OFFSET_BYTE_LENGTH + cast(View, el).value_byte_length() for el in iter(self))

    @classmethod
    def chunk_to_gindex(cls, chunk_i: int) -> Gindex:
        gindex, _, _ = to_gindex_progressive(chunk_i)
        return gindex

    def readonly_iter(self):
        length = self.length()
        backing = self.get_backing().get_left()

        elem_type: Type[View] = self.element_cls()
        is_packed = self.is_packed()

        yield from readonly_iter_progressive(backing, length, elem_type, is_packed)

    def append(self, v: View):
        ll = self.length()
        i = ll

        elem_type = self.__class__.element_cls()
        is_packed = self.__class__.is_packed()
        gindex, d, i = to_target_progressive_elem(elem_type, is_packed, i)

        if not isinstance(v, elem_type):
            v = elem_type.coerce_view(v)

        next_backing = self.get_backing()
        if i == 0:  # Create new subtree
            next_backing = next_backing.setter(gindex)(PairNode(zero_node(0), zero_node(d)))
        gindex = (gindex << 1) + 1
        next_backing = next_backing.setter(gindex)(append_view(
            next_backing.getter(gindex), d, i, v, elem_type, is_packed))

        next_backing = next_backing.setter(3)(uint256(ll + 1).get_backing())
        self.set_backing(next_backing)

    def pop(self):
        ll = self.length()
        if ll == 0:
            raise Exception('progressive list is empty, cannot pop')
        i = ll - 1

        if i == 0:
            self.set_backing(PairNode(zero_node(0), zero_node(0)))
            return

        elem_type = self.__class__.element_cls()
        is_packed = self.__class__.is_packed()
        gindex, d, i = to_target_progressive_elem(elem_type, is_packed, i)

        next_backing = self.get_backing()
        if i == 0:  # Delete entire subtree
            next_backing = next_backing.setter(gindex)(zero_node(0))
        else:
            gindex = (gindex << 1) + 1
            next_backing = next_backing.setter(gindex)(pop_and_summarize(
                next_backing.getter(gindex), d, i, elem_type, is_packed))

        next_backing = next_backing.setter(3)(uint256(ll - 1).get_backing())
        self.set_backing(next_backing)

    def get(self, i: int) -> View:
        i = int(i)
        if i < 0 or i >= self.length():
            raise IndexError
        return super().get(i)

    def set(self, i: int, v: View) -> None:
        i = int(i)
        if i < 0 or i >= self.length():
            raise IndexError
        super().set(i, v)

    @classmethod
    def type_repr(cls) -> str:
        return f'ProgressiveList[{cls.element_cls().__name__}]'

    @classmethod
    def is_valid_count(cls, count: int) -> bool:
        return 0 <= count

    @classmethod
    def min_byte_length(cls) -> int:
        return 0

    @classmethod
    def max_byte_length(cls) -> int:
        return 1 << 32  # Essentially unbounded, limited by offsets if nested

    def to_obj(self) -> ObjType:
        return list(el.to_obj() for el in self.readonly_iter())


def iter_progressive_bitlist(backing: Node, bitlen: int) -> Iterator[Tuple[Node, int, int, bool]]:
    yield from []
    tree_depth = 0
    while bitlen > 0:
        base_bits = 256 << tree_depth
        is_final_chunk = bitlen <= base_bits
        yield backing.get_right(), tree_depth, min(bitlen, base_bits), is_final_chunk
        backing = backing.get_left()
        bitlen -= base_bits
        tree_depth += 2


def to_target_progressive_bitlist(i: int) -> Tuple[Gindex, int, int]:
    return to_target_progressive(i, elems_per_chunk=256)


class ProgressiveBitlist(BitsView):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        vals = list(args)
        if len(vals) > 0:
            if len(vals) == 1 and isinstance(vals[0], (GeneratorType, list, tuple)):
                vals = list(vals[0])
            input_bits = list(map(bool, vals))
            input_nodes = pack_bits_to_chunks(input_bits)
            contents = subtree_fill_progressive(input_nodes)
            kwargs['backing'] = PairNode(contents, uint256(len(input_bits)).get_backing())
        return super().__new__(cls, **kwargs)

    def __iter__(self):
        bitlen = self.length()
        if bitlen == 0:
            yield from []
            return

        for backing, tree_depth, chunk_bitlen, _ in iter_progressive_bitlist(
            self.get_backing().get_left(), bitlen
        ):
            yield from BitfieldIter(backing, tree_depth, chunk_bitlen)

    @classmethod
    def default_node(cls) -> Node:
        return PairNode(zero_node(0), zero_node(0))  # mix-in 0 as list length

    @classmethod
    def type_repr(cls) -> str:
        return f'ProgressiveBitlist'

    @classmethod
    def min_byte_length(cls) -> int:
        return 1  # the delimiting bit will always require at least 1 byte

    @classmethod
    def max_byte_length(cls) -> int:
        return 1 << 32  # Essentially unbounded, limited by offsets if nested

    def length(self) -> int:
        return int(uint256.view_from_backing(self.get_backing().get_right()))

    def append(self, v: boolean):
        ll = self.length()
        i = ll

        gindex, d, i = to_target_progressive_bitlist(i)

        next_backing = self.get_backing()
        if i == 0:  # Create new subtree
            next_backing = next_backing.setter(gindex)(PairNode(zero_node(0), zero_node(d)))
        gindex = (gindex << 1) + 1
        next_backing = next_backing.setter(gindex)(append_bit(
            next_backing.getter(gindex), d, i, v))

        next_backing = next_backing.setter(3)(uint256(ll + 1).get_backing())
        self.set_backing(next_backing)

    def pop(self):
        ll = self.length()
        if ll == 0:
            raise Exception('progressive bitlist is empty, cannot pop')
        i = ll - 1

        if i == 0:
            self.set_backing(PairNode(zero_node(0), zero_node(0)))
            return

        gindex, d, i = to_target_progressive_bitlist(i)

        next_backing = self.get_backing()
        if i == 0:  # Delete entire subtree
            next_backing = next_backing.setter(gindex)(zero_node(0))
        else:
            gindex = (gindex << 1) + 1
            next_backing = next_backing.setter(gindex)(pop_bit(
                next_backing.getter(gindex), d, i))

        next_backing = next_backing.setter(3)(uint256(ll - 1).get_backing())
        self.set_backing(next_backing)

    def value_byte_length(self) -> int:
        # bit count in bytes rounded up + delimiting bit
        return (self.length() + 7 + 1) // 8

    @classmethod
    def chunk_to_gindex(cls, chunk_i: int) -> Gindex:
        gindex, _, _ = to_gindex_progressive(chunk_i)
        return gindex

    @classmethod
    def deserialize(cls: Type[V], stream: BinaryIO, scope: int) -> V:
        if scope < 1:
            raise Exception('cannot have empty scope for progressive bitlist, need at least a delimiting bit')
        chunks, bitlen = deserialize_bits(stream, scope, with_delimiting_bit=True)
        contents = subtree_fill_progressive(chunks)
        backing = PairNode(contents, uint256(bitlen).get_backing())
        return cls.view_from_backing(backing)

    def serialize(self, stream: BinaryIO) -> int:
        bitlen = self.length()
        if bitlen == 0:
            stream.write(b'\x01')  # empty bitlist still has a delimiting bit
            return 1

        byte_len = 0
        for backing, tree_depth, chunk_bitlen, is_final_chunk in iter_progressive_bitlist(
            self.get_backing().get_left(), bitlen
        ):
            byte_len += serialize_bits(
                backing, tree_depth, chunk_bitlen, stream,
                with_delimiting_bit=is_final_chunk,
            )
        return byte_len


def iter_progressive_container(backing: Node, active_fields: PyList[Literal[0, 1]], elem_types: PyList[Type[View]]):
    yield from []
    tree_depth = 0
    node_index = 0
    field_index = 0
    length = len(active_fields)
    while length > 0:
        base_size = 1 << tree_depth
        subtree_len = min(base_size, length)
        for node in NodeIter(backing.get_right(), tree_depth, subtree_len):
            if node_index >= len(active_fields):
                return
            if active_fields[node_index] == 1:
                yield elem_types[field_index].view_from_backing(node, None)
                field_index += 1
            node_index += 1
        backing = backing.get_left()
        length -= subtree_len
        tree_depth += 2


class ProgressiveContainer(Container):
    _active_fields: PyList[Literal[0, 1]]
    _field_indices: Dict[str, int]
    __slots__ = '_active_fields', '_field_indices'

    def __init_subclass__(cls, **kwargs):
        if '_active_fields' not in kwargs:
            raise TypeError(f'`active_fields` missing: `{cls.__name__}(ProgressiveContainer)`')
        cls._active_fields = kwargs.pop('_active_fields')

    def __new__(cls, active_fields: PyList[Literal[0, 1]]):
        class ProgressiveContainerMeta(ViewMeta):
            def __new__(cls, name, bases, dct):
                return super().__new__(cls, name, bases, dct, _active_fields=active_fields)

        class ProgressiveContainerView(ProgressiveContainer, metaclass=ProgressiveContainerMeta):
            def __init_subclass__(cls, **kwargs):
                if not all(x in (0, 1) for x in active_fields):
                    raise TypeError(
                        f'`active_fields` invalid: '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                if len(active_fields) == 0:
                    raise TypeError(
                        f'`active_fields` cannot be empty: '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                if active_fields[-1] == 0:
                    raise TypeError(
                        f'`active_fields` cannot end in 0: '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                if sum(active_fields) != len(cls.fields()):
                    raise TypeError(
                        f'`active_fields` count of 1 entries ({sum(active_fields)}) must match number of fields ({len(cls.fields())}): '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                if len(active_fields) > 256:
                    raise TypeError(
                        f'`active_fields` cannot have more than 256 entries: '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                if len(active_fields) >= 256 and 'more' not in cls.fields():
                    # A `ProgressiveContainer` cannot have more than 256 fields across iterations.
                    # A `more` field is recommended to enable introducing additional fields
                    raise TypeError(
                        f'`active_fields` at capacity but no `more` field present: '
                        f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')

                cls._field_indices = {}
                field_index = 0
                for fkey, ftyp in cls.fields().items():
                    if not isinstance(ftyp, View):
                        raise TypeError(
                            f'`{fkey}: {ftyp}` is not a `View`: '
                            f'`{cls.__name__}(ProgressiveContainer(active_fields={active_fields}))`')
                    while active_fields[field_index] == 0:
                        field_index += 1
                    cls._field_indices[fkey] = field_index
                    field_index += 1

            def __new__(cls, *args, backing: Optional[Node] = None, hook: Optional[ViewHook] = None, **kwargs):
                if len(args) > 0:
                    raise Exception('use keyword arguments, positional arguments not supported')
                if backing is not None:
                    if len(kwargs) != 0:
                        raise Exception('cannot have both a backing and elements to init fields')
                    return Container.__new__(cls, backing=backing, hook=hook, **kwargs)

                input_nodes = []
                field_index = 0
                for fkey, ftyp in cls.fields().items():
                    while active_fields[field_index] == 0:
                        input_nodes.append(zero_node(0))
                        field_index += 1
                    if fkey in kwargs:
                        finput = kwargs.pop(fkey)
                        if isinstance(finput, View):
                            fnode = finput.get_backing()
                        else:
                            fnode = ftyp.coerce_view(finput).get_backing()
                    else:
                        fnode = ftyp.default_node()
                    input_nodes.append(fnode)
                    field_index += 1
                if len(kwargs) > 0:
                    raise AttributeError(f'The field names [{"".join(kwargs.keys())}] are not defined in {cls}')

                backing = PairNode(
                    subtree_fill_progressive(input_nodes),
                    Bitvector[256](active_fields + [0] * (256 - len(active_fields))).get_backing(),
                )
                return Container.__new__(cls, backing=backing, hook=hook, **kwargs)

        return ProgressiveContainerView

    @classmethod
    def fields(cls) -> Fields:
        return cls.__dict__.get('__annotations__', {})

    @classmethod
    def tree_depth(cls) -> int:
        raise AttributeError('Progressive containers do not have a fixed tree depth')

    @classmethod
    def item_elem_cls(cls, i: int) -> Type[View]:
        for fkey, field_index in cls._field_indices.items():
            if field_index == i:
                return cls.fields()[fkey]
        raise IndexError(f'no field with index {i}')

    @classmethod
    def default_node(cls) -> Node:
        active_fields = cls._active_fields
        input_nodes = []
        field_index = 0
        for ftyp in cls.fields().values():
            while active_fields[field_index] == 0:
                input_nodes.append(zero_node(0))
                field_index += 1
            input_nodes.append(ftyp.default_node())
            field_index += 1
        return PairNode(
            subtree_fill_progressive(input_nodes),
            Bitvector[256](active_fields + [0] * (256 - len(active_fields))).get_backing(),
        )

    def active_fields(self) -> Bitvector[256]:
        active_fields_node = super().get_backing().get_right()
        return Bitvector[256].view_from_backing(active_fields_node)

    @classmethod
    def chunk_to_gindex(cls, chunk_i: int) -> Gindex:
        gindex, _, _ = to_gindex_progressive(chunk_i)
        return gindex

    def __iter__(self):
        active_fields = self.__class__._active_fields
        backing = self.get_backing().get_left()
        yield from iter_progressive_container(backing, active_fields, list(self.__class__.fields().values()))

    def __repr__(self):
        active_fields = self.__class__._active_fields
        return f'{self.__class__.__name__}(ProgressiveContainer(active_fields={active_fields})):\n' + '\n'.join(
            indent(get_field_val_repr(self, fkey, ftype), '  ')
            for fkey, ftype in self.__class__.fields().items())

    @classmethod
    def type_repr(cls) -> str:
        active_fields = cls._active_fields
        return f'{cls.__name__}(ProgressiveContainer(active_fields={active_fields})):\n' + '\n'.join(
            ('    ' + fkey + ': ' + ftype.__name__) for fkey, ftype in cls.fields().items())

    @classmethod
    def navigate_type(cls, key: Any) -> Type[View]:
        if key == '__active_fields__':
            return Bitvector[256]
        return cls.fields()[key]

    @classmethod
    def key_to_static_gindex(cls, key: Any) -> Gindex:
        if key == '__active_fields__':
            return RIGHT_GINDEX
        field_index = cls._field_indices[key]
        return cls.chunk_to_gindex(field_index)
