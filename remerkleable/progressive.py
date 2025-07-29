# This file implements `ProgressiveList` according to https://eips.ethereum.org/EIPS/eip-7916
# The EIP is still under review, functionality may change or go away without deprecation.

from itertools import chain
from typing import BinaryIO, Iterator, List as PyList, Optional, Tuple, Type, TypeVar, cast
from types import GeneratorType
from remerkleable.basic import boolean, uint8, uint256
from remerkleable.bitfields import BitsView, append_bit, deserialize_bits, pop_bit, serialize_bits
from remerkleable.core import BasicView, ObjType, View, ViewHook, OFFSET_BYTE_LENGTH, pack_bits_to_chunks
from remerkleable.complex import MonoSubtreeView, create_readonly_iter, append_view, pop_and_summarize
from remerkleable.readonly_iters import BitfieldIter
from remerkleable.tree import Gindex, Node, PairNode, subtree_fill_to_contents, zero_node

V = TypeVar('V', bound=View)


def subtree_fill_progressive(nodes: PyList[Node], depth=0) -> Node:
    if len(nodes) == 0:
        return zero_node(0)
    base_size = 1 << depth
    return PairNode(
        subtree_fill_progressive(nodes[base_size:], depth + 2),
        subtree_fill_to_contents(nodes[:base_size], depth),
    )


def readonly_iter_progressive(backing: Node, length: int, elem_type: Type[View], is_packed: bool, depth=0):
    if length == 0:
        assert uint256.view_from_backing(backing) == uint256(0)

        class EmptyIter(object):
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration
        return EmptyIter()

    base_size = 1 << depth
    elems_per_chunk = 32 // elem_type.type_byte_length() if is_packed else 1

    subtree_len = min(base_size * elems_per_chunk, length)
    return chain(
        create_readonly_iter(backing.get_right(), depth, subtree_len, elem_type, is_packed),
        readonly_iter_progressive(backing.get_left(), length - subtree_len, elem_type, is_packed, depth + 2),
    )


def to_gindex_progressive(chunk_i: int) -> Tuple[Gindex, int, int]:
    depth = 0
    gindex = 2
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

    target = 2
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

        return readonly_iter_progressive(backing, length, elem_type, is_packed)

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
    if bitlen == 0:
        assert uint256.view_from_backing(backing) == uint256(0)
        yield from []
        return

    tree_depth = 0
    while True:
        base_bits = 256 << tree_depth
        is_final_chunk = bitlen <= base_bits
        yield backing.get_right(), tree_depth, min(bitlen, base_bits), is_final_chunk
        if is_final_chunk:
            return
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
        return f"ProgressiveBitlist"

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
