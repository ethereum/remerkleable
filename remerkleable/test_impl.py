# type: ignore

# flake8:noqa E501  Ignore long lines, some test cases are just inherently long

from typing import Iterable, List as PyList, Optional, Type
import io
from remerkleable.complex import Container, Vector, List
from remerkleable.basic import boolean, bit, byte, uint8, uint16, uint32, uint64, uint128, uint256
from remerkleable.bitfields import Bitvector, Bitlist
from remerkleable.byte_arrays import ByteVector, ByteList
from remerkleable.core import View, ObjType
from remerkleable.progressive import CompatibleUnion, ProgressiveBitlist, ProgressiveContainer, ProgressiveList
from remerkleable.union import Union
from hashlib import sha256

import json

import pytest


def bytes_hash(data: bytes):
    return sha256(data).digest()


class SingleFieldTestStruct(Container):
    A: byte


class SmallTestStruct(Container):
    A: uint16
    B: uint16


class FixedTestStruct(Container):
    A: uint8
    B: uint64
    C: uint32


class VarTestStruct(Container):
    A: uint16
    B: List[uint16, 1024]
    C: uint8


class ComplexTestStruct(Container):
    A: uint16
    B: List[uint16, 128]
    C: uint8
    D: List[byte, 256]
    E: VarTestStruct
    F: Vector[FixedTestStruct, 4]
    G: Vector[VarTestStruct, 2]


class ProgressiveTestStruct(Container):
    A: ProgressiveList[byte]
    B: ProgressiveList[uint64]
    C: ProgressiveList[SmallTestStruct]
    D: ProgressiveList[ProgressiveList[VarTestStruct]]

class ProgressiveSingleFieldContainerTestStruct(
    ProgressiveContainer(active_fields=[1])
):
    A: byte


class ProgressiveSingleListContainerTestStruct(
    ProgressiveContainer(active_fields=[0, 0, 0, 0, 1])
):
    C: ProgressiveBitlist


class ProgressiveVarTestStruct(
    ProgressiveContainer(active_fields=[1, 0, 1, 0, 1])
):
    A: byte
    B: List[uint16, 123]
    C: ProgressiveBitlist


class ProgressiveComplexTestStruct(
    ProgressiveContainer(active_fields=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
):
    A: byte
    B: List[uint16, 123]
    C: ProgressiveBitlist
    D: ProgressiveList[uint64]
    E: ProgressiveList[SmallTestStruct]
    F: ProgressiveList[ProgressiveList[VarTestStruct]]
    G: List[ProgressiveSingleFieldContainerTestStruct, 10]
    H: ProgressiveList[ProgressiveVarTestStruct]


sig_test_data = [0 for i in range(96)]
for k, v in {0: 1, 32: 2, 64: 3, 95: 0xff}.items():
    sig_test_data[k] = v


def chunk(hex: str) -> str:
    return (hex + ("00" * 32))[:64]  # just pad on the right, to 32 bytes (64 hex chars)


def h(a: str, b: str) -> str:
    return bytes_hash(bytes.fromhex(a) + bytes.fromhex(b)).hex()


# zero hashes, as strings, for
zero_hashes = [chunk("")]
for layer in range(1, 32):
    zero_hashes.append(h(zero_hashes[layer - 1], zero_hashes[layer - 1]))


def merge(a: str, branch: Iterable[str]) -> str:
    """
    Merge (out on left, branch on right) leaf a with branch items, branch is from bottom to top.
    """
    out = a
    for b in branch:
        out = h(out, b)
    return out


test_data = [
    ("bit F", bit, bit(False), "00", chunk("00"), False),
    ("bit T", bit, bit(True), "01", chunk("01"), True),
    ("boolean F", boolean, boolean(False), "00", chunk("00"), False),
    ("boolean T", boolean, boolean(True), "01", chunk("01"), True),
    ("bitlist empty", Bitlist[8], Bitlist[8](), "01", h(chunk(""), chunk("00")), "0x01"),
    ("progressiveBitlist empty", ProgressiveBitlist, ProgressiveBitlist(), "01", h(chunk(""), chunk("00")), "0x01"),
    ("bitvector TTFTFTFF", Bitvector[8], Bitvector[8](1, 1, 0, 1, 0, 1, 0, 0), "2b", chunk("2b"), "0x2b"),
    ("bitlist TTFTFTFF", Bitlist[8], Bitlist[8](1, 1, 0, 1, 0, 1, 0, 0), "2b01", h(chunk("2b"), chunk("08")), "0x2b01"),
    ("progressiveBitlist TTFTFTFF", ProgressiveBitlist, ProgressiveBitlist(1, 1, 0, 1, 0, 1, 0, 0), "2b01", h(h(chunk(""), chunk("2b")), chunk("08")), "0x2b01"),
    ("bitvector FTFT", Bitvector[4], Bitvector[4](0, 1, 0, 1), "0a", chunk("0a"), "0x0a"),
    ("bitlist FTFT", Bitlist[4], Bitlist[4](0, 1, 0, 1), "1a", h(chunk("0a"), chunk("04")), "0x1a"),
    ("progressiveBitlist FTFT", ProgressiveBitlist, ProgressiveBitlist(0, 1, 0, 1), "1a", h(h(chunk(""), chunk("0a")), chunk("04")), "0x1a"),
    ("bitvector FTF", Bitvector[3], Bitvector[3](0, 1, 0), "02", chunk("02"), "0x02"),
    ("bitlist FTF", Bitlist[3], Bitlist[3](0, 1, 0), "0a", h(chunk("02"), chunk("03")), "0x0a"),
    ("progressiveBitlist FTF", ProgressiveBitlist, ProgressiveBitlist(0, 1, 0), "0a", h(h(chunk(""), chunk("02")), chunk("03")), "0x0a"),
    ("bitvector TFTFFFTTFT", Bitvector[10], Bitvector[10](1, 0, 1, 0, 0, 0, 1, 1, 0, 1),
     "c502", chunk("c502"), "0xc502"),
    ("bitlist TFTFFFTTFT", Bitlist[16], Bitlist[16](1, 0, 1, 0, 0, 0, 1, 1, 0, 1),
     "c506", h(chunk("c502"), chunk("0a")), "0xc506"),
    ("progressiveBitlist TFTFFFTTFT", ProgressiveBitlist, ProgressiveBitlist(1, 0, 1, 0, 0, 0, 1, 1, 0, 1),
     "c506", h(h(chunk(""), chunk("c502")), chunk("0a")), "0xc506"),
    ("bitvector TFTFFFTTFTFFFFTT", Bitvector[16], Bitvector[16](1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1),
     "c5c2", chunk("c5c2"), "0xc5c2"),
    ("bitlist TFTFFFTTFTFFFFTT", Bitlist[16], Bitlist[16](1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1),
     "c5c201", h(chunk("c5c2"), chunk("10")), "0xc5c201"),
    ("progressiveBitlist TFTFFFTTFTFFFFTT", ProgressiveBitlist, ProgressiveBitlist(1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1),
     "c5c201", h(h(chunk(""), chunk("c5c2")), chunk("10")), "0xc5c201"),
    ("bitvector 256", Bitvector[256], Bitvector[256](1 for i in range(256)),
     "ff" * 32, chunk("ff" * 32), "0x" + ("ff" * 32)),
    ("bitlist 256", Bitlist[256], Bitlist[256](1 for i in range(256)),
     "ff" * 32 + "01", h(chunk("ff" * 32), chunk("0001")), "0x" + ("ff" * 32) + "01"),
    ("progressiveBitlist 256", ProgressiveBitlist, ProgressiveBitlist(1 for i in range(256)),
     "ff" * 32 + "01", h(h(chunk(""), chunk("ff" * 32)), chunk("0001")), "0x" + ("ff" * 32) + "01"),
    ("bitvector 257", Bitvector[257], Bitvector[257](1 for i in range(257)),
     "ff" * 32 + "01", h(chunk("ff" * 32), chunk("01")), "0x" + ("ff" * 32) + "01"),
    ("bitlist 257", Bitlist[257], Bitlist[257](1 for i in range(257)),
     "ff" * 32 + "03", h(h(chunk("ff" * 32), chunk("01")), chunk("0101")), "0x" + ("ff" * 32) + "03"),
    ("progressiveBitlist 257", ProgressiveBitlist, ProgressiveBitlist(1 for i in range(257)),
     "ff" * 32 + "03", h(h(h(chunk(""), merge(chunk("01"), zero_hashes[0:2])), chunk("ff" * 32)), chunk("0101")), "0x" + ("ff" * 32) + "03"),
    ("long bitvector", Bitvector[512], Bitvector[512](1 for i in range(512)),
     "ff" * 64, h("ff" * 32, "ff" * 32), "0x" + ("ff" * 64)),
    ("long bitlist", Bitlist[512], Bitlist[512](1),
     "03", h(h(chunk("01"), chunk("")), chunk("01")), "0x03"),
    ("long progressiveBitlist", ProgressiveBitlist, ProgressiveBitlist(1),
     "03", h(h(chunk(""), chunk("01")), chunk("01")), "0x03"),
    ("long bitlist", Bitlist[512], Bitlist[512](1 for i in range(512)),
     "ff" * 64 + "01", h(h("ff" * 32, "ff" * 32), chunk("0002")), "0x" + ("ff" * 64 + "01")),
    ("long progressiveBitlist", ProgressiveBitlist, ProgressiveBitlist(1 for i in range(512)),
     "ff" * 64 + "01", h(h(h(chunk(""), merge(chunk("ff" * 32), zero_hashes[0:2])), chunk("ff" * 32)), chunk("0002")), "0x" + ("ff" * 64 + "01")),
    ("odd bitvector", Bitvector[513], Bitvector[513](1 for i in range(513)),
     "ff" * 64 + "01", h(h("ff" * 32, "ff" * 32), h(chunk("01"), chunk(""))), "0x" + ("ff" * 64) + "01"),
    ("odd bitlist", Bitlist[513], Bitlist[513](1 for i in range(513)),
     "ff" * 64 + "03", h(h(h("ff" * 32, "ff" * 32), h(chunk("01"), chunk(""))), chunk("0102")),
     "0x" + ("ff" * 64) + "03"),
    ("odd progressiveBitlist", ProgressiveBitlist, ProgressiveBitlist(1 for i in range(513)),
     "ff" * 64 + "03", h(h(h(chunk(""), h(h(chunk("ff" * 32), chunk("01")), h(chunk(""), chunk("")))), chunk("ff" * 32)), chunk("0102")),
     "0x" + ("ff" * 64) + "03"),
    ("uint8 00", uint8, uint8(0x00), "00", chunk("00"), 0),
    ("uint8 01", uint8, uint8(0x01), "01", chunk("01"), 1),
    ("uint8 ab", uint8, uint8(0xab), "ab", chunk("ab"), 0xab),
    ("byte 00", byte, byte(0x00), "00", chunk("00"), 0),
    ("byte 01", byte, byte(0x01), "01", chunk("01"), 1),
    ("byte ab", byte, byte(0xab), "ab", chunk("ab"), 0xab),
    ("uint16 0000", uint16, uint16(0x0000), "0000", chunk("0000"), 0),
    ("uint16 abcd", uint16, uint16(0xabcd), "cdab", chunk("cdab"), 0xabcd),
    ("uint32 00000000", uint32, uint32(0x00000000), "00000000", chunk("00000000"), 0),
    ("uint32 01234567", uint32, uint32(0x01234567), "67452301", chunk("67452301"), 0x01234567),
    ("small (4567, 0123)", SmallTestStruct, SmallTestStruct(A=0x4567, B=0x0123), "67452301", h(chunk("6745"), chunk("2301")), {'A': 0x4567, 'B': 0x0123}),
    ("small [4567, 0123]::2", Vector[uint16, 2], Vector[uint16, 2](uint16(0x4567), uint16(0x0123)), "67452301", chunk("67452301"), (0x4567, 0x0123)),
    ("uint32 01234567", uint32, uint32(0x01234567), "67452301", chunk("67452301"), 0x01234567),
    ("uint64 0000000000000000", uint64, uint64(0x00000000), "0000000000000000", chunk("0000000000000000"), 0),
    ("uint64 0123456789abcdef", uint64, uint64(0x0123456789abcdef), "efcdab8967452301", chunk("efcdab8967452301"), 0x0123456789abcdef),
    ("uint128 00000000000000000000000000000000", uint128, uint128(0), "00000000000000000000000000000000", chunk("00000000000000000000000000000000"), '0x00000000000000000000000000000000'),
    ("uint128 11223344556677880123456789abcdef", uint128, uint128(0x11223344556677880123456789abcdef), "efcdab89674523018877665544332211", chunk("efcdab89674523018877665544332211"), '0xefcdab89674523018877665544332211'),
    ("bytes48", Vector[byte, 48], Vector[byte, 48](*range(48)), "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f",
     h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "202122232425262728292a2b2c2d2e2f00000000000000000000000000000000"), tuple(range(48))),
    ("raw bytes48", ByteVector[48], ByteVector[48](*range(48)), "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f",
     h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "202122232425262728292a2b2c2d2e2f00000000000000000000000000000000"),
     "0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f"),
    ("small empty bytelist", List[byte, 10], List[byte, 10](), "", h(chunk(""), chunk("00")), []),
    ("big empty bytelist", List[byte, 2048], List[byte, 2048](), "", h(zero_hashes[6], chunk("00")), []),
    ("raw small empty bytelist", ByteList[10], ByteList[10](), "", h(chunk(""), chunk("00")), "0x"),
    ("raw big empty bytelist", ByteList[2048], ByteList[2048](), "", h(zero_hashes[6], chunk("00")), "0x"),
    ("bytelist 7", List[byte, 7], List[byte, 7](*range(7)), "00010203040506",
     h(chunk("00010203040506"), chunk("07")), list(range(7))),
    ("raw bytelist 7", ByteList[7], ByteList[7](*range(7)), "00010203040506",
     h(chunk("00010203040506"), chunk("07")), "0x00010203040506"),
    ("bytelist 50", List[byte, 50], List[byte, 50](*range(50)), "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031",
     h(h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "202122232425262728292a2b2c2d2e2f30310000000000000000000000000000"), chunk("32")), list(range(50))),
    ("raw bytelist 50", ByteList[50], ByteList[50](*range(50)), "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031",
     h(h("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "202122232425262728292a2b2c2d2e2f30310000000000000000000000000000"), chunk("32")),
     "0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031"),
    ("bytelist 6/256", List[byte, 256], List[byte, 256](*range(6)), "000102030405",
     h(h(h(h(chunk("000102030405"), zero_hashes[0]), zero_hashes[1]), zero_hashes[2]), chunk("06")), list(range(6))),
    ("raw bytelist 6/256", ByteList[256], ByteList[256](*range(6)), "000102030405",
     h(h(h(h(chunk("000102030405"), zero_hashes[0]), zero_hashes[1]), zero_hashes[2]), chunk("06")), "0x000102030405"),
    ("sig", Vector[byte, 96], Vector[byte, 96](*sig_test_data),
     "0100000000000000000000000000000000000000000000000000000000000000"
     "0200000000000000000000000000000000000000000000000000000000000000"
     "03000000000000000000000000000000000000000000000000000000000000ff",
     h(h(chunk("01"), chunk("02")),
       h("03000000000000000000000000000000000000000000000000000000000000ff", chunk(""))), tuple(sig_test_data)),
    ("raw sig", ByteVector[96], ByteVector[96](*sig_test_data),
     "0100000000000000000000000000000000000000000000000000000000000000"
     "0200000000000000000000000000000000000000000000000000000000000000"
     "03000000000000000000000000000000000000000000000000000000000000ff",
     h(h(chunk("01"), chunk("02")),
       h("03000000000000000000000000000000000000000000000000000000000000ff", chunk(""))), "0x"
     "0100000000000000000000000000000000000000000000000000000000000000"
     "0200000000000000000000000000000000000000000000000000000000000000"
     "03000000000000000000000000000000000000000000000000000000000000ff",),
    ("3 sigs", Vector[ByteVector[96], 3], Vector[ByteVector[96], 3](
        [1] + [0 for i in range(95)],
        [2] + [0 for i in range(95)],
        [3] + [0 for i in range(95)]
    ),
     "01" + ("00" * 95) + "02" + ("00" * 95) + "03" + ("00" * 95),
     h(h(h(h(chunk("01"), chunk("")), zero_hashes[1]), h(h(chunk("02"), chunk("")), zero_hashes[1])),
       h(h(h(chunk("03"), chunk("")), zero_hashes[1]), chunk(""))),
     ("0x01" + ("00" * 95), "0x02" + ("00" * 95), "0x03" + ("00" * 95)),
     ),
    ("compatibleUnionA", CompatibleUnion({1: ProgressiveSingleFieldContainerTestStruct}),
     CompatibleUnion({1: ProgressiveSingleFieldContainerTestStruct})(
         selector=1, data=ProgressiveSingleFieldContainerTestStruct(A=0xab)
     ), "01ab", h(h(h(chunk(""), chunk("ab")), chunk("01")), chunk("01")), {'selector': 1, 'data': {'A': 0xab}}),
    ("compatibleUnionBC", CompatibleUnion({2: ProgressiveSingleListContainerTestStruct, 3: ProgressiveVarTestStruct}),
     CompatibleUnion({2: ProgressiveSingleListContainerTestStruct, 3: ProgressiveVarTestStruct})(
         selector=2, data=ProgressiveSingleListContainerTestStruct(C=ProgressiveBitlist(1, 0, 0, 1, 0)),
     ), "020400000029", h(h(
         h(h(chunk(""), h(zero_hashes[1], h(zero_hashes[0], h(h(chunk(""), chunk("09")), chunk("05"))))), zero_hashes[0]),
         chunk("10")
     ), chunk("02")), {'selector': 2, 'data': {'C': "0x29"}}),
    ("compatibleUnionABCA", CompatibleUnion({1: ProgressiveSingleFieldContainerTestStruct, 2: ProgressiveSingleListContainerTestStruct, 3: ProgressiveVarTestStruct, 4: ProgressiveSingleFieldContainerTestStruct}),
     CompatibleUnion({1: ProgressiveSingleFieldContainerTestStruct, 2: ProgressiveSingleListContainerTestStruct, 3: ProgressiveVarTestStruct, 4: ProgressiveSingleFieldContainerTestStruct})(
         selector=4, data=ProgressiveSingleFieldContainerTestStruct(A=0xab)
     ), "04ab", h(h(h(chunk(""), chunk("ab")), chunk("01")), chunk("04")), {'selector': 4, 'data': {'A': 0xab}}),
    ("single_type_union", Union[uint16], Union[uint16](selector=0, value=uint16(0xaabb)),
     "00bbaa", h(chunk("bbaa"), chunk("")), {'selector': 0, 'value': 0xaabb}),
    ("simple_union", Union[uint16, uint32], Union[uint16, uint32](selector=0, value=uint16(0xaabb)),
     "00bbaa", h(chunk("bbaa"), chunk("")), {'selector': 0, 'value': 0xaabb}),
    ("union with none", Union[None, uint16, uint32], Union[None, uint16, uint32](selector=0, value=None),
     "00", h(chunk(""), chunk("")), {'selector': 0, 'value': None}),
    ("union other than none", Union[None, uint16, uint32],
     Union[None, uint16, uint32](selector=1, value=uint16(0xaabb)),
     "01bbaa", h(chunk("bbaa"), chunk("01")), {'selector': 1, 'value': 0xaabb}),
    ("simple_union_other", Union[uint16, uint32], Union[uint16, uint32](selector=1, value=uint32(0xdeadbeef)),
     "01efbeadde", h(chunk("efbeadde"), chunk("01")), {'selector': 1, 'value': 0xdeadbeef}),
    ("simple_large_union", Union[uint16, uint32, uint8, List[uint16, 8]],
     Union[uint16, uint32, uint8, List[uint16, 8]](selector=2, value=uint8(0xaa)),
     "02aa", h(chunk("aa"), chunk("02")), {'selector': 2, 'value': 0xaa}),
    ("duplicate_type_union", Union[SingleFieldTestStruct, SingleFieldTestStruct],
     Union[SingleFieldTestStruct, SingleFieldTestStruct](selector=1, value=SingleFieldTestStruct(A=0xab)),
     "01ab", h(chunk("ab"), chunk("01")), {'selector': 1, 'value': {'A': 0xab}}),
    ("complex_union", Union[Vector[uint8, 3], SingleFieldTestStruct, VarTestStruct, ComplexTestStruct, uint16],
     Union[Vector[uint8, 3], SingleFieldTestStruct, VarTestStruct, ComplexTestStruct, uint16](
         selector=2, value=VarTestStruct(A=0xabcd, B=List[uint16, 1024](1, 2, 3), C=0xff)),
        "02" + "cdab07000000ff010002000300", h(
        h(
            h(
                chunk("cdab"),
                h(
                    merge(
                        chunk("010002000300"),
                        zero_hashes[0:6]
                    ),
                    chunk("03")  # length mix in
                )
            ),
            h(chunk("ff"), chunk(""))
        ), chunk("02")
    ), {'selector': 2, 'value': {'A': 0xabcd, 'B': [1, 2, 3], 'C': 0xff}}),
    ("singleFieldTestStruct", SingleFieldTestStruct, SingleFieldTestStruct(A=0xab), "ab", chunk("ab"), {'A': 0xab}),
    ("progressiveSingleFieldContainerTestStruct", ProgressiveSingleFieldContainerTestStruct,
     ProgressiveSingleFieldContainerTestStruct(A=0xab), "ab",
     h(h(chunk(""), chunk("ab")), chunk("01")), {'A': 0xab}),
    ("uint16 list", List[uint16, 32], List[uint16, 32](uint16(0xaabb), uint16(0xc0ad), uint16(0xeeff)), "bbaaadc0ffee",
     h(h(chunk("bbaaadc0ffee"), chunk("")), chunk("03000000")),  # max length: 32 * 2 = 64 bytes = 2 chunks
     [0xaabb, 0xc0ad, 0xeeff]),
    ("uint32 list", List[uint32, 128], List[uint32, 128](uint32(0xaabb), uint32(0xc0ad), uint32(0xeeff)), "bbaa0000adc00000ffee0000",
     # max length: 128 * 4 = 512 bytes = 16 chunks
     h(merge(chunk("bbaa0000adc00000ffee0000"), zero_hashes[0:4]), chunk("03")),
     [0xaabb, 0xc0ad, 0xeeff]  # still the same, no padding, just literals
     ),
    ("uint256 list", List[uint256, 32], List[uint256, 32](uint256(0xaabb), uint256(0xc0ad), uint256(0xeeff)),
     "bbaa000000000000000000000000000000000000000000000000000000000000"
     "adc0000000000000000000000000000000000000000000000000000000000000"
     "ffee000000000000000000000000000000000000000000000000000000000000",
     h(merge(h(h(chunk("bbaa"), chunk("adc0")), h(chunk("ffee"), chunk(""))), zero_hashes[2:5]), chunk("03")),
     [
         "0xbbaa000000000000000000000000000000000000000000000000000000000000",
         "0xadc0000000000000000000000000000000000000000000000000000000000000",
         "0xffee000000000000000000000000000000000000000000000000000000000000",
     ]
     ),
    ("uint256 list long", List[uint256, 128], List[uint256, 128](i for i in range(1, 20)),
     "".join([i.to_bytes(length=32, byteorder='little').hex() for i in range(1, 20)]),
     h(merge(
         h(
             h(
                 h(
                     h(h(chunk("01"), chunk("02")), h(chunk("03"), chunk("04"))),
                     h(h(chunk("05"), chunk("06")), h(chunk("07"), chunk("08"))),
                 ),
                 h(
                     h(h(chunk("09"), chunk("0a")), h(chunk("0b"), chunk("0c"))),
                     h(h(chunk("0d"), chunk("0e")), h(chunk("0f"), chunk("10"))),
                 )
             ),
             h(
                 h(
                     h(h(chunk("11"), chunk("12")), h(chunk("13"), chunk(""))),
                     zero_hashes[2]
                 ),
                 zero_hashes[3]
             )
         ),
         zero_hashes[5:7]), chunk("13")),  # 128 chunks = 7 deep
     ['0x' + i.to_bytes(length=32, byteorder='little').hex() for i in range(1, 20)],
     ),
    ("progressiveSingleListContainerTestStruct", ProgressiveSingleListContainerTestStruct,
     ProgressiveSingleListContainerTestStruct(C=ProgressiveBitlist(1, 0, 0, 1, 0)),
     "0400000029", h(
         h(h(chunk(""), h(zero_hashes[1], h(zero_hashes[0], h(h(chunk(""), chunk("09")), chunk("05"))))), zero_hashes[0]),
         chunk("10")
     ), {'C': "0x29"}),
    ("fixedTestStruct", FixedTestStruct, FixedTestStruct(A=0xab, B=0xaabbccdd00112233, C=0x12345678), "ab33221100ddccbbaa78563412",
     h(h(chunk("ab"), chunk("33221100ddccbbaa")), h(chunk("78563412"), chunk(""))), {'A': 0xab, 'B': 0xaabbccdd00112233, 'C': 0x12345678}),
    ("varTestStruct nil", VarTestStruct, VarTestStruct(A=0xabcd, C=0xff), "cdab07000000ff",
     h(h(chunk("cdab"), h(zero_hashes[6], chunk("00000000"))), h(chunk("ff"), chunk(""))), {'A': 0xabcd, 'B': [], 'C': 0xff}),
    ("varTestStruct empty", VarTestStruct, VarTestStruct(A=0xabcd, B=List[uint16, 1024](), C=0xff), "cdab07000000ff",
     h(h(chunk("cdab"), h(zero_hashes[6], chunk("00000000"))), h(chunk("ff"), chunk(""))),  # log2(1024*2/32)= 6 deep
     {'A': 0xabcd, 'B': [], 'C': 0xff}),
    ("varTestStruct some", VarTestStruct, VarTestStruct(A=0xabcd, B=List[uint16, 1024](1, 2, 3), C=0xff),
     "cdab07000000ff010002000300",
     h(
         h(
             chunk("cdab"),
             h(
                 merge(
                     chunk("010002000300"),
                     zero_hashes[0:6]
                 ),
                 chunk("03")  # length mix in
             )
         ),
         h(chunk("ff"), chunk(""))
    ), {'A': 0xabcd, 'B': [1, 2, 3], 'C': 0xff}),
    ("progressiveVarTestStruct nil", ProgressiveVarTestStruct,
     ProgressiveVarTestStruct(A=0xab),
     "ab090000000900000001", h(
         h(h(chunk(""), h(
             h(
                 zero_hashes[0],
                 h(zero_hashes[3], chunk("00"))
             ),
             h(
                 zero_hashes[0],
                 h(chunk(""), chunk("00"))
             )
         )), chunk("ab")),
         chunk("15")
     ), {'A': 0xab, 'B': [], 'C': "0x01"}),
    ("progressiveVarTestStruct empty", ProgressiveVarTestStruct,
     ProgressiveVarTestStruct(A=0xab, B=List[uint16, 123](), C=ProgressiveBitlist()),
     "ab090000000900000001", h(
         h(h(chunk(""), h(
             h(
                 zero_hashes[0],
                 h(zero_hashes[3], chunk("00"))
             ),
             h(
                 zero_hashes[0],
                 h(chunk(""), chunk("00"))
             )
         )), chunk("ab")),
         chunk("15")
     ), {'A': 0xab, 'B': [], 'C': "0x01"}),
    ("progressiveVarTestStruct some", ProgressiveVarTestStruct,
     ProgressiveVarTestStruct(A=0xab, B=List[uint16, 123](0x1122, 0x3344), C=ProgressiveBitlist(1, 0, 0, 1, 0)),
     "ab090000000d0000002211443329", h(
         h(h(chunk(""), h(
             h(
                 zero_hashes[0],
                 h(merge(chunk("22114433"), zero_hashes[0:3]), chunk("02"))
             ),
             h(
                 zero_hashes[0],
                 h(h(chunk(""), chunk("09")), chunk("05"))
             )
         )), chunk("ab")),
         chunk("15")
     ), {'A': 0xab, 'B': [0x1122, 0x3344], 'C': "0x29"}),
    ("complexTestStruct", ComplexTestStruct,
     ComplexTestStruct(
         A=0xaabb,
         B=List[uint16, 128](0x1122, 0x3344),
         C=0xff,
         D=List[byte, 256](b"foobar"),
         E=VarTestStruct(A=0xabcd, B=List[uint16, 1024](1, 2, 3), C=0xff),
         F=Vector[FixedTestStruct, 4](
             FixedTestStruct(A=0xcc, B=0x4242424242424242, C=0x13371337),
             FixedTestStruct(A=0xdd, B=0x3333333333333333, C=0xabcdabcd),
             FixedTestStruct(A=0xee, B=0x4444444444444444, C=0x00112233),
             FixedTestStruct(A=0xff, B=0x5555555555555555, C=0x44556677)),
         G=Vector[VarTestStruct, 2](
             VarTestStruct(A=0xdead, B=List[uint16, 1024](1, 2, 3), C=0x11),
             VarTestStruct(A=0xbeef, B=List[uint16, 1024](4, 5, 6), C=0x22)),
     ),
     "bbaa"
     "47000000"  # offset of B, []uint16
     "ff"
     "4b000000"  # offset of foobar
     "51000000"  # offset of E
     "cc424242424242424237133713"
     "dd3333333333333333cdabcdab"
     "ee444444444444444433221100"
     "ff555555555555555577665544"
     "5e000000"  # pointer to G
     "22114433"  # contents of B
     "666f6f626172"  # foobar
     "cdab07000000ff010002000300"  # contents of E
     "08000000" "15000000"  # [start G]: local offsets of [2]varTestStruct
     "adde0700000011010002000300"
     "efbe0700000022040005000600",
     h(
         h(
             h(  # A and B
                 chunk("bbaa"),
                 h(merge(chunk("22114433"), zero_hashes[0:3]), chunk("02000000"))  # 2*128/32 = 8 chunks
             ),
             h(  # C and D
                 chunk("ff"),
                 h(merge(chunk("666f6f626172"), zero_hashes[0:3]), chunk("06000000"))  # 256/32 = 8 chunks
             )
         ),
         h(
             h(  # E and F
                 h(h(chunk("cdab"), h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03000000"))),
                   h(chunk("ff"), chunk(""))),
                 h(
                     h(
                         h(h(chunk("cc"), chunk("4242424242424242")), h(chunk("37133713"), chunk(""))),
                         h(h(chunk("dd"), chunk("3333333333333333")), h(chunk("cdabcdab"), chunk(""))),
                     ),
                     h(
                         h(h(chunk("ee"), chunk("4444444444444444")), h(chunk("33221100"), chunk(""))),
                         h(h(chunk("ff"), chunk("5555555555555555")), h(chunk("77665544"), chunk(""))),
                     ),
                 )
             ),
             h(  # G and padding
                 h(
                     h(h(chunk("adde"), h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03000000"))),
                       h(chunk("11"), chunk(""))),
                     h(h(chunk("efbe"), h(merge(chunk("040005000600"), zero_hashes[0:6]), chunk("03000000"))),
                       h(chunk("22"), chunk(""))),
                 ),
                 chunk("")
             )
         )
     ), {
        'A': 0xaabb,
        'B': [0x1122, 0x3344],
        'C': 0xff,
        'D': list(b"foobar"),
        'E': {'A': 0xabcd, 'B': [1, 2, 3], 'C': 0xff},
        'F': (
            {'A': 0xcc, 'B': 0x4242424242424242, 'C': 0x13371337},
            {'A': 0xdd, 'B': 0x3333333333333333, 'C': 0xabcdabcd},
            {'A': 0xee, 'B': 0x4444444444444444, 'C': 0x00112233},
            {'A': 0xff, 'B': 0x5555555555555555, 'C': 0x44556677},
        ),
        'G': (
            {'A': 0xdead, 'B': [1, 2, 3], 'C': 0x11},
            {'A': 0xbeef, 'B': [4, 5, 6], 'C': 0x22},
        ),
     }),
    ("progressiveComplexTestStruct", ProgressiveComplexTestStruct,
     ProgressiveComplexTestStruct(
         A=0xab,
         B=List[uint16, 123](0x1122, 0x3344),
         C=ProgressiveBitlist(1, 0, 0, 1, 0),
         D=ProgressiveList[uint64](0x4242424242424242, 0x3333333333333333),
         E=ProgressiveList[SmallTestStruct](
             SmallTestStruct(A=0x4567, B=0x0123),
             SmallTestStruct(A=0x89ab, B=0xcdef)
         ),
         F=ProgressiveList[ProgressiveList[VarTestStruct]](
             ProgressiveList[VarTestStruct](
                 VarTestStruct(A=0x123, B=List[uint16, 1024](1, 2, 3), C=0x12),
                 VarTestStruct(A=0x456, B=List[uint16, 1024](4, 5, 6), C=0x45),
                 VarTestStruct(A=0x789, B=List[uint16, 1024](7, 8, 9), C=0x78),
             )
         ),
         G=List[ProgressiveSingleFieldContainerTestStruct, 10](
             ProgressiveSingleFieldContainerTestStruct(),
             ProgressiveSingleFieldContainerTestStruct(A=0x00),
             ProgressiveSingleFieldContainerTestStruct(A=0x42),
         ),
         H=ProgressiveList[ProgressiveVarTestStruct](
             ProgressiveVarTestStruct(A=0xab, B=List[uint16, 123](0x1122, 0x3344), C=ProgressiveBitlist(1, 0, 0, 1, 0)),
         )
     ),
     "ab"
     "1d0000002100000022000000320000003a0000007100000074000000"
     "22114433"
     "29"
     "42424242424242423333333333333333"
     "67452301ab89efcd"
     "04000000"  # offset of F[0]
     "0c000000"  # offset of F[0][0]
     "19000000"  # offset of F[0][1]
     "26000000"  # offset of F[0][2]
     "23010700000012010002000300"  # F[0][0]
     "56040700000045040005000600"  # F[0][1]
     "89070700000078070008000900"  # F[0][2]
     "000042"
     "04000000"  # offset of H[0]
     "ab090000000d0000002211443329",
     h(
         h(
             h(
                 h(
                     h(chunk(""), merge(
                         h(  # H
                             h(
                                 zero_hashes[0],  # terminator
                                 h(
                                     h(h(chunk(""), h(
                                         h(
                                             zero_hashes[0],
                                             h(merge(chunk("22114433"), zero_hashes[0:3]), chunk("02"))
                                         ),
                                         h(
                                             zero_hashes[0],
                                             h(h(chunk(""), chunk("09")), chunk("05"))
                                         )
                                     )), chunk("ab")),
                                     chunk("15")
                                 )
                             ),
                             chunk("01")  # length mix in
                         ), zero_hashes[0:6])),
                     h(  # D, E, F, G
                         h(
                             h(zero_hashes[1], h(zero_hashes[0], h(  # D
                                 h(zero_hashes[0], chunk("42424242424242423333333333333333")),
                                 chunk("02")  # length mix in
                             ))),
                             h(zero_hashes[1], h(zero_hashes[0], h(  # E
                                 h(
                                     h(
                                         zero_hashes[0],  # terminator
                                         merge(h(chunk("ab89"), chunk("efcd")), zero_hashes[0:2])
                                     ),
                                     h(chunk("6745"), chunk("2301"))
                                 ),
                                 chunk("02")  # length mix in
                             )))
                         ),
                         h(
                             h(h(h(  # F
                                 h(
                                     zero_hashes[0],  # terminator
                                     h(
                                         h(
                                             h(
                                                 zero_hashes[0],  # terminator
                                                 h(
                                                     h(
                                                         h(
                                                             h(
                                                                 chunk("5604"),
                                                                 h(merge(chunk("040005000600"), zero_hashes[0:6]), chunk("03"))
                                                             ),
                                                             h(
                                                                 chunk("45"),
                                                                 zero_hashes[0]
                                                             )
                                                         ),
                                                         h(
                                                             h(
                                                                 chunk("8907"),
                                                                 h(merge(chunk("070008000900"), zero_hashes[0:6]), chunk("03"))
                                                             ),
                                                             h(
                                                                 chunk("78"),
                                                                 zero_hashes[0]
                                                             )
                                                         )
                                                     ),
                                                     zero_hashes[1]
                                                 )
                                             ),
                                             h(
                                                 h(
                                                     chunk("2301"),
                                                     h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03"))
                                                 ),
                                                 h(
                                                     chunk("12"),
                                                     zero_hashes[0]
                                                 )
                                             )
                                         ),
                                         chunk("03")  # length mix in
                                     )
                                 ),
                                 chunk("01")  # length mix in
                             ), zero_hashes[0]), zero_hashes[1]),
                             h(zero_hashes[1], h(zero_hashes[0], h(  # G
                                 merge(h(
                                     h(
                                         h(h(chunk(""), chunk("00")), chunk("01")),
                                         h(h(chunk(""), chunk("00")), chunk("01"))
                                     ),
                                     h(
                                         h(h(chunk(""), chunk("42")), chunk("01")),
                                         zero_hashes[0]
                                     )
                                 ), zero_hashes[2:4]),
                                 chunk("03")  # length mix in
                             )))
                         )
                     )
                 ),
                 h(  # B, C
                     h(zero_hashes[0], h(  # B
                         merge(chunk("22114433"), zero_hashes[0:3]),
                         chunk("02")  # length mix in
                     )),
                     h(zero_hashes[0], h( # C
                         h(chunk(""), chunk("09")),
                         chunk("05")  # length mix in
                     ))
                 )
             ),
             chunk("ab")  # A
         ),
         chunk("153130")
     ), {
         'A': 0xab,
         'B': [0x1122, 0x3344],
         'C': "0x29",
         'D': [0x4242424242424242, 0x3333333333333333],
         'E': [{'A': 0x4567, 'B': 0x0123}, {'A': 0x89ab, 'B': 0xcdef}],
         'F': [
             [
                 {'A': 0x123, 'B': [1, 2, 3], 'C': 0x12},
                 {'A': 0x456, 'B': [4, 5, 6], 'C': 0x45},
                 {'A': 0x789, 'B': [7, 8, 9], 'C': 0x78}
             ],
         ],
         'G': [
             {'A': 0x00},
             {'A': 0x00},
             {'A': 0x42},
         ],
         'H': [
             {'A': 0xab, 'B': [0x1122, 0x3344], 'C': "0x29"},
         ],
     }),
    ("progressiveList 0", ProgressiveList[uint16],
     ProgressiveList[uint16](),
     "",
     h(
         zero_hashes[0],  # terminator
         chunk("00")  # length mix in
     ), []),
    ("progressiveList 1", ProgressiveList[uint16],
     ProgressiveList[uint16](0x0100, 0x0101),
     "00010101",
     h(
         h(
             zero_hashes[0],  # terminator
             chunk("00010101")
         ),
         chunk("02")  # length mix in
     ), [0x0100, 0x0101]),
    ("progressiveList 2", ProgressiveList[uint16],
     ProgressiveList[uint16](
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200
     ),
     "00010101020103010401050106010701080109010a010b010c010d010e010f01"
     "0002",
     h(
         h(
             h(
                 zero_hashes[0],  # terminator
                 merge(chunk("0002"), zero_hashes[0:2])
             ),
             chunk("00010101020103010401050106010701080109010a010b010c010d010e010f01")
         ),
         chunk("11")  # length mix in
     ), [
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200
     ]),
    ("progressiveList 3", ProgressiveList[uint16],
     ProgressiveList[uint16](
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
         0x0208, 0x0209, 0x020a, 0x020b, 0x020c, 0x020d, 0x020e, 0x020f,
         0x0210, 0x0211, 0x0212, 0x0213, 0x0214, 0x0215, 0x0216, 0x0217,
         0x0218, 0x0219, 0x021a, 0x021b, 0x021c, 0x021d, 0x021e, 0x021f,
         0x0220, 0x0221, 0x0222, 0x0223, 0x0224, 0x0225, 0x0226, 0x0227,
         0x0228, 0x0229, 0x022a, 0x022b, 0x022c, 0x022d, 0x022e, 0x022f,
         0x0230, 0x0231, 0x0232, 0x0233, 0x0234, 0x0235, 0x0236, 0x0237,
         0x0238, 0x0239, 0x023a, 0x023b, 0x023c, 0x023d, 0x023e, 0x023f
     ),
     "00010101020103010401050106010701080109010a010b010c010d010e010f01"
     "00020102020203020402050206020702080209020a020b020c020d020e020f02"
     "10021102120213021402150216021702180219021a021b021c021d021e021f02"
     "20022102220223022402250226022702280229022a022b022c022d022e022f02"
     "30023102320233023402350236023702380239023a023b023c023d023e023f02",
     h(
         h(
             h(
                 zero_hashes[0],  # terminator
                 h(
                     h(
                         chunk("00020102020203020402050206020702080209020a020b020c020d020e020f02"),
                         chunk("10021102120213021402150216021702180219021a021b021c021d021e021f02")
                     ),
                     h(
                         chunk("20022102220223022402250226022702280229022a022b022c022d022e022f02"),
                         chunk("30023102320233023402350236023702380239023a023b023c023d023e023f02")
                     )
                 )
             ),
             chunk("00010101020103010401050106010701080109010a010b010c010d010e010f01")
         ),
         chunk("50")  # length mix in
     ), [
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
         0x0208, 0x0209, 0x020a, 0x020b, 0x020c, 0x020d, 0x020e, 0x020f,
         0x0210, 0x0211, 0x0212, 0x0213, 0x0214, 0x0215, 0x0216, 0x0217,
         0x0218, 0x0219, 0x021a, 0x021b, 0x021c, 0x021d, 0x021e, 0x021f,
         0x0220, 0x0221, 0x0222, 0x0223, 0x0224, 0x0225, 0x0226, 0x0227,
         0x0228, 0x0229, 0x022a, 0x022b, 0x022c, 0x022d, 0x022e, 0x022f,
         0x0230, 0x0231, 0x0232, 0x0233, 0x0234, 0x0235, 0x0236, 0x0237,
         0x0238, 0x0239, 0x023a, 0x023b, 0x023c, 0x023d, 0x023e, 0x023f
     ]),
    ("progressiveList 4", ProgressiveList[uint16],
     ProgressiveList[uint16](
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
         0x0208, 0x0209, 0x020a, 0x020b, 0x020c, 0x020d, 0x020e, 0x020f,
         0x0210, 0x0211, 0x0212, 0x0213, 0x0214, 0x0215, 0x0216, 0x0217,
         0x0218, 0x0219, 0x021a, 0x021b, 0x021c, 0x021d, 0x021e, 0x021f,
         0x0220, 0x0221, 0x0222, 0x0223, 0x0224, 0x0225, 0x0226, 0x0227,
         0x0228, 0x0229, 0x022a, 0x022b, 0x022c, 0x022d, 0x022e, 0x022f,
         0x0230, 0x0231, 0x0232, 0x0233, 0x0234, 0x0235, 0x0236, 0x0237,
         0x0238, 0x0239, 0x023a, 0x023b, 0x023c, 0x023d, 0x023e, 0x023f,
         0x0300
     ),
     "00010101020103010401050106010701080109010a010b010c010d010e010f01"
     "00020102020203020402050206020702080209020a020b020c020d020e020f02"
     "10021102120213021402150216021702180219021a021b021c021d021e021f02"
     "20022102220223022402250226022702280229022a022b022c022d022e022f02"
     "30023102320233023402350236023702380239023a023b023c023d023e023f02"
     "0003",
     h(
         h(
             h(
                 h(
                     zero_hashes[0],  # terminator
                     merge(chunk("0003"), zero_hashes[0:4])
                 ),
                 h(
                     h(
                         chunk("00020102020203020402050206020702080209020a020b020c020d020e020f02"),
                         chunk("10021102120213021402150216021702180219021a021b021c021d021e021f02")
                     ),
                     h(
                         chunk("20022102220223022402250226022702280229022a022b022c022d022e022f02"),
                         chunk("30023102320233023402350236023702380239023a023b023c023d023e023f02")
                     )
                 )
             ),
             chunk("00010101020103010401050106010701080109010a010b010c010d010e010f01")
         ),
         chunk("51")  # length mix in
     ), [
         0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
         0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
         0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
         0x0208, 0x0209, 0x020a, 0x020b, 0x020c, 0x020d, 0x020e, 0x020f,
         0x0210, 0x0211, 0x0212, 0x0213, 0x0214, 0x0215, 0x0216, 0x0217,
         0x0218, 0x0219, 0x021a, 0x021b, 0x021c, 0x021d, 0x021e, 0x021f,
         0x0220, 0x0221, 0x0222, 0x0223, 0x0224, 0x0225, 0x0226, 0x0227,
         0x0228, 0x0229, 0x022a, 0x022b, 0x022c, 0x022d, 0x022e, 0x022f,
         0x0230, 0x0231, 0x0232, 0x0233, 0x0234, 0x0235, 0x0236, 0x0237,
         0x0238, 0x0239, 0x023a, 0x023b, 0x023c, 0x023d, 0x023e, 0x023f,
         0x0300
     ]),
    ("progressiveList 5", ProgressiveList[SingleFieldTestStruct],
     ProgressiveList[SingleFieldTestStruct](),
     "",
     h(
         zero_hashes[0],  # terminator
         chunk("00")  # length mix in
     ), []),
    ("progressiveList 6", ProgressiveList[SingleFieldTestStruct],
     ProgressiveList[SingleFieldTestStruct](
         SingleFieldTestStruct(A=1),
     ),
     "01",
     h(
         h(
             zero_hashes[0],  # terminator
             chunk("01")
         ),
         chunk("01")  # length mix in
     ), [{'A': 0x01}]),
    ("progressiveList 7", ProgressiveList[SingleFieldTestStruct],
     ProgressiveList[SingleFieldTestStruct](
         SingleFieldTestStruct(A=1),
         SingleFieldTestStruct(A=2),
     ),
     "0102",
     h(
         h(
             h(
                 zero_hashes[0],  # terminator
                 merge(chunk("02"), zero_hashes[0:2])
             ),
             chunk("01")
         ),
         chunk("02")  # length mix in
     ), [{'A': 0x01}, {'A': 0x02}]),
    ("progressiveList 8", ProgressiveList[SingleFieldTestStruct],
     ProgressiveList[SingleFieldTestStruct](
         SingleFieldTestStruct(A=1),
         SingleFieldTestStruct(A=2),
         SingleFieldTestStruct(A=3),
         SingleFieldTestStruct(A=4),
         SingleFieldTestStruct(A=5),
     ),
     "0102030405",
     h(
         h(
             h(
                 zero_hashes[0],  # terminator
                 h(
                     h(
                         chunk("02"),
                         chunk("03")
                     ),
                     h(
                         chunk("04"),
                         chunk("05")
                     )
                 )
             ),
             chunk("01")
         ),
         chunk("05")  # length mix in
     ), [{'A': 0x01}, {'A': 0x02}, {'A': 0x03}, {'A': 0x04}, {'A': 0x05}]),
    ("progressiveTestStruct", ProgressiveTestStruct,
     ProgressiveTestStruct(
         A=ProgressiveList[byte](b"foobar"),
         B=ProgressiveList[uint64](1, 2, 3, 4, 5),
         C=ProgressiveList[SmallTestStruct](
             SmallTestStruct(A=0x13, B=0x37),
             SmallTestStruct(A=0x420, B=0x69),
         ),
         D=ProgressiveList[ProgressiveList[VarTestStruct]](
             ProgressiveList[VarTestStruct](
                 VarTestStruct(A=0x123, B=List[uint16, 1024](1, 2, 3), C=0x12),
                 VarTestStruct(A=0x456, B=List[uint16, 1024](4, 5, 6), C=0x45),
                 VarTestStruct(A=0x789, B=List[uint16, 1024](7, 8, 9), C=0x78),
             ),
             ProgressiveList[VarTestStruct](
                 VarTestStruct(A=0x123, B=List[uint16, 1024](1, 2, 3), C=0x12),
                 VarTestStruct(A=0x456, B=List[uint16, 1024](4, 5, 6), C=0x45),
             ),
             ProgressiveList[VarTestStruct](
                 VarTestStruct(A=0x123, B=List[uint16, 1024](1, 2, 3), C=0x12),
             ),
             ProgressiveList[VarTestStruct](),
         ),
     ),
     "10000000"  # offset of A
     "16000000"  # offset of B
     "3e000000"  # offset of C
     "46000000"  # offset of D
     "666f6f626172"  # foobar
     "01000000000000000200000000000000030000000000000004000000000000000500000000000000"  # B
     "1300370020046900"  # C
     "10000000"  # offset of D[0]
     "43000000"  # offset of D[1]
     "65000000"  # offset of D[2]
     "76000000"  # offset of D[3]
     "0c000000"  # offset of D[0][0]
     "19000000"  # offset of D[0][1]
     "26000000"  # offset of D[0][2]
     "23010700000012010002000300"  # D[0][0]
     "56040700000045040005000600"  # D[0][1]
     "89070700000078070008000900"  # D[0][2]
     "08000000"  # offset of D[1][0]
     "15000000"  # offset of D[1][1]
     "23010700000012010002000300"  # D[1][0]
     "56040700000045040005000600"  # D[1][1]
     "04000000"  # offset of D[2][0]
     "23010700000012010002000300",  # D[2][0]
     h(
         h(
             h(h(zero_hashes[0], chunk("666f6f626172")), chunk("06")),
             h(
                 h(
                     h(
                         zero_hashes[0],  # terminator
                         merge(chunk("0500000000000000"), zero_hashes[0:2])
                     ),
                     chunk("0100000000000000020000000000000003000000000000000400000000000000")
                 ),
                 chunk("05")  # length mix in
             )
         ),
         h(
             h(
                 h(
                     h(
                         zero_hashes[0],  # terminator
                         merge(h(chunk("2004"), chunk("6900")), zero_hashes[0:2])
                     ),
                     h(chunk("1300"), chunk("3700"))
                 ),
                 chunk("02")  # length mix in
             ),
             h(
                 h(
                     h(
                         zero_hashes[0],  # terminator
                         h(
                             h(
                                 h(
                                     h(
                                         h(
                                             zero_hashes[0],  # terminator
                                             h(
                                                 h(
                                                     h(
                                                         h(
                                                             chunk("5604"),
                                                             h(merge(chunk("040005000600"), zero_hashes[0:6]), chunk("03"))
                                                         ),
                                                         h(
                                                             chunk("45"),
                                                             zero_hashes[0]
                                                         )
                                                     ),
                                                     zero_hashes[0]
                                                 ),
                                                 zero_hashes[1]
                                             )
                                         ),
                                         h(
                                             h(
                                                 chunk("2301"),
                                                 h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03")),
                                             ),
                                             h(
                                                 chunk("12"),
                                                 zero_hashes[0]
                                             )
                                         )
                                     ),
                                     chunk("02")  # length mix in
                                 ),
                                 h(
                                     h(
                                         zero_hashes[0],  # terminator
                                         h(
                                             h(
                                                 chunk("2301"),
                                                 h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03")),
                                             ),
                                             h(
                                                 chunk("12"),
                                                 zero_hashes[0]
                                             )
                                         )
                                     ),
                                     chunk("01")  # length mix in
                                 ),
                             ),
                             h(
                                 h(
                                     zero_hashes[0],  # terminator
                                     chunk("00")  # length mix in
                                 ),
                                 zero_hashes[0]
                             )
                         )
                     ),
                     h(
                         h(
                             h(
                                 zero_hashes[0],  # terminator
                                 h(
                                     h(
                                         h(
                                             h(
                                                 chunk("5604"),
                                                 h(merge(chunk("040005000600"), zero_hashes[0:6]), chunk("03"))
                                             ),
                                             h(
                                                 chunk("45"),
                                                 zero_hashes[0]
                                             )
                                         ),
                                         h(
                                             h(
                                                 chunk("8907"),
                                                 h(merge(chunk("070008000900"), zero_hashes[0:6]), chunk("03"))
                                             ),
                                             h(
                                                 chunk("78"),
                                                 zero_hashes[0]
                                             )
                                         )
                                     ),
                                     zero_hashes[1]
                                 )
                             ),
                             h(
                                 h(
                                     chunk("2301"),
                                     h(merge(chunk("010002000300"), zero_hashes[0:6]), chunk("03"))
                                 ),
                                 h(
                                     chunk("12"),
                                     zero_hashes[0]
                                 )
                             )
                         ),
                         chunk("03")  # length mix in
                     ),
                 ),
                 chunk("04")  # length mix in
             )
         )
     ), {
         'A': list(b"foobar"),
         'B': [1, 2, 3, 4, 5],
         'C': [{'A': 0x13, 'B': 0x37}, {'A': 0x420, 'B': 0x69}],
         'D': [
             [
                 {'A': 0x123, 'B': [1, 2, 3], 'C': 0x12},
                 {'A': 0x456, 'B': [4, 5, 6], 'C': 0x45},
                 {'A': 0x789, 'B': [7, 8, 9], 'C': 0x78},
             ],
             [
                 {'A': 0x123, 'B': [1, 2, 3], 'C': 0x12},
                 {'A': 0x456, 'B': [4, 5, 6], 'C': 0x45},
             ],
             [
                 {'A': 0x123, 'B': [1, 2, 3], 'C': 0x12},
             ],
             [],
         ],
     }),
]


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_to_obj(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    assert value.to_obj() == obj


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_from_obj(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    assert typ.from_obj(obj) == value


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_json_dump(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    assert json.dumps(value.to_obj()) == json.dumps(obj)


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_json_load(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    # Bigger round trip: check if a json-like obj can be parsed correctly.
    assert value.from_obj(json.loads(json.dumps(obj))).to_obj() == obj


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_type_bounds(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    byte_len = len(bytes.fromhex(serialized))
    assert typ.min_byte_length() <= byte_len <= typ.max_byte_length()
    if typ.is_fixed_byte_length():
        assert byte_len == typ.type_byte_length()


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_value_byte_length(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    assert value.value_byte_length() == len(bytes.fromhex(serialized))


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_serialize(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    stream = io.BytesIO()
    length = value.serialize(stream)
    stream.seek(0)
    encoded = stream.read()
    assert encoded.hex() == serialized
    assert length*2 == len(serialized)


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_encode_bytes(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    encoded = value.encode_bytes()
    assert encoded.hex() == serialized


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_hash_tree_root(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    assert value.hash_tree_root().hex() == root


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_deserialize(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    stream = io.BytesIO()
    bytez = bytes.fromhex(serialized)
    stream.write(bytez)
    stream.seek(0)
    decoded = typ.deserialize(stream, len(bytez))
    assert decoded == value


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_decode_bytes(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    bytez = bytes.fromhex(serialized)
    decoded = typ.decode_bytes(bytez)
    assert decoded == value


@pytest.mark.parametrize("name, typ, value, serialized, root, obj", test_data)
def test_readonly_iters(name: str, typ: Type[View], value: View, serialized: str, root: str, obj: ObjType):
    if hasattr(value, 'readonly_iter'):
        r_iter = value.readonly_iter()
        i = 0
        for expected_elem in iter(value):
            got_elem = r_iter.__next__()
            assert expected_elem == got_elem
            i += 1
        try:
            r_iter.__next__()
            assert False
        except StopIteration:
            pass
    if isinstance(value, Container) or isinstance(value, ProgressiveContainer):
        fields = list(value)
        expected = [getattr(value, fkey) for fkey in value.__class__.fields().keys()]
        assert fields == expected


@pytest.mark.parametrize("base_typs", [
    (Container, Container),
    (ProgressiveContainer(active_fields=[1, 1, 1]), ProgressiveContainer(active_fields=[1, 1, 1, 1])),
    (ProgressiveContainer(active_fields=[1, 1, 0, 0, 0, 1]), ProgressiveContainer(active_fields=[0, 1, 1, 1, 1])),
])
def test_container_equality(base_typs: PyList[Type[View]]):
    class A(base_typs[0]):
        x: uint8
        y: uint8
        z: uint8

    class B(base_typs[1]):
        x: uint8
        y: uint8
        z: uint8
        v: uint8

    assert A(x=1, y=2, z=3) == A(x=1, y=2, z=3)
    assert B(x=1, y=2, z=3, v=0) == B(x=1, y=2, z=3, v=0)
    assert A(x=1, y=2, z=3) != B(x=1, y=2, z=3, v=0)
    assert A(x=1, y=2, z=3) in {A(x=1, y=2, z=3)}
    assert A(x=1, y=2, z=3) not in {B(x=1, y=2, z=3, v=0)}

    class ContainerA(Container):
        x: uint8
        y: uint8
        z: uint8

    class ContainerB(Container):
        x: uint8
        y: uint8
        z: uint8
        v: uint8

    assert A().encode_bytes() == ContainerA().encode_bytes()
    assert B().encode_bytes() == ContainerB().encode_bytes()
    assert A(x=1, y=2, z=3).encode_bytes() == ContainerA(x=1, y=2, z=3).encode_bytes()
    assert B(x=1, y=2, z=3, v=4).encode_bytes() == ContainerB(x=1, y=2, z=3, v=4).encode_bytes()


def test_progressive():
    class X(Container):
        A: ProgressiveList[SingleFieldTestStruct]
        B: ProgressiveList[uint16]

    x = X(
        A=ProgressiveList[SingleFieldTestStruct](
            SingleFieldTestStruct(A=1),
            SingleFieldTestStruct(A=2),
        ),
        B=ProgressiveList[uint16](0x0100, 0x0101),
    )
    x_bytes = bytes.fromhex("080000000a000000010200010101")
    assert x.encode_bytes() == x_bytes
