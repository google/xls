# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from cocotb.binary import BinaryValue
from dbe import MarkedByte, TokMK, TokCP, TokLT, Token, Mark
import enum
from typing import List, Tuple, Mapping, Any, Type


class Struct:
    _STRUCT_FIELDS_ = []

    @classmethod
    def from_int(cls, v: int):
        return cls(v)

    @classmethod
    def from_fields(cls, *fields_seq, **fields_dict):
        if fields_seq:
            return cls(cls._fields_to_int(fields_seq))
        else:
            fields = cls._fields_dict_to_list(fields_dict)
            return cls(cls._fields_to_int(fields))

    @classmethod
    def from_binary(cls, bin: BinaryValue):
        if bin.n_bits != cls._structlen():
            raise ValueError(
                f'Given {bin.n_bits} bits value, expect {cls._structlen()}')
        return cls.from_int(bin.integer)

    @classmethod
    def _fields(cls) -> List[Tuple[str, int, Type]]:
        return list(reversed(cls._STRUCT_FIELDS_))

    @classmethod
    def _structlen(cls) -> int:
        return sum(d[1] for d in cls._fields())

    @classmethod
    def _masks(cls) -> List[Tuple[int, int]]:
        r = []
        off = 0
        for f in cls._fields():
            assert f[1] >= 1, f"Field {f} width should be at least 1"
            mask = (1 << f[1]) - 1
            r.append((off, mask))
            off += f[1]
        return r

    @classmethod
    def _int_to_fields(cls, v: int) -> List[Any]:
        defs = cls._fields()
        masks = cls._masks()
        r = []
        for d, m in zip(defs, masks):
            t = d[2]
            r.append(t((v >> m[0]) & m[1]))
        return r

    @classmethod
    def _fields_to_int(cls, fields: List[Any]) -> int:
        defs = cls._fields()
        masks = cls._masks()
        if len(fields) != len(defs):
            raise RuntimeError(
                f"Need {len(defs)} fields, passed: {fields}")
        r = 0
        for f, d, m in zip(fields, defs, masks):
            i = int(f)
            # TODO: signed ints will need special handling
            if (i & m[1]) != i:
                raise ValueError(
                    f'Field {d[0]} was given value {f!r} (int {i}) that is out of range')
            r |= (i & m[1]) << m[0]
        return r

    @classmethod
    def _fields_dict_to_list(cls, dct: Mapping[str, Any]) -> List[Any]:
        r = []
        for d in cls._fields():
            r.append(dct[d[0]])
        return r

    def __init__(self, v: int):
        fields = self._int_to_fields(v)
        for f, d in zip(fields, self._fields()):
            setattr(self, d[0], d[2](f))

    def __int__(self) -> int:
        return self._fields_to_int(self._fields_dict_to_list(self.__dict__))
    
    def __repr__(self) -> str:
        attrs = ','.join(f'{d[0]}={getattr(self, d[0])}'
                         for d in reversed(self._fields()))
        return f'{self.__class__.__name__}({attrs})'

    def as_binary(self) -> BinaryValue:
        b = BinaryValue(
            value=int(self), n_bits=self._structlen(), bigEndian=False)
        return  b

    # Convenience methods
    @staticmethod
    def to_int_list(values: List[Any]) -> List[int]:
        return [int(x) for x in values]

    @classmethod
    def to_binary_list(cls, values: List[Any]) -> List[BinaryValue]:
        return [BinaryValue(int(x), cls._structlen()) for x in values]


class DslxPlainData8(Struct):
    _STRUCT_FIELDS_ = [
        ('is_mark', 1, bool),
        ('data', 8, int),
        ('mark', 4, Mark),
    ]

    @classmethod
    def from_markedbyte(cls, value: MarkedByte):
        if isinstance(value, Mark):
            d = {
                'is_mark': True,
                'mark': value,
                'data': 0
            }
        else:
            d = {
                'is_mark': False,
                'mark': Mark.NONE,
                'data': value
            }
        return cls.from_fields(**d)

    def to_markedbyte(self) -> MarkedByte:
        if self.is_mark:
            return Mark(self.mark)
        return self.data


class DslxToken(Struct):
    class Kind(enum.IntEnum):
        LT = 0
        CP = 1
        MK = 2

    _STRUCT_FIELDS_ = [
        ('kind', 2, Kind),
        ('lt_sym', 8, int),
        ('cp_off', 16, int),
        ('cp_cnt', 16, int),
        ('mark', 4, Mark)
    ]

    @classmethod
    def from_token(cls, tok: Token):
        d = {
            'kind': cls.Kind.MK,
            'lt_sym': 0,
            'cp_off': 0,
            'cp_cnt': 0,
            'mark': Mark.NONE
        }
        if isinstance(tok, TokMK):
            d['mark'] = tok.mark
        elif isinstance(tok, TokCP):
            assert tok.cnt >= 1
            d['kind'] = cls.Kind.CP
            d['cp_off'] = tok.ofs
            d['cp_cnt'] = tok.cnt - 1
        elif isinstance(tok, TokLT):
            d['kind'] = cls.Kind.LT
            d['lt_sym'] = tok.sym
        else:
            raise TypeError(f'Unexpected token type: {tok!r} ({type(tok)})')
        return cls.from_fields(**d)

    def to_token(self) -> Token:
        if self.kind == self.Kind.LT:
            return TokLT(self.lt_sym)
        elif self.kind == self.Kind.CP:
            return TokCP(self.cp_off, self.cp_cnt + 1)
        elif self.kind == self.Kind.MK:
            return TokMK(self.mark)
        else:
            return RuntimeError(f'Decoding unsupported token kind: {self}')
