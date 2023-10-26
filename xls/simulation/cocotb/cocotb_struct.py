from cocotb.binary import BinaryValue
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
        return b

    # Convenience methods
    @staticmethod
    def to_int_list(values: List[Any]) -> List[int]:
        return [int(x) for x in values]

    @classmethod
    def to_binary_list(cls, values: List[Any]) -> List[BinaryValue]:
        return [BinaryValue(int(x), cls._structlen()) for x in values]
