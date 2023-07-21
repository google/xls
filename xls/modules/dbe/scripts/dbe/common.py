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


from abc import abstractmethod
from enum import IntEnum
from typing import Union, List, Iterable


class Params:
    def __init__(self, symbits: int, ptrbits: int, cntbits: int):
        self.wsym = symbits
        self.wptr = ptrbits
        self.wcnt = cntbits
        self.hb_maxsz = 1 << self.wptr
        self.sym_max = (1 << self.wsym) - 1
        self.cnt_max = (1 << self.wcnt)

    def __eq__(self, oth: object) -> bool:
        if not isinstance(oth, Params):
            return False
        ok = self.wsym == oth.wsym
        ok = ok and self.wptr == oth.wptr
        ok = ok and self.wcnt == oth.wcnt
        return ok


class Mark(IntEnum):
    NONE = 0
    END = 1
    RESET = 2

    _ERROR_FIRST = 8
    # Only errors have values >= __ERROR_FIRST
    ERROR_BAD_MARK = 8
    ERROR_INVAL_CP = 9

    @classmethod
    def is_error(cls, v: 'Mark') -> bool:
        return v >= cls._ERROR_FIRST


PlainData = Union[int, Mark]


def plain_data_to_dslx(b: PlainData, cfg: Params) -> str:
    is_mark = isinstance(b, Mark)
    attrs = [
        f'is_mark: {"true" if is_mark else "false"}',
        f'data: uN[{cfg.wsym}]:{0 if is_mark else b}',
        f'mark: Mark::{b.name if is_mark else "NONE"}',
    ]
    return f'PlainData{{{", ".join(attrs)}}}'



class Token:
    @abstractmethod
    def to_dslx(self, cfg: Params) -> str:
        ...


class TokLT(Token):
    """
    TokLT - literal token

    Makes decoder emit a single symbol specified by the token
    """
    def __init__(self, sym: int):
        super().__init__()
        self.sym = sym
    
    def __repr__(self) -> str:
        return f'TokLT(sym={self.sym})'
    
    def to_dslx(self, cfg: Params) -> str:
        attrs = [
            f'kind: TokenKind::LT',
            f'lt_sym: uN[{cfg.wsym}]:{self.sym:2d}',
            f'cp_off: uN[{cfg.wptr}]:0',
            f'cp_cnt: uN[{cfg.wcnt}]:0',
            f'mark: Mark::NONE',
        ]
        return f'Token{{{", ".join(attrs)}}}'


class TokCP(Token):
    """
    TokCP - copy pointer token

    Makes decoder emit a sequence of symbols that repeats the continous
    sequence of symbols that was emitted in the past. It specifies how far
    to go into the past, and how many symbols to copy.

    - 'offset' tells decoder where the beginning of past sequence if located.
    It is counted starting from the last written character, so offset of 0
    means "copy beginning with the last output character".
    - 'count' is just a number of characters to copy. Count of 0 can be
    deemed illegal by some token encoding schemes.
    """
    def __init__(self, ofs: int, cnt: int):
        super().__init__()
        self.ofs = ofs
        self.cnt = cnt

    def __repr__(self) -> str:
        return f'TokCP(ofs={self.ofs}, cnt={self.cnt})'

    def to_dslx(self, cfg: Params) -> str:
        assert self.cnt >= 1
        attrs = [
            f'kind: TokenKind::CP',
            f'lt_sym: uN[{cfg.wsym}]:0',
            f'cp_off: uN[{cfg.wptr}]:{self.ofs:2d}',
            f'cp_cnt: uN[{cfg.wcnt}]:{self.cnt-1:2d}',
            f'mark: Mark::NONE',
        ]
        return f'Token{{{", ".join(attrs)}}}'


class TokMK(Token):
    """
    TokMK - control marker token

    Contains 8-bit wide marker control code specified by Marker enum.
    """
    def __init__(self, mark: Mark):
        super().__init__()
        self.mark = mark
    
    def __repr__(self) -> str:
        return f'TokMK({self.mark!r})'
    
    def to_dslx(self, cfg: Params) -> str:
        attrs = [
            f'kind: TokenKind::MK',
            f'lt_sym: uN[{cfg.wsym}]:0',
            f'cp_off: uN[{cfg.wptr}]:0',
            f'cp_cnt: uN[{cfg.wcnt}]:0',
            f'mark: Mark::{self.mark.name}',
        ]
        return f'Token{{{", ".join(attrs)}}}'

