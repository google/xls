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


from .common import (
    Params,
    Token,
    TokCP,
    TokLT,
    TokMK,
    Mark
)
from typing import Sequence, List


class Decoder:
    def __init__(self, cfg: Params):
        self.cfg = cfg
        self.hb = []
        self.toks: List[Token] = []
        self.out: List[int] = []

    def _pout(self, sym):
        self.out.append(sym)
        self.hb.append(sym)
        if len(self.hb) >= self.cfg.hb_maxsz:
            self.hb = self.hb[-self.cfg.hb_maxsz:]

    def feed(self, tok):
        assert isinstance(tok, Token)

        self.toks.append(tok)
        if isinstance(tok, TokLT):
            # Literal
            self._pout(tok.sym)
        elif isinstance(tok, TokCP):
            # CP
            for i in range(tok.cnt):
                try:
                    self._pout(self.hb[-tok.ofs-1])
                except IndexError as e:
                    print(self.out)
                    print(len(self.out))
                    raise e
        elif isinstance(tok, TokMK):
            if tok.mark == Mark.RESET:
                self.hb.clear()
            elif tok.mark == Mark.END:
                # In Python we do not handle END tokens
                ...
            else:
                raise RuntimeError(f'Unexpected marker: {tok}')
        else:
            raise RuntimeError(f'Unexpected token type: {type(tok)!r}')
    
    def decode(self, toks: Sequence[Token]) -> List[int]:
        self.toks.clear()
        self.out.clear()
        for t in toks:
            self.feed(t)
        return self.out
