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


from typing import List
import random
from dbe.common import Params, Token, TokLT, TokCP, TokMK, Mark


def get_random_tokens(seqlen: int, cfg: Params, ofsslope: float = 2, cntslope: float = 4) -> List[Token]:
    toks = []
    nsym = 0

    def emit(t: Token):
        nonlocal nsym
        if isinstance(t, TokLT):
            nsym += 1
        elif isinstance(t, TokCP):
            nsym += t.cnt
        toks.append(t)

    emit(TokLT(random.randint(0, cfg.sym_max)))
    # Then we generate CP or LIT with equal probability and fill their contents randomly
    while len(toks) < (seqlen - 1):
        kind = random.randint(0, 1)
        if kind:
            # Literal
            emit(TokLT(random.randint(0, cfg.sym_max)))
        else:
            # CP
            # exp-like distribution for number of repetitions
            rx = random.random()
            cnt = int(1.5 + (cfg.cnt_max - 1) * (rx ** cntslope))
            cnt = min(max(cnt, 1), cfg.cnt_max)
            # exp-like distribution for offset
            assert nsym >= 1
            rx = random.random()
            ofs = int(0.5 + (cfg.hb_maxsz - 1) * (rx ** ofsslope))
            ofs = min(max(ofs, 0), nsym - 1)
            emit(TokCP(ofs, cnt))
    emit(TokMK(Mark.END))

    return toks
