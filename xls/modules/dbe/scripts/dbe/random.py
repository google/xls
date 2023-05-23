from typing import List
import random
from dbe.common import Params, Token, TokLT, TokCP
from dbe.decoder import Decoder


def get_random_symbols(seqlen: int, symwidth: int, repmax: int = 200, repslope: float = 100) -> List[int]:
    seq = []
    vmax = (1 << symwidth) - 1

    while len(seq) < seqlen:
        sym = random.randint(0, vmax)
        # exp-like distribution for number of repetitions
        rx = random.random()  # uniform [0.0, 1.0)
        # 0.0 maps to 1
        # 1.0 maps to REPMAX
        nrep = int(1.5 + (repmax - 1) * (rx ** repslope))
        nrep = min(max(nrep, 1), repmax)
        seq += [sym] * nrep

    seq = seq[:seqlen]
    return seq


def get_random_tokens(seqlen: int, cfg: Params, ofsslope: float = 2, cntslope: float = 4) -> List[Token]:
    d = Decoder(cfg)
    d.feed(TokLT(cfg, False, random.randint(0, cfg.sym_max)))
    # Then we generate CP or LIT with equal probability and fill their contents randomly
    while len(d.toks) < seqlen:
        kind = random.randint(0, 1)
        if kind:
            # Literal
            d.feed(TokLT(cfg, False, random.randint(0, cfg.sym_max)))
        else:
            # CP
            # exp-like distribution for number of repetitions
            rx = random.random()
            cnt = int(1.5 + (cfg.cnt_max - 1) * (rx ** cntslope))
            cnt = min(max(cnt, 1), cfg.cnt_max)
            # exp-like distribution for offset
            assert len(d.hb) >= 1
            rx = random.random()
            ofs = int(0.5 + (cfg.hb_maxsz - 1) * (rx ** ofsslope))
            ofs = min(max(ofs, 0), len(d.hb) - 1)
            d.feed(TokCP(cfg, False, ofs, cnt))

    # Set 'last' on the last token
    toks = list(d.toks)
    toks[-1].last = True

    return toks
