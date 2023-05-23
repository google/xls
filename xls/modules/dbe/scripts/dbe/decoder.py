from .common import (
    Params,
    Token,
    TokCP,
    TokLT,
)
from typing import Sequence, List


class Decoder:
    def __init__(self, cfg: Params):
        self.cfg = cfg
        self.hb = []
        self.toks = []
        self.out = []

    def _pout(self, sym):
        self.out.append(sym)
        self.hb.append(sym)
        if len(self.hb) >= self.cfg.hb_maxsz:
            self.hb = self.hb[-self.cfg.hb_maxsz:]

    def feed(self, tok):
        assert isinstance(tok, Token)
        assert tok.cfg == self.cfg

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
        else:
            assert False, "unknown token type"
    
    def decode(self, toks: Sequence[Token]) -> List[int]:
        for t in toks:
            self.feed(t)
        return self.out
