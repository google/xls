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


class Token:
    def __init__(self, cfg: Params, last: bool):
        self.cfg = cfg
        self.last = last


class TokLT(Token):
    def __init__(self, cfg: Params, last: bool, sym: int):
        super().__init__(cfg, last)
        self.sym = sym
    
    def __repr__(self) -> str:
        pars = [
            f'kind: TokenKind::LT',
            f'last: {"true" if self.last else "false"}',
            f'lt_sym: uN[{self.cfg.wsym}]:{self.sym:2d}',
            f'cp_off: uN[{self.cfg.wptr}]:0',
            f'cp_cnt: uN[{self.cfg.wcnt}]:0'
        ]
        return f'Token{{{", ".join(pars)}}}'


class TokCP(Token):
    def __init__(self, cfg: Params, last: bool, ofs: int, cnt: int):
        super().__init__(cfg, last)
        self.ofs = ofs
        self.cnt = cnt
    
    def __repr__(self) -> str:
        assert self.cnt >= 1
        pars = [
            f'kind: TokenKind::CP',
            f'last: {"true" if self.last else "false"}',
            f'lt_sym: uN[{self.cfg.wsym}]:0',
            f'cp_off: uN[{self.cfg.wptr}]:{self.ofs:2d}',
            f'cp_cnt: uN[{self.cfg.wcnt}]:{self.cnt-1:2d}'
        ]
        return f'Token{{{", ".join(pars)}}}'
