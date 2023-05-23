#!/usr/bin/env python3

import random
from dbe import prettify, get_random_tokens, Params, Decoder


SYM_BITS = 4
PTR_BITS = 3
CNT_BITS = 4

OFSSLOPE = 2
CNTSLOPE = 4
NTOKS = 32
NPERLINE = 16


random.seed(0)

cfg = Params(SYM_BITS, PTR_BITS, CNT_BITS)
toks = get_random_tokens(NTOKS, cfg, OFSSLOPE, CNTSLOPE)
d = Decoder(cfg)
for t in toks:
    d.feed(t)

print(f'Tokens (len={len(d.toks)}):')
print(',\n'.join(repr(t) for t in d.toks))
print('')
print(f'Reference decoding (len={len(d.out)}):')
print(prettify((f'{x:2d}' for x in d.out), NPERLINE))
