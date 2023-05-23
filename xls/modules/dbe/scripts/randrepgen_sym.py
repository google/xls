#!/usr/bin/env python3

import random
from dbe import prettify, get_random_symbols

BITS = 4
REPMAX = 200
REPSLOPE = 100
SEQLEN = 2048
NPERLINE = 16

random.seed(0)

seq = get_random_symbols(SEQLEN, BITS, REPMAX, REPSLOPE)

print(f'Len = {len(seq)}')
print('Seq:')
print(prettify((f'{x:2d}' for x in seq), NPERLINE))
