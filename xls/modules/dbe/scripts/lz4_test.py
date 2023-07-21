#!/usr/bin/env python3

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


import logging
import sys
import argparse
from dbe import Lz4Params, Lz4Encoder, Decoder, estimate_lz4_size, TokLT,\
    TokCP, TokMK, Mark
from dbe_data import DataFiles


def test(maxlen: int = -1, blocklen: int = -1, verbose: bool = False):
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)

    # Load source text
    with open(DataFiles.DICKENS_TXT, "rb") as f:
        syms_ref = list(f.read(maxlen if maxlen >= 0 else None))
    logging.info(f"Data length: {len(syms_ref)}")

    # LZ4 params
    cfg = Lz4Params(8, 16, 16, 13)
    enc = Lz4Encoder(cfg, verbose=verbose)
    dec = Decoder(cfg)

    if blocklen < 1:
        blocklen = max(1, len(syms_ref))
    
    toks = []
    syms_dec = []
    nblocks = 0
    for boff in range(0, max(len(syms_ref), 1), blocklen):
        block_syms = syms_ref[boff:boff + blocklen] + [Mark.END]
        block_toks = enc.encode_block(block_syms)
        block_dec = dec.decode(block_toks)
        assert isinstance(block_toks[-1], TokMK)\
            and block_toks[-1].mark == Mark.END
        assert isinstance(block_dec[-1], Mark)\
            and block_dec[-1] == Mark.END
        # Remove END markers
        toks.extend(block_toks[:-1])
        syms_dec.extend(block_dec[:-1])
        nblocks += 1

    for i, s in enumerate(syms_dec):
        logging.debug(f'DEC @ {i}: {s}')

    for i, s in enumerate(syms_ref):
        logging.debug(f'REF @ {i}: {s}')

    for i in range(max(len(syms_ref), len(syms_dec))):
        assert i < len(syms_ref), (
            f"Extra chars in decoded text: "
            f"ref_len={len(syms_ref)} dec_len={len(syms_dec)}"
        )
        assert i < len(syms_dec), (
            f"Preliminary EOF in decoded text: "
            f"ref_len={len(syms_ref)} dec_len={len(syms_dec)}"
        )
        sr = syms_ref[i]
        sd = syms_dec[i]
        assert sr == sd, f"Mismatch @ {i}: expected {sr!r} got {sd!r}"

    logging.info("Token statistics:")
    nlit = sum(1 for _ in filter(lambda t: isinstance(t, TokLT), toks))
    ncp = sum(1 for _ in filter(lambda t: isinstance(t, TokCP), toks))
    ncplen = sum(t.cnt for t in filter(lambda t: isinstance(t, TokCP), toks))
    ntot = nlit + ncplen
    assert ntot == len(syms_ref)
    lz4_sz = estimate_lz4_size(toks)
    ratio = ntot / max(lz4_sz, 1)

    logging.info(f"TOTAL SYMBOLS:          {nlit + ncplen}")
    logging.info(f"NUMBER OF BLOCKS:       {nblocks}")
    logging.info(f"NUM OF LITERALS:        {nlit}")
    logging.info(f"NUM OF CP:              {ncp}")
    logging.info(f"CP-ENCODED SYMBOLS:     {ncplen}")
    logging.info(f"LZ4 ESTIM SIZE:         {lz4_sz}")
    logging.info(f"RATIO (overestim.):     {ratio:.2f}")
    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Lz4 test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--maxlen", type=int, default=100000)
    parser.add_argument("-b", "--blocklen", type=int, default=10000)
    args = parser.parse_args()
    test(args.maxlen, args.blocklen, args.verbose)


if __name__ == "__main__":
    main()
