#!/usr/bin/env python3

import logging
import sys
import os
import argparse
from dbe import Lz4Params, Lz4Encoder, Decoder, estimate_lz4_size, TokLT, TokCP


def test(maxlen: int = 10000, verbose: bool = False):
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr)

    # Load source text
    sdir = os.path.dirname(__file__)
    with open(f"{sdir}/data/dickens.txt", "rb") as f:
        syms1 = f.read(maxlen if maxlen else None)
    text1 = syms1.decode("ascii")
    logging.info(f"Data length: {len(syms1)}")

    # LZ4 params
    cfg = Lz4Params(8, 16, 16, 13)
    lz4 = Lz4Encoder(cfg, verbose=verbose)
    toks = lz4.encode_block(syms1)

    for t in toks:
        logging.debug(t)

    for s in syms1:
        logging.debug(s)

    dec = Decoder(cfg)
    syms2 = dec.decode(toks)
    text2 = bytes(syms2).decode("ascii")

    for i in range(max(len(text1), len(text2))):
        assert i < len(text1), (
            f"Extra chars in decoded text: "
            f"l1={len(text1)} l2={len(text2)}, char: {text2[i]!r}"
        )
        assert i < len(text2), (
            f"Preliminary EOF in decoded text: "
            f"l1={len(text1)} l2={len(text2)}, char: {text1[i]!r}"
        )
        s1 = syms1[i]
        s2 = syms2[i]
        assert s1 == s2, f"Mismatch @ {i}: expected {s1!r} got {s2!r}"

    logging.info("Token statistics:")
    nlit = sum(1 for _ in filter(lambda t: isinstance(t, TokLT), toks))
    ncp = sum(1 for _ in filter(lambda t: isinstance(t, TokCP), toks))
    ncplen = sum(t.cnt for t in filter(lambda t: isinstance(t, TokCP), toks))
    ntot = nlit + ncplen
    assert ntot == len(syms1)
    lz4_sz = estimate_lz4_size(toks)
    ratio = ntot / max(lz4_sz, 1)

    logging.info(f"TOTAL SYMBOLS:          {nlit + ncplen}")
    logging.info(f"NUM OF LITERALS:        {nlit}")
    logging.info(f"NUM OF CP:              {ncp}")
    logging.info(f"CP-ENCODED SYMBOLS:     {ncplen}")
    logging.info(f"LZ4 ESTIM SIZE:         {lz4_sz}")
    logging.info(f"RATIO (overestim.):     {ratio:.2f}")
    logging.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Lz4 test")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-m", "--maxlen", type=int, default=0)
    args = parser.parse_args()
    test(args.maxlen, args.verbose)


if __name__ == "__main__":
    main()
