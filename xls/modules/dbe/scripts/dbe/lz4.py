#!/usr/bin/env python3

import sys
import argparse
import os
import logging
import enum
from typing import Sequence, List, Tuple
from copy import deepcopy

from .common import Params, Token, TokCP, TokLT


class Lz4Params(Params):
    def __init__(
        self,
        symbits: int,
        ptrbits: int,
        cntbits: int,
        hash_bits: int,
        minmatch: int = 4,
        final_literals: int = 12,
    ):
        super().__init__(symbits, ptrbits, cntbits)
        self.hash_bits = hash_bits
        self.minmatch = minmatch
        self.final_literals = final_literals
        self.hb_mask = (1 << self.wptr) - 1
        self.ht_sz = 1 << self.hash_bits
        self.ht_mask = (1 << self.hash_bits) - 1
        self.fifo_cache_sz = self.minmatch - 1
        self.fifo_in_sz = max(self.minmatch, self.final_literals)


@enum.unique
class _FsmSt(enum.IntEnum):
    RESET = 0
    HASH_TABLE_CLEAR = 1
    INITIAL_FILL = 2
    START_MATCH_0 = 3
    START_MATCH_1 = 4
    CONTINUE_MATCH_0 = 5
    CONTINUE_MATCH_1 = 6
    TERMINATE_MATCH = 7
    DUMP_FINAL_LITERALS = 8


class Lz4Encoder:
    class State:
        def __init__(self, cfg: Lz4Params) -> None:
            # Two connected FIFOs:
            # 1. Cache FIFO, length: MINMATCH-1
            #       Remembers past symbols that may need to be emitted as
            #       literals in case our match is too short
            #       (shorter than MINMATCH)
            #
            # 2. Input FIFO, length: max(MINMATCH, FINAL_LITERALS)
            #       Stores incoming symbols used for hashing & delays incoming
            #       stream to form proper LZ4 block termination upon EOF
            #
            # Data first enters Input FIFO from the right, exits from the
            # left, then enters Cache FIFO from the left. Such arrangement
            # simplifies indexing.

            self.fifo_cache = [0] * cfg.fifo_cache_sz
            self.fifo_in = [0] * cfg.fifo_in_sz
            self.fifo_in_nvalid = 0

            # HB write pointer
            # Normally WP is the index of _last_ written HB element.
            # When a new element is written to HB, that's how it is done:
            #   WP = (WP + 1) & mask
            #   hb[WP] = new_element
            #
            # WP is also used to keep track of fill rate of HB (which elements
            # are valid and which are not).
            # Initially WP is set to 0. The first valid element will thus
            # be the one at idx=1. Element at idx=0 will continue to stay
            # invalid till the first time WP rolls over to 0.
            # Checking for WP==0 _after_ we've performed at least one write to
            # the buffer is an easy way to determine how many elements are
            # valid in the HB:
            #   Have we observed WP==0 after writing at least 1 element to HB?
            #       NO => HB has `WP` valid elements at idx in [1, WP]
            #       YES => all HB elements are valid
            self.wp = 0
            self.hb_all_valid = False

            # -- Match pointer registers --
            # Pointer to the beginning of the string we're trying to match
            # (not yet emitted as tokens)
            self.pnew = 0
            # Pointer to the beginning of historic string in HB, for which
            # tokens should've been already emitted, and that we're trying to
            # match to incoming symbols specified by match_pnew
            self.pold = 0
            # -- Hash table pointer --
            # User to address hash table during clear
            self.ht_ptr = 0
            # -- Recycle flag --
            # When we want to read a new input symbol, we do so if recycle
            # is False. If it is True, we instead recycle a symbol read in
            # the previous step.
            self.recycle = False
            # -- EOF flag --
            # This is set to True once we've read the symbol marked as
            # end-of-block. We drain all internal buffers, emit TokenEnd and
            # prepare the encoder to accept a new data block.
            self.eof = False
            # -- FSM state --
            self.fsm: _FsmSt = _FsmSt.RESET

    def __init__(self, cfg: Lz4Params, verbose: bool = False) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        self.log = logging.getLogger(__class__.__name__)
        self.log.setLevel(level)

        # -------------------------
        # Configuration (immutable)
        # -------------------------
        self.cfg = cfg

        # ---------------
        # State
        # ---------------
        self.state = self.State(self.cfg)

        # -------------
        # Memory blocks
        # -------------

        # History buffer
        self.hb = [0] * self.cfg.hb_maxsz

        # LZ4 hashtable
        # Given:
        #   hashtable[hash(syms[:MINMATCH])] == pos
        # We have:
        #   if is_valid(pos):
        #       hb[pos:pos+4] =? four_bytes
        # Here '=?' means "tentatively equal" because of possibility of
        # collisions.
        #
        #
        # LZ4 attempts to find a match by collecting a few unprocessed (in
        # the sense that no Tokens were emitted for them) input symbols;
        # computing hash for those symbols and making a bet that a
        # corresponding location in the hash table points to the matching
        # sequence of symbols in history buffer.
        # Compared to algorithms like ALDC that perform exhaustive matching,
        # this is not guaranteed to find the longest matching string existing
        # in the history buffer, or even to find a matching string at all
        # (even though it may exist). To maintain compression ratio, LZ4 uses
        # larger history buffer than what is usually used by ALDC.
        # Compression ratio usually increases in `1-1/exp(sz)`-like fashion
        # with increasing hash table size.
        #
        # Upon reset, hash table can be random-initialized. We anyway check
        # whether a position returned by the table points to a valid (written)
        # HB location, and then we read symbols from that location in HB to
        # detect potential collisions. However, it may be a good idea to
        # deterministically initialize hash table in the beginning of each LZ4
        # block, as otherwise compression result, although always being
        # correct, will not be deterministic and will depend on data
        # previously consumed by the encoder.
        self.ht = [0] * self.cfg.ht_sz

        # -----------
        # IO channels
        # -----------
        self.input_iter = iter(())
        self.output_list = []

    def _hb_read(self, addr: int) -> int:
        assert 0 <= addr <= self.cfg.hb_mask
        return self.hb[addr]

    def _hb_write(self, addr: int, v: int):
        assert 0 <= addr <= self.cfg.hb_mask
        assert 0 <= v <= self.cfg.sym_max
        self.hb[addr] = v

    def _ht_read(self, addr: int) -> int:
        assert 0 <= addr <= self.cfg.ht_mask
        return self.ht[addr]

    def _ht_write(self, addr: int, v: int):
        assert 0 <= addr <= self.cfg.ht_mask
        assert 0 <= v <= self.cfg.hb_mask
        self.ht[addr] = v

    def _recv_input(self) -> Tuple[int, bool]:
        sym, eof = next(self.input_iter)
        self.log.debug(f"RECV {sym}, eof={eof}")
        return sym, eof

    def _send_output(self, tok: Token):
        self.log.debug(f"SEND {tok}")
        self.output_list.append(tok)

    @staticmethod
    def _hash(cfg: Lz4Params, st: "Lz4Encoder.State") -> int:
        "Combinational hash function evaluated on Input FIFO"
        bs = bytes(st.fifo_in[: cfg.minmatch])
        x = int.from_bytes(bs, "little")
        h = (x * 2654435761) & 0xFFFFFFFF
        h >>= 32 - cfg.hash_bits
        assert 0 <= h <= cfg.ht_mask
        return h

    def _step(self):
        c = self.cfg
        cur = self.state
        upd = deepcopy(cur)

        self.log.debug(
            f" ---STEP--- fsm: {cur.fsm.name}, recycle: {cur.recycle}"
        )

        # Read new symbol from input
        if not cur.recycle and cur.fsm in (
            _FsmSt.INITIAL_FILL,
            _FsmSt.START_MATCH_0,
            _FsmSt.CONTINUE_MATCH_0,
        ):
            assert not upd.eof
            _sym, upd.eof = self._recv_input()
            # Push symbol to the HB and shift input FIFOs
            upd.wp = (upd.wp + 1) & c.hb_mask
            self._hb_write(upd.wp, _sym)
            upd.fifo_cache = [upd.fifo_in[0]] + upd.fifo_cache[:-1]
            upd.fifo_in = upd.fifo_in[1:] + [_sym]
            # Update HB-full flag
            if upd.wp == 0:
                upd.hb_all_valid = True
            # Update input FIFO fill counter
            upd.fifo_in_nvalid = min(upd.fifo_in_nvalid + 1, c.fifo_in_sz)
        upd.recycle = False

        # Calc origin pointer OP from WP
        # WP points to the symbol in HB which corresponds to fifo_in[-1]
        # (the last received symbol), while OP points to the symbol in HB
        # which corresponds to fifo_in[0] ("the origin", the first
        # symbol of matched sequence)
        op = (upd.wp - c.fifo_in_sz + 1) & c.hb_mask

        # Hash table lookup is synchronized with hash table update and
        # is performed on each symbol till a match is found. Hash table is
        # not updated when growing an existing matching.
        _hsh = self._hash(c, upd)
        if cur.fsm == _FsmSt.START_MATCH_0:
            upd.pold = self._ht_read(_hsh)
            upd.pnew = op
        elif cur.fsm == _FsmSt.START_MATCH_1:
            self._ht_write(_hsh, op)

        # Calculate length of current match
        match_len = (op - upd.pnew) & c.hb_mask
        match_is_long = match_len >= c.minmatch

        # Check for match
        if cur.fsm in (_FsmSt.START_MATCH_1, _FsmSt.CONTINUE_MATCH_1):
            # Fetch historical symbol from HB to check for match
            _pos = (upd.pold + match_len) & c.hb_mask
            _sym = self._hb_read(_pos)
            # For match to happen, following criteria have to be met:
            # 1. _pos should not point to an unwritten HB entry
            _iswritten = upd.hb_all_valid or _pos < upd.wp
            # 2. _pos should not point between OP and WP inclusive
            _isold = ((_pos - op) & c.hb_mask) > ((upd.wp - op) & c.hb_mask)
            # 3. _sym should match current "origin" symbol
            _matches = _sym == upd.fifo_in[0]
            # 4. our existing matching string should not be too long
            #    (match needs to be force-terminated if it is)
            _canextend = match_len < c.cnt_max
            is_match = _iswritten and _isold and _matches and _canextend
            self.log.debug(
                f"OP {upd.fifo_in[0]}: " + ("MATCH" if is_match else "MISS")
            )

        # Emit compressed data token
        if cur.fsm == _FsmSt.START_MATCH_1:
            if not upd.eof and not is_match:
                # Emit symbol at OP as literal
                self._send_output(TokLT(c, False, upd.fifo_in[0]))
        elif cur.fsm == _FsmSt.TERMINATE_MATCH:
            # If match is long enough, emit a single CP
            # If not, emit symbols from Cache FIFO as literals
            assert match_len >= 1
            if match_is_long:
                _off = (upd.pnew - upd.pold) & c.hb_mask
                assert _off >= 1
                _off -= 1
                self._send_output(TokCP(c, False, _off, match_len))
            else:
                # For literal emission we use match_len as index into
                # Cache FIFO. After each produced literal, we increment
                # pnew to decrease match_len by 1
                self._send_output(TokLT(c, False, upd.fifo_cache[match_len - 1]))
                upd.pnew = (upd.pnew + 1) & c.hb_mask
        elif cur.fsm == _FsmSt.DUMP_FINAL_LITERALS:
            # We dump literals from Input FIFO till nvalid becomes 0
            assert upd.fifo_in_nvalid <= c.fifo_in_sz
            if upd.fifo_in_nvalid > 0:
                _last = upd.fifo_in_nvalid == 1
                _idx = c.fifo_in_sz - upd.fifo_in_nvalid
                self._send_output(TokLT(c, _last, upd.fifo_in[_idx]))
                upd.fifo_in_nvalid -= 1

        # Clear hash table (one cell at a time)
        if cur.fsm == _FsmSt.HASH_TABLE_CLEAR:
            self._ht_write(upd.ht_ptr, 0)
            upd.ht_ptr = (upd.ht_ptr + 1) & c.ht_mask

        # Handle reset requests
        if cur.fsm == _FsmSt.RESET:
            upd = self.State(c)

        # State change logic
        if cur.fsm == _FsmSt.RESET:
            upd.fsm = _FsmSt.HASH_TABLE_CLEAR
        elif cur.fsm == _FsmSt.HASH_TABLE_CLEAR:
            if upd.ht_ptr == 0:
                upd.fsm = _FsmSt.INITIAL_FILL
        elif cur.fsm == _FsmSt.INITIAL_FILL:
            if upd.eof:
                upd.fsm = _FsmSt.DUMP_FINAL_LITERALS
            elif upd.fifo_in_nvalid >= c.fifo_in_sz:
                upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.START_MATCH_0:
            upd.fsm = _FsmSt.START_MATCH_1
        elif cur.fsm == _FsmSt.START_MATCH_1:
            if upd.eof:
                upd.fsm = _FsmSt.DUMP_FINAL_LITERALS
            elif is_match:
                upd.fsm = _FsmSt.CONTINUE_MATCH_0
            else:
                upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_0:
            upd.fsm = _FsmSt.CONTINUE_MATCH_1
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_1:
            if upd.eof or not is_match:
                upd.fsm = _FsmSt.TERMINATE_MATCH
            else:
                upd.fsm = _FsmSt.CONTINUE_MATCH_0
        elif cur.fsm == _FsmSt.TERMINATE_MATCH:
            if match_is_long or match_len == 1:
                if upd.eof:
                    upd.fsm = _FsmSt.DUMP_FINAL_LITERALS
                else:
                    upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.DUMP_FINAL_LITERALS:
            if upd.fifo_in_nvalid == 0:
                # Reset and prepare for compression of next block
                upd.fsm = _FsmSt.RESET
        else:
            raise NotImplementedError(f"Unexpected state {cur.fsm}")

        # Set 'recycle' flag when moving into START_MATCH_0 state from
        # any state other than START_MATCH_1.
        if upd.fsm == _FsmSt.START_MATCH_0 and cur.fsm != _FsmSt.START_MATCH_1:
            upd.recycle = True

        self.state = upd

    def encode_block(self, data: Sequence[int]) -> List[Token]:
        assert data
        igen = ((v, i == len(data) - 1) for i, v in enumerate(data))
        self.input_iter = iter(igen)
        self.output_list = []
        self.state.fsm = _FsmSt.RESET

        # Run till we get TokEnd on the output
        while not (self.output_list and self.output_list[-1].last):
            self._step()

        return self.output_list


def estimate_lz4_size(toks: Sequence[Token]) -> int:
    sz = 0

    def count_block(lit_count, cp_count) -> int:
        assert cp_count == 0 or cp_count >= 4
        cp_count_off = cp_count - 4
        # 4+4 bit token
        bs = 1
        # Extra bytes encoding literal count
        num_lc = (lit_count - 15 + 255) // 255
        bs += num_lc
        # Literals themselves
        bs += lit_count
        # MATCH part is generated onlt when cp_count >= 4
        if cp_count_off >= 0:
            # Match offset
            bs += 2
            # Extra bytes encoding match length
            num_cpc = (cp_count_off - 15 + 255) // 255
            bs += num_cpc
        return bs

    litcount = 0
    for t in toks:
        if isinstance(t, TokLT):
            litcount += 1
        elif isinstance(t, TokCP):
            sz += count_block(litcount, t.cnt)
            litcount = 0
        if t.last:
            assert not isinstance(t, TokCP), "last Token can't be a CP"
            sz += count_block(litcount, 0)
            litcount = 0

    return sz
