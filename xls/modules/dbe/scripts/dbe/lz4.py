#!/usr/bin/env python3

import logging
import enum
from typing import Sequence, List
from copy import deepcopy
import itertools

from .common import Params, Token, TokCP, TokLT, TokMK, Mark, PlainData


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
    RESTART = 2
    FIFO_PREFILL = 3
    FIFO_POSTFILL = 4
    START_MATCH_0 = 5
    START_MATCH_1 = 6
    CONTINUE_MATCH_0 = 7
    CONTINUE_MATCH_1 = 8
    EMIT_SHORT_MATCH = 9
    EMIT_FINAL_LITERALS = 10
    EMIT_END = 11
    ERROR = 12


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
            self.fifo_in_count = 0
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
            # -- Match continuations counter --
            # This stores number of iterations spent in CONTINUE_MATCH state.
            # It is used to compute match length and for other needs when 
            # terminating a match that has been started but needs to be
            # finished for any reason.
            self.match_nconts = 0
            # -- Hash table pointer --
            # User to address hash table during clear
            self.ht_ptr = 0
            # -- Recycle flag --
            # When we want to read a new input symbol, we do so if recycle
            # is False. If it is True, we instead recycle a symbol read in
            # the previous step.
            self.recycle = False
            # -- End sequence flag --
            # This is set to True once we've received the END marker.
            # We drain all internal buffers, emit END marker token and
            # prepare the encoder to accept a new data block.
            self.finalize = False
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

    def _recv_input(self) -> PlainData:
        rx = next(self.input_iter)
        self.log.debug(f" *** RECV {type(rx)}:{rx!r}")
        return rx

    def _send_output(self, tok: Token):
        self.log.debug(f" ***** SEND {tok}")
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
        do_recv = not cur.recycle and cur.fsm in (
            _FsmSt.FIFO_PREFILL,
            _FsmSt.START_MATCH_0,
            _FsmSt.CONTINUE_MATCH_0,
            _FsmSt.ERROR,
        )
        rx = self._recv_input() if do_recv else 0

        # Classify input markers
        rx_symbol = do_recv and not isinstance(rx, Mark)
        rx_mark = do_recv and isinstance(rx, Mark)
        rx_end = rx_mark and rx is Mark.END
        rx_error = rx_mark and Mark.is_error(rx)
        rx_reset = rx_mark and rx is Mark.RESET
        rx_unexpected_mark = rx_mark and not (rx_end or rx_reset or rx_error)

        if cur.fsm == _FsmSt.ERROR or cur.recycle:
            # Do not shift FIFOs / write HB in these states
            pass
        elif rx_symbol:
            # Push symbol to the HB and shift input FIFOs
            assert not upd.finalize
            # Update WP and HB-full flag
            upd.wp = (upd.wp + 1) & c.hb_mask
            if upd.wp == 0:
                upd.hb_all_valid = True
            # Write to FIFO
            upd.fifo_cache = [upd.fifo_in[0]] + upd.fifo_cache[:-1]
            upd.fifo_in = upd.fifo_in[1:] + [rx]
            upd.fifo_in_count = min(upd.fifo_in_count + 1, c.fifo_in_sz)
        elif rx_end:
            # When receiving END marker, shift input FIFO anyhow as we're
            # expected to drop the symbol at OP
            assert not upd.finalize
            upd.finalize = True
            upd.fifo_cache = [upd.fifo_in[0]] + upd.fifo_cache[:-1]
            upd.fifo_in = upd.fifo_in[1:] + [0]
            upd.fifo_in_count = min(upd.fifo_in_count + 1, c.fifo_in_sz)
            upd.fifo_in_nvalid = upd.fifo_in_count - 1
        elif cur.fsm in (_FsmSt.FIFO_POSTFILL, _FsmSt.EMIT_FINAL_LITERALS):
            # Feed input FIFO with 0s and shift
            upd.fifo_in = upd.fifo_in[1:] + [0]
            if cur.fsm == _FsmSt.FIFO_POSTFILL:
                upd.fifo_in_count = min(upd.fifo_in_count + 1, c.fifo_in_sz)
            else:
                upd.fifo_in_nvalid -= 1

        # Calc origin pointer OP from WP
        # WP points to the symbol in HB which corresponds to fifo_in[-1]
        # (the last received symbol), while OP points to the symbol in HB
        # which corresponds to fifo_in[0] ("the origin", the first
        # symbol of matched sequence)
        op = (upd.wp - c.fifo_in_sz + 1) & c.hb_mask

        # Hash table lookup is synchronized with hash table update and
        # is performed on each symbol till a match is found. Hash table is
        # not updated when growing an existing match.

        # Calculate u32 Fibonacci hash function
        _hsh = self._hash(c, upd)

        # Update match pointers & access HT RAM
        if cur.fsm == _FsmSt.START_MATCH_0:
            upd.pold = self._ht_read(_hsh)
            upd.pnew = op
        elif cur.fsm == _FsmSt.START_MATCH_1:
            self._ht_write(_hsh, op)
        elif cur.fsm == _FsmSt.HASH_TABLE_CLEAR:
            self._ht_write(upd.ht_ptr, 0)
            upd.ht_ptr = (upd.ht_ptr + 1) & c.ht_mask

        # Prepare to check for a match
        if cur.fsm == _FsmSt.START_MATCH_1:
            mchk_do = True
            mchk_pos = upd.pold
            mchk_canextend = True
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_1:
            mchk_do = True
            mchk_pos = (upd.pold + upd.match_nconts + 1) & c.hb_mask
            mchk_canextend = upd.match_nconts < (c.cnt_max - 1)
        else:
            mchk_do = False
        
        # Access HB RAM
        if rx_symbol and cur.fsm != _FsmSt.ERROR:
            self._hb_write(upd.wp, rx)
        elif mchk_do:
            mchk_sym = self._hb_read(mchk_pos)

        # Actually check for a match
        if mchk_do:
            # For match to happen, following criteria have to be met:
            # 1. _pos should not point to an unwritten HB entry
            _iswritten = upd.hb_all_valid or mchk_pos < upd.wp
            # 2. _pos should not point between OP and WP inclusive
            _isold = ((mchk_pos - op) & c.hb_mask) > ((upd.wp - op) & c.hb_mask)
            # 3. _sym should match current "origin" symbol
            _matches = mchk_sym == upd.fifo_in[0]
            # 4. our existing matching string should not be too long
            is_match = _iswritten and _isold and _matches and mchk_canextend
            self.log.debug(
                f"OP {upd.fifo_in[0]}: " + ("MATCH" if is_match else "MISS")
            )
        else:
            is_match = False

        # Update match_nconts
        if cur.fsm == _FsmSt.START_MATCH_1:
            upd.match_nconts = 0
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_1:
            upd.match_nconts += 1
        elif cur.fsm == _FsmSt.EMIT_SHORT_MATCH:
            assert upd.match_nconts >= 0
            upd.match_nconts -= 1

        # NOTE: match_len and other variables set below are available only
        # when terminating match in CONTINUE_MATCH_1 state.
        if cur.fsm == _FsmSt.CONTINUE_MATCH_1:
            # Do we terminate the current match?
            is_match_terminated = not is_match or upd.finalize
            # - If termination is due to a failed match, OP is excluded from
            #   match by definition.
            # - If termination is due to an EOF (upd.finalized=1), we will
            #   exclude OP from the match.
            # - That means, when match is terminated, OP symbol is always
            #   excluded from the match.
            # - If match was long enough, CP is emitted and we jump
            #   to EMIT_FINAL_LITERALS.
            # - Otherwise, we emit a single literal from fifo_cache
            #   and let EMIT_SHORT_MATCH handle the rest.
            # - EMIT_FINAL_LITERALS or next START_MATCH/CONTINUE_MATCH takes
            #   care of emitting OP as a literal.
            if is_match_terminated:
                match_len = upd.match_nconts
                match_is_long = match_len >= c.minmatch

        # Emit compressed data token
        if cur.fsm == _FsmSt.ERROR:
            # Do not emit anything in error state
            pass
        elif rx_error:
            # Propagate error
            self._send_output(TokMK(rx))
        elif rx_unexpected_mark:
            # Generate error
            self._send_output(TokMK(Mark.ERROR_BAD_MARK))
        elif (cur.fsm == _FsmSt.START_MATCH_1
              and (not is_match or upd.finalize)):
                # Emit symbol at OP as literal
                self._send_output(TokLT(upd.fifo_in[0]))
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_1 and is_match_terminated:
            # If match is long enough, emit a single CP
            # If not, emit symbols from Cache FIFO as literals
            assert upd.match_nconts > 0
            if match_is_long:
                _off = (upd.pnew - upd.pold) & c.hb_mask
                assert _off >= 1
                self._send_output(TokCP(_off - 1, match_len))
            else:
                self._send_output(TokLT(upd.fifo_cache[upd.match_nconts - 1]))
        elif cur.fsm == _FsmSt.EMIT_SHORT_MATCH:
            assert upd.match_nconts > 0
            self._send_output(TokLT(upd.fifo_cache[upd.match_nconts - 1]))
        elif cur.fsm == _FsmSt.EMIT_FINAL_LITERALS:
            # We dump literals from Input FIFO till nvalid becomes 0
            assert 1 <= upd.fifo_in_nvalid <= c.fifo_in_sz
            self._send_output(TokLT(upd.fifo_in[0]))
        elif cur.fsm == _FsmSt.EMIT_END:
            self._send_output(TokMK(Mark.END))

        # Handle state re-initialization
        if cur.fsm == _FsmSt.RESET:
            # Full reset
            upd = self.State(c)
        elif cur.fsm == _FsmSt.RESTART:
            # Intra-block partial reset, keeping HB and HT intact
            upd.fifo_in_count = 0
            upd.fifo_in_nvalid = 0
            upd.match_nconts = 0
            upd.finalize = False
            upd.recycle = False

        # State change logic
        if rx_error or rx_unexpected_mark:
            upd.fsm = _FsmSt.ERROR
        elif rx_reset:
            upd.fsm = _FsmSt.RESET
        elif cur.fsm == _FsmSt.RESET:
            upd.fsm = _FsmSt.HASH_TABLE_CLEAR
        elif cur.fsm == _FsmSt.HASH_TABLE_CLEAR:
            if upd.ht_ptr == 0:
                upd.fsm = _FsmSt.RESTART
        elif cur.fsm == _FsmSt.RESTART:
            upd.fsm = _FsmSt.FIFO_PREFILL
        elif cur.fsm == _FsmSt.FIFO_PREFILL:
            if rx_end:
                if upd.fifo_in_nvalid:
                    if upd.fifo_in_count < c.fifo_in_sz:
                        upd.fsm = _FsmSt.FIFO_POSTFILL
                    else:
                        upd.fsm = _FsmSt.EMIT_FINAL_LITERALS
                else:
                    # Empty input block
                    upd.fsm = _FsmSt.EMIT_END
            elif upd.fifo_in_count >= c.fifo_in_sz:
                upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.START_MATCH_0:
            upd.fsm = _FsmSt.START_MATCH_1
        elif cur.fsm == _FsmSt.START_MATCH_1:
            if upd.finalize:
                upd.fsm = _FsmSt.EMIT_FINAL_LITERALS
            elif is_match:
                upd.fsm = _FsmSt.CONTINUE_MATCH_0
            else:
                upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_0:
            upd.fsm = _FsmSt.CONTINUE_MATCH_1
        elif cur.fsm == _FsmSt.CONTINUE_MATCH_1:
            if is_match_terminated:
                if match_is_long or upd.match_nconts == 1:
                    if upd.finalize:
                        upd.fsm = _FsmSt.EMIT_FINAL_LITERALS
                    else:
                        upd.fsm = _FsmSt.START_MATCH_0
                else:
                    upd.fsm = _FsmSt.EMIT_SHORT_MATCH
            else:
                upd.fsm = _FsmSt.CONTINUE_MATCH_0
        elif cur.fsm == _FsmSt.EMIT_SHORT_MATCH:
            if upd.match_nconts == 1:
                if upd.finalize:
                    upd.fsm = _FsmSt.EMIT_FINAL_LITERALS
                else:
                    upd.fsm = _FsmSt.START_MATCH_0
        elif cur.fsm == _FsmSt.FIFO_POSTFILL:
            if upd.fifo_in_count == c.fifo_in_sz:
                upd.fsm = _FsmSt.EMIT_FINAL_LITERALS
        elif cur.fsm == _FsmSt.EMIT_FINAL_LITERALS:
            if upd.fifo_in_nvalid == 1:
                upd.fsm = _FsmSt.EMIT_END
        elif cur.fsm == _FsmSt.EMIT_END:
            upd.fsm = _FsmSt.RESTART

        # Set 'recycle' flag when we need to "stall" Input FIFO to not lose
        # the symbol at the OP, which is needed for certain state transitions
        if upd.fsm == _FsmSt.START_MATCH_0 and cur.fsm != _FsmSt.START_MATCH_1:
            upd.recycle = True
        elif (upd.fsm == _FsmSt.EMIT_FINAL_LITERALS and
            cur.fsm != _FsmSt.EMIT_FINAL_LITERALS and
            cur.fsm != _FsmSt.START_MATCH_1):
            upd.recycle = True
        else:
            upd.recycle = False

        self.state = upd

    def encode_block(self, data: Sequence[PlainData]) -> List[Token]:
        if not data or not isinstance(data[-1], Mark) or data[-1] != Mark.END:
            # If 'data' does not end with the END mark, append it.
            data = itertools.chain(data, [Mark.END])
        self.input_iter = iter(data)
        self.output_list = []

        # Run till we get END marker on the output
        while not (self.output_list
                   and isinstance(self.output_list[-1], TokMK)
                   and self.output_list[-1].mark == Mark.END):
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
        elif isinstance(t, TokMK) and t.mark == Mark.END:
            sz += count_block(litcount, 0)
            litcount = 0
        else:
            raise RuntimeError(f'Unexpected token: {t}')

    return sz
