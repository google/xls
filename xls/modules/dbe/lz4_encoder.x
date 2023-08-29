// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import std
import xls.examples.ram as ram
import xls.modules.dbe.common as dbe

type Mark = dbe::Mark;
type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type AbstractRamWriteReq = ram::WriteReq;
type AbstractRamWriteResp = ram::WriteResp;
type AbstractRamReadReq = ram::ReadReq;
type AbstractRamReadResp = ram::ReadResp;
type SimultaneousReadWriteBehavior = ram::SimultaneousReadWriteBehavior;


fn fifo_shift_left<FIFO_SIZE: u32, DATA_WIDTH: u32>(
    fifo: uN[DATA_WIDTH][FIFO_SIZE],
    data: uN[DATA_WIDTH]
) -> uN[DATA_WIDTH][FIFO_SIZE] {
   let fifo = for (i, arr): (u32, uN[DATA_WIDTH][FIFO_SIZE])
               in u32:1..FIFO_SIZE {
       update(arr, i - u32:1, fifo[i])
   } (uN[DATA_WIDTH][FIFO_SIZE]: [0, ...]);
   let fifo = update(fifo, FIFO_SIZE - u32:1, data);
   fifo
}

fn fibbonacci_factor(width: u32) -> u64 {
    // This is equal to:
    // round((1 << width) / golden), where golden=1.618033989...
    match width {
        u32:16 => u64:40503,
        u32:24 => u64:10368890,
        u32:32 => u64:2654435769,
        u32:48 => u64:173961102589771,
        u32:64 => u64:11400714819323198485,
        _ => fail!("unsupported_fibbonacci_factor", u64:0),
    }
}

fn hash_fibonacci<
    SYMBOL_WIDTH: u32, HASH_SYMBOLS: u32, HASH_WIDTH: u32,
    TOTAL_WIDTH: u32 = {SYMBOL_WIDTH * HASH_SYMBOLS},
    FACTOR: u64 = {fibbonacci_factor(TOTAL_WIDTH)}
> (
    symbols: uN[SYMBOL_WIDTH][HASH_SYMBOLS]
) -> uN[HASH_WIDTH] {
    // Flatten the input, placing symbol 0 into LSBs
    let input_word = array_rev(symbols) as uN[TOTAL_WIDTH];
    // Calculate the full hash
    let hash = input_word * (FACTOR as uN[TOTAL_WIDTH]);
    // Truncate, leaving only MSBs
    (hash >> (TOTAL_WIDTH - HASH_WIDTH)) as uN[HASH_WIDTH]
}

enum FsmSt: u4 {
    RESET = 0,
    HASH_TABLE_CLEAR = 1,
    RESTART = 2,
    FIFO_PREFILL = 3,
    START_MATCH_0 = 4,
    START_MATCH_1 = 5,
    CONTINUE_MATCH_0 = 6,
    CONTINUE_MATCH_1 = 7,
    FIFO_DRAIN = 8,
    EMIT_END = 9,
    ERROR = 10,
}

struct State<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32,
    HASH_SYMBOLS: u32, HASH_WIDTH: u32
> {
    fifo_in: uN[SYMBOL_WIDTH][HASH_SYMBOLS],
    fifo_in_count: u32,
    hb_wr_ptr: uN[MATCH_OFFSET_WIDTH],
    hb_all_valid: bool,
    hb_match_ptr: uN[MATCH_OFFSET_WIDTH],
    match_offset: uN[MATCH_OFFSET_WIDTH],
    match_length: uN[MATCH_LENGTH_WIDTH],
    ht_ptr: uN[HASH_WIDTH],
    recycle: bool,
    finalize: bool,
    is_match_dly: bool,
    fsm: FsmSt,
}

fn init_state<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32,
    HASH_SYMBOLS: u32, HASH_WIDTH: u32
> () -> State<
    SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH,
    HASH_SYMBOLS, HASH_WIDTH
> {
    State{
        fifo_in: uN[SYMBOL_WIDTH][HASH_SYMBOLS]: [uN[SYMBOL_WIDTH]:0, ...],
        fifo_in_count: u32:0,
        hb_wr_ptr: uN[MATCH_OFFSET_WIDTH]:0,
        hb_all_valid: false,
        hb_match_ptr: uN[MATCH_OFFSET_WIDTH]:0,
        match_offset: uN[MATCH_OFFSET_WIDTH]:0,
        match_length: uN[MATCH_LENGTH_WIDTH]:0,
        ht_ptr: uN[HASH_WIDTH]:0,
        recycle: false,
        finalize: false,
        is_match_dly: false,
        fsm: FsmSt::RESET
    }
}

pub proc encoder_base<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32,
    HASH_SYMBOLS: u32, HASH_WIDTH: u32, DO_CLEAR_HASH_TABLE: bool
> {
    plain_data: chan<PlainData<SYMBOL_WIDTH>> in;
    encoded_data:
        chan<Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>> out;

    /// History buffer RAM
    /// Size: (1<<MATCH_OFFSET_WIDTH) x SYMBOL_WIDTH
    ram_hb_rd_req: chan<AbstractRamReadReq<MATCH_OFFSET_WIDTH, 1>> out;
    ram_hb_rd_resp: chan<AbstractRamReadResp<SYMBOL_WIDTH>> in;
    ram_hb_wr_req:
        chan<AbstractRamWriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>> out;
    ram_hb_wr_comp: chan<AbstractRamWriteResp> in;

    /// Hash table RAM
    /// Size: (1<<HASH_WIDTH) x MATCH_OFFSET_WIDTH
    ram_ht_rd_req: chan<AbstractRamReadReq<HASH_WIDTH, 1>> out;
    ram_ht_rd_resp: chan<AbstractRamReadResp<MATCH_OFFSET_WIDTH>> in;
    ram_ht_wr_req:
        chan<AbstractRamWriteReq<HASH_WIDTH, MATCH_OFFSET_WIDTH, 1>> out;
    ram_ht_wr_comp: chan<AbstractRamWriteResp> in;

    init {
        init_state<
            SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH,
            HASH_SYMBOLS, HASH_WIDTH
        >()
    }

    config (
        // Data in, tokens out
        plain_data: chan<PlainData<SYMBOL_WIDTH>> in,
        encoded_data:
            chan<
                Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>
            > out,
        // RAM
        ram_hb_rd_req: chan<AbstractRamReadReq<MATCH_OFFSET_WIDTH, 1>> out,
        ram_hb_rd_resp: chan<AbstractRamReadResp<SYMBOL_WIDTH>> in,
        ram_hb_wr_req:
            chan<
                AbstractRamWriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>
            > out,
        ram_hb_wr_comp: chan<AbstractRamWriteResp> in,
        ram_ht_rd_req: chan<AbstractRamReadReq<HASH_WIDTH, 1>> out,
        ram_ht_rd_resp: chan<AbstractRamReadResp<MATCH_OFFSET_WIDTH>> in,
        ram_ht_wr_req:
            chan<
                AbstractRamWriteReq<HASH_WIDTH, MATCH_OFFSET_WIDTH, 1>
            > out,
        ram_ht_wr_comp: chan<AbstractRamWriteResp> in,
    ) {
        (
            plain_data, encoded_data,
            ram_hb_rd_req, ram_hb_rd_resp, ram_hb_wr_req, ram_hb_wr_comp,
            ram_ht_rd_req, ram_ht_rd_resp, ram_ht_wr_req, ram_ht_wr_comp,
        )
    }

    next (tok: token, cur: State) {
        type EncToken =
            Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>;
        type EncData = PlainData<SYMBOL_WIDTH>;
        type HbRamReadReq = AbstractRamReadReq<MATCH_OFFSET_WIDTH, 1>;
        type HbRamReadResp = AbstractRamReadResp<SYMBOL_WIDTH>;
        type HbRamWriteReq =
            AbstractRamWriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>;
        type HtRamReadReq = AbstractRamReadReq<HASH_WIDTH, 1>;
        type HtRamReadResp = AbstractRamReadResp<MATCH_OFFSET_WIDTH>;
        type HtRamWriteReq =
            AbstractRamWriteReq<HASH_WIDTH, MATCH_OFFSET_WIDTH, 1>;

        let upd = cur;

        // Read new symbol from input
        let do_recv = !cur.recycle && (
            cur.fsm == FsmSt::FIFO_PREFILL
            || cur.fsm == FsmSt::START_MATCH_0
            || cur.fsm == FsmSt::CONTINUE_MATCH_0
            || cur.fsm == FsmSt::ERROR
        );

        // I_DATA RECV
        let (tok, rx) = recv_if(tok, plain_data, do_recv,
            zero!<EncData>());

        // Classify input markers
        let rx_symbol = do_recv && !rx.is_marker;
        let rx_mark = do_recv && rx.is_marker;
        let rx_end = rx_mark && rx.mark == Mark::END;
        let rx_error = rx_mark && dbe::is_error(rx.mark);
        let rx_reset = rx_mark && rx.mark == Mark::RESET;
        let rx_unexpected_mark = rx_mark && !(rx_end || rx_reset || rx_error);

        let upd = if cur.fsm == FsmSt::ERROR || cur.recycle {
            // Do not shift FIFO in these states
            upd
        } else if rx_symbol {
            let upd = State{
                fifo_in:
                    fifo_shift_left(upd.fifo_in, rx.data),
                fifo_in_count:
                    std::umin(upd.fifo_in_count + u32:1, HASH_SYMBOLS),
                ..upd
            };
            upd
        } else if rx_end {
            // When receiving END marker, shift input FIFO anyhow as we're
            // expected to drop the current symbol
            let upd = State{
                finalize: true,
                fifo_in:
                    fifo_shift_left(upd.fifo_in, uN[SYMBOL_WIDTH]:0),
                fifo_in_count:
                    std::umax(upd.fifo_in_count, u32:1) - u32:1,
                ..upd
            };
            upd
        } else if cur.fsm == FsmSt::FIFO_DRAIN {
            // Feed input FIFO with 0s and shift
            let upd = State{
                fifo_in:
                    fifo_shift_left(upd.fifo_in, uN[SYMBOL_WIDTH]:0),
                fifo_in_count:
                    std::umax(upd.fifo_in_count, u32:1) - u32:1,
                ..upd
            };
            upd
        } else {
            upd
        };
    
        // Current symbol and its location in HB
        let current_symbol = upd.fifo_in[u32:0];
        // Address of current symbol in HB
        let current_ptr = if !cur.recycle {
            // When 'current' symbol is not recycled, it is not yet in HB, so
            // its address is the address of the next unwritten location,
            // 'hb_wr_ptr', but when it has been recycled, it's already in
            // the HB, we need the address of the last written location.
            upd.hb_wr_ptr
        } else {
            upd.hb_wr_ptr - uN[MATCH_OFFSET_WIDTH]:1
        };

        // Calculate hash function
        let hash = hash_fibonacci<
            SYMBOL_WIDTH, HASH_SYMBOLS, HASH_WIDTH
        >(upd.fifo_in);

        // NOTE: HT RAM, HB RAM accesses and o_token emission all happen
        // in parallel.

        // Read HT RAM (stage 0)
        let (ht_rd_vld, ht_rd_req) = if cur.fsm == FsmSt::START_MATCH_0 {
            (true, ram::ReadWordReq<u32:1>(hash))
        } else {
            (false, zero!<HtRamReadReq>())
        };
        let tok_ht = send_if(tok, ram_ht_rd_req, ht_rd_vld, ht_rd_req);
        let (tok_ht, ht_rd_resp) = recv_if(tok_ht, ram_ht_rd_resp, ht_rd_vld,
            zero!<HtRamReadResp>());
        let possible_match_ptr = ht_rd_resp.data;

        // Update matching string pointer
        let upd = if cur.fsm == FsmSt::START_MATCH_0 {
            State {
                hb_match_ptr: possible_match_ptr,
                match_offset: current_ptr - possible_match_ptr,
                match_length: uN[MATCH_LENGTH_WIDTH]:0,
                ..upd
            }
        } else if cur.fsm == FsmSt::CONTINUE_MATCH_0 {
            State {
                hb_match_ptr: upd.hb_match_ptr + uN[MATCH_OFFSET_WIDTH]:1,
                ..upd
            }
        } else {
            upd
        };

        // Prepare for matching
        let do_match =
            cur.fsm == FsmSt::START_MATCH_0
            || cur.fsm == FsmSt::CONTINUE_MATCH_0;
        
        // Read HB RAM (if match_ptr points to an already-written location)
        let hb_match_ptr_valid =
            upd.hb_all_valid || (upd.hb_match_ptr < current_ptr);
        let (hb_rd_vld, hb_rd_req) = if do_match && hb_match_ptr_valid {
            (true, ram::ReadWordReq<u32:1>(upd.hb_match_ptr))
        } else {
            (false, zero!<HbRamReadReq>())
        };
        let tok_hb = send_if(tok, ram_hb_rd_req, hb_rd_vld, hb_rd_req);
        let (tok_hb, hb_rd_resp) = recv_if(tok_hb, ram_hb_rd_resp, hb_rd_vld,
            zero!<HbRamReadResp>());
        let candidate_symbol = hb_rd_resp.data;

        // Check for match
        // - disallow matches with offset equal to 0 (or HB_SIZE)
        // - terminate the match if the match string is at max len
        // - terminate the match if END token has been received
        let is_match = (
            do_match && hb_match_ptr_valid
            && candidate_symbol == current_symbol
            && upd.match_offset != uN[MATCH_OFFSET_WIDTH]:0
            && upd.match_length < std::unsigned_max_value<MATCH_LENGTH_WIDTH>()
            && !upd.finalize
        ); 
        
        // Update match length
        let upd = if is_match && cur.fsm == FsmSt::CONTINUE_MATCH_0 {
            State {
                match_length: upd.match_length + uN[MATCH_LENGTH_WIDTH]:1,
                ..upd
            }
        } else {
            upd
        };

        // Update ht_ptr used when clearing HT RAM
        let upd = if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            State {
                ht_ptr: upd.ht_ptr + uN[HASH_WIDTH]:1,
                ..upd
            }
        } else {
            upd
        };

        // Write HT RAM
        let (ht_wr_vld, ht_wr_req) = if cur.fsm == FsmSt::START_MATCH_1 {
            (true, ram::WriteWordReq<u32:1>(hash, current_ptr))
        } else if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            (
                true,
                ram::WriteWordReq<u32:1>(upd.ht_ptr, uN[MATCH_OFFSET_WIDTH]:0)
            )
        } else {
            (false, zero!<HtRamWriteReq>())
        };
        let tok_ht = send_if(tok, ram_ht_wr_req, ht_wr_vld, ht_wr_req);
        let (tok_ht, _) = recv_if(tok_ht, ram_ht_wr_comp, ht_wr_vld,
            zero!<AbstractRamWriteResp>());

        // Write HB RAM
        let do_write_hb =
            !cur.recycle && (
                cur.fsm == FsmSt::START_MATCH_1
                || cur.fsm == FsmSt::CONTINUE_MATCH_1
                || cur.fsm == FsmSt::FIFO_DRAIN
            );
        let (hb_wr_vld, hb_wr_req) = if do_write_hb {
            (true, ram::WriteWordReq<u32:1>(upd.hb_wr_ptr, current_symbol))
        } else {
            (false, zero!<HbRamWriteReq>())
        };
        let tok_hb = send_if(tok, ram_hb_wr_req, hb_wr_vld, hb_wr_req);
        let (tok_hb, _) = recv_if(tok_hb, ram_hb_wr_comp, hb_wr_vld,
            zero!<AbstractRamWriteResp>());
        
        // Update hb_wr_ptr
        let upd = if do_write_hb {
            let hb_wr_ptr = upd.hb_wr_ptr + uN[MATCH_OFFSET_WIDTH]:1;
            State {
                hb_wr_ptr: hb_wr_ptr,
                hb_all_valid:
                    if hb_wr_ptr == uN[MATCH_OFFSET_WIDTH]:0 {
                        true
                    } else {
                        upd.hb_all_valid
                    },
                ..upd
            }
        } else {
            upd
        };

        // Emit compressed data token
        let zero_tok = zero!<EncToken>();
        let (do_emit, encoded_tok) = if rx_reset {
            // Propagate RESET
            (true, Token{
                kind: TokenKind::MARKER,
                mark: Mark::RESET,
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::ERROR {
            // Do not emit anything in error state
            (false, zero_tok)
        } else if rx_error {
            // Propagate error
            (true, Token{
                kind: TokenKind::MARKER,
                mark: rx.mark,
                ..zero_tok
            })
        } else if rx_unexpected_mark {
            // Generate error
            (true, Token{
                kind: TokenKind::MARKER,
                mark: Mark::ERROR_BAD_MARK,
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::START_MATCH_0 {
            if is_match  {
                // First symbol in a match - emit as matched
                (true, Token{
                    kind: TokenKind::MATCHED_SYMBOL,
                    symbol: current_symbol,
                    ..zero_tok
                })
            } else {
                // Emit current symbol as unmatched
                (true, Token{
                    kind: TokenKind::UNMATCHED_SYMBOL,
                    symbol: current_symbol,
                    ..zero_tok
                })
            }
        } else if cur.fsm == FsmSt::CONTINUE_MATCH_0 {
            if !is_match {
                // Emit a single MATCH token
                (true, Token {
                    kind: TokenKind::MATCH,
                    match_offset: upd.match_offset - uN[MATCH_OFFSET_WIDTH]:1,
                    match_length: upd.match_length,
                    ..zero_tok
                })
            } else {
                // Match continues, we emit all symbols as MATCHED_SYMBOLs
                (true, Token{
                    kind: TokenKind::MATCHED_SYMBOL,
                    symbol: current_symbol,
                    ..zero_tok
                })
            }
        } else if cur.fsm == FsmSt::FIFO_PREFILL {
            if upd.finalize && upd.fifo_in_count >= u32:1 {
                (true, Token {
                    kind: TokenKind::UNMATCHED_SYMBOL,
                    symbol: current_symbol,
                    ..zero_tok
                })
            } else {
                (false, zero_tok)
            }            
        } else if cur.fsm == FsmSt::FIFO_DRAIN {
            (true, Token {
                kind: TokenKind::UNMATCHED_SYMBOL,
                symbol: current_symbol,
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::EMIT_END {
            (true, Token {
                kind: TokenKind::MARKER,
                mark: Mark::END,
                ..zero_tok
            })
        } else {
            (false, zero_tok)
        };
        let tok_out = send_if(tok, encoded_data, do_emit, encoded_tok);

        // Handle state re-initialization
        let upd = if cur.fsm == FsmSt::RESET {
            // Full reset
            // NOTE: .fsm value will be overridden by the state change logic
            init_state<
                SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH,
                HASH_SYMBOLS, HASH_WIDTH
            >()
        } else if cur.fsm == FsmSt::RESTART {
            // Intra-block partial reset, keeping HB and HT intact
            State {
                fifo_in_count: u32:0,
                finalize: false,
                recycle: false,
                ..upd
            }
        } else {
            upd
        };

        // State change logic
        let fsm = if rx_error || rx_unexpected_mark {
            FsmSt::ERROR
        } else if rx_reset {
            FsmSt::RESET
        } else {
            match cur.fsm {
                FsmSt::RESET => {
                    if DO_CLEAR_HASH_TABLE {
                        FsmSt::HASH_TABLE_CLEAR
                    } else {
                        FsmSt::RESTART
                    }
                },
                FsmSt::HASH_TABLE_CLEAR => {
                    if upd.ht_ptr == uN[HASH_WIDTH]:0 {
                        FsmSt::RESTART
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::RESTART => {
                    if HASH_SYMBOLS > u32:1 {
                        FsmSt::FIFO_PREFILL
                    } else {
                        // When entering START_MATCH_0, there should be
                        // exactly 1 unfilled spot in the FIFO since it will
                        // be filled by START_MATCH_0 itself
                        FsmSt::START_MATCH_0
                    }
                },
                FsmSt::FIFO_PREFILL => {
                    if rx_end {
                        if upd.fifo_in_count >= u32:2 {
                            FsmSt::FIFO_DRAIN
                        } else {
                            // If there was 1 symbol in the block it has been
                            // already handled by FIFO_PREFILL state
                            FsmSt::EMIT_END
                        }
                    } else if upd.fifo_in_count == (HASH_SYMBOLS - u32:1) {
                        // One empty spot will be filled in by START_MATCH
                        // itself, we can start matching
                        FsmSt::START_MATCH_0
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::START_MATCH_0 => FsmSt::START_MATCH_1,
                FsmSt::CONTINUE_MATCH_0 => FsmSt::CONTINUE_MATCH_1,
                FsmSt::START_MATCH_1 | FsmSt::CONTINUE_MATCH_1 => {
                    if upd.finalize {
                        if HASH_SYMBOLS > u32:1 {
                            FsmSt::FIFO_DRAIN
                        } else {
                            FsmSt::EMIT_END
                        }
                    } else if upd.is_match_dly {
                        // Continue growing the matching string
                        FsmSt::CONTINUE_MATCH_0
                    } else {
                        FsmSt::START_MATCH_0
                    }
                },
                FsmSt::FIFO_DRAIN => {
                    // We quit FIFO_DRAIN stage when there's 1 symbol left
                    // because it has been already processed (written to HB
                    // and emitted as a token) by the FIFO_DRAIN stage itself
                    if upd.fifo_in_count == u32:1 {
                        FsmSt::EMIT_END
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::EMIT_END => FsmSt::RESTART,
                _ => fail!("unhandled_fsm_state", cur.fsm)
            }
        };

        let upd = State {
            fsm: fsm,
            ..upd
        };

        // Set 'recycle' flag when we want next tick to reuse the current
        // oldest symbol in the input FIFO instead of discarding it
        let recycle = if (
            upd.fsm == FsmSt::START_MATCH_1
            || upd.fsm == FsmSt::CONTINUE_MATCH_1
        ) {
            // Keep value from MATCH_0 state so that MATCH_1 can use it.
            upd.recycle
        } else if (
            cur.fsm == FsmSt::CONTINUE_MATCH_1
            && upd.fsm != FsmSt::CONTINUE_MATCH_0
        ) {
            true
        } else {
            false
        };
        
        // Set 'recycle' and delayed 'is_match'
        let upd = State {
            recycle: recycle,
            is_match_dly: is_match,
            ..upd
        };

        upd
    }
}

/// Version of `encoder_base` that uses RamModel
/// Intended to be used only for tests
pub proc encoder_base_modelram<
    SYMBOL_WIDTH: u32, MATCH_OFFSET_WIDTH: u32, MATCH_LENGTH_WIDTH: u32,
    HASH_SYMBOLS: u32, HASH_WIDTH: u32, DO_CLEAR_HASH_TABLE: bool
> {
    init{()}

    config (
        plain_data: chan<PlainData<SYMBOL_WIDTH>> in,
        encoded_data:
            chan<
                Token<SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH>
            > out,
    ) {
        let (hb_rd_req_s, hb_rd_req_r) =
            chan<AbstractRamReadReq<MATCH_OFFSET_WIDTH, 1>>;
        let (hb_rd_resp_s, hb_rd_resp_r) =
            chan<AbstractRamReadResp<SYMBOL_WIDTH>>;
        let (hb_wr_req_s, hb_wr_req_r) =
            chan<AbstractRamWriteReq<MATCH_OFFSET_WIDTH, SYMBOL_WIDTH, 1>>;
        let (hb_wr_comp_s, hb_wr_comp_r) =
            chan<AbstractRamWriteResp>;
        let (ht_rd_req_s, ht_rd_req_r) =
            chan<AbstractRamReadReq<HASH_WIDTH, 1>>;
        let (ht_rd_resp_s, ht_rd_resp_r) =
            chan<AbstractRamReadResp<MATCH_OFFSET_WIDTH>>;
        let (ht_wr_req_s, ht_wr_req_r) =
            chan<AbstractRamWriteReq<HASH_WIDTH, MATCH_OFFSET_WIDTH, 1>>;
        let (ht_wr_comp_s, ht_wr_comp_r) =
            chan<AbstractRamWriteResp>;

        spawn ram::RamModel<
            SYMBOL_WIDTH, {u32:1<<MATCH_OFFSET_WIDTH}, SYMBOL_WIDTH,
            SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
            true
        >(hb_rd_req_r, hb_rd_resp_s, hb_wr_req_r, hb_wr_comp_s);
        spawn ram::RamModel<
            MATCH_OFFSET_WIDTH, {u32:1<<HASH_WIDTH}, MATCH_OFFSET_WIDTH,
            SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
            true
        >(ht_rd_req_r, ht_rd_resp_s, ht_wr_req_r, ht_wr_comp_s);

        spawn encoder_base<
            SYMBOL_WIDTH, MATCH_OFFSET_WIDTH, MATCH_LENGTH_WIDTH,
            HASH_SYMBOLS, HASH_WIDTH, DO_CLEAR_HASH_TABLE
        > (
            plain_data, encoded_data,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_comp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_comp_r,
        );
    }

    next (tok: token, state: ()) {
    }
}

/// LZ4 encoder with 8K hash table

const LZ4_SYMBOL_WIDTH = u32:8;
const LZ4_OFFSET_WIDTH = u32:16;
const LZ4_COUNT_WIDTH = u32:16;
const LZ4_HASH_SYMBOLS = u32:4;
const LZ4_HASH_WIDTH_8K = u32:13;

pub proc encoder_8k {
    init{()}

    config (
        plain_data:
            chan<PlainData<LZ4_SYMBOL_WIDTH>> in,
        encoded_data:
            chan<
                Token<LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH>
            > out,
        ram_hb_rd_req: chan<AbstractRamReadReq<LZ4_OFFSET_WIDTH, 1>> out,
        ram_hb_rd_resp: chan<AbstractRamReadResp<LZ4_SYMBOL_WIDTH>> in,
        ram_hb_wr_req:
            chan<
                AbstractRamWriteReq<LZ4_OFFSET_WIDTH, LZ4_SYMBOL_WIDTH, 1>
            > out,
        ram_hb_wr_comp: chan<AbstractRamWriteResp> in,
        ram_ht_rd_req: chan<AbstractRamReadReq<LZ4_HASH_WIDTH_8K, 1>> out,
        ram_ht_rd_resp: chan<AbstractRamReadResp<LZ4_OFFSET_WIDTH>> in,
        ram_ht_wr_req:
            chan<
                AbstractRamWriteReq<LZ4_HASH_WIDTH_8K, LZ4_OFFSET_WIDTH, 1>
            > out,
        ram_ht_wr_comp: chan<AbstractRamWriteResp> in,
    ) {
        spawn encoder_base <
            LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH,
            LZ4_HASH_SYMBOLS, LZ4_HASH_WIDTH_8K, true
        > (
            plain_data, encoded_data,
            ram_hb_rd_req, ram_hb_rd_resp, ram_hb_wr_req, ram_hb_wr_comp,
            ram_ht_rd_req, ram_ht_rd_resp, ram_ht_wr_req, ram_ht_wr_comp,
        );
    }

    next(tok: token, st: ()) {
    }
}

/// Version of `encoder_8k` that uses RamModel
/// Intended to be used only for tests
pub proc encoder_8k_modelram {
    init{()}

    config (
        plain_data:
            chan<PlainData<LZ4_SYMBOL_WIDTH>> in,
        encoded_data:
            chan<Token<
                LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH
            >> out,
    ) {
        spawn encoder_base_modelram<
            LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH,
            LZ4_HASH_SYMBOLS, LZ4_HASH_WIDTH_8K, false
        > (plain_data, encoded_data);
    }

    next (tok: token, state: ()) {
    }
}

///
/// Tests
///

import xls.modules.dbe.common_test as test

const TEST_DATA_LEN = u32:10;
const TEST_DATA = PlainData<8>[TEST_DATA_LEN]: [
    // Check basic LT and MATCH generation
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:2, mark: Mark::NONE},
    // Final token sequence
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_marker: true, data: u8:0, mark: Mark::END},
];

const TEST_TOKENS_LEN = u32:11;
const TEST_TOKENS = Token<
    LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH
>[TEST_TOKENS_LEN]: [
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:1, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCHED_SYMBOL, symbol: u8:2, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MATCH, symbol: u8:0, match_offset: u16:1, match_length: u16:3, mark: Mark::NONE},
    // Final token sequence
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::UNMATCHED_SYMBOL, symbol: u8:7, match_offset: u16:0, match_length: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MARKER, symbol: u8:0, match_offset: u16:0, match_length: u16:0, mark: Mark::END},
];

#[test_proc]
proc test_encoder_8k_simple {
    term: chan<bool> out;
    recv_term_r: chan<bool> in;

    init {()}

    config(term: chan<bool> out) {
        let (send_data_s, send_data_r) =
            chan<PlainData<LZ4_SYMBOL_WIDTH>>;
        let (enc_toks_s, enc_toks_r) =
            chan<Token<LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH, LZ4_COUNT_WIDTH>>;
        let (recv_term_s, recv_term_r) =
            chan<bool>;

        spawn test::data_sender<TEST_DATA_LEN, LZ4_SYMBOL_WIDTH>
            (TEST_DATA, send_data_s);
        spawn encoder_8k_modelram(
            send_data_r, enc_toks_s,
        );
        spawn test::token_validator<
            TEST_TOKENS_LEN, LZ4_SYMBOL_WIDTH, LZ4_OFFSET_WIDTH,
            LZ4_COUNT_WIDTH
        >(TEST_TOKENS, enc_toks_r, recv_term_s);

        (term, recv_term_r)
    }

    next (tok: token, state: ()) {
        let (tok, recv_term) = recv(tok, recv_term_r);
        send(tok, term, recv_term);
    }
}
