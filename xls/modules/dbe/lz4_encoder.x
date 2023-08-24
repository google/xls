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


fn fifo_shift_left<FIFO_SIZE: u32, DATA_WIDTH: u32>(
    fifo: uN[DATA_WIDTH][FIFO_SIZE],
    data: uN[DATA_WIDTH]) -> uN[DATA_WIDTH][FIFO_SIZE] {
   let fifo = for (i, arr): (u32, uN[DATA_WIDTH][FIFO_SIZE])
               in u32:1..FIFO_SIZE {
       update(arr, i - u32:1, fifo[i])
   } (uN[DATA_WIDTH][FIFO_SIZE]: [0, ...]);
   let fifo = update(fifo, FIFO_SIZE - u32:1, data);
   fifo
}

fn fifo_shift_right<FIFO_SIZE: u32, DATA_WIDTH: u32>(
     fifo: uN[DATA_WIDTH][FIFO_SIZE],
     data: uN[DATA_WIDTH]) -> uN[DATA_WIDTH][FIFO_SIZE] {
    let fifo = for (i, arr): (u32, uN[DATA_WIDTH][FIFO_SIZE])
                in u32:1..FIFO_SIZE {
        update(arr, i, fifo[i - u32:1])
    } (uN[DATA_WIDTH][FIFO_SIZE]: [0, ...]);
    let fifo = update(fifo, u32:0, data);
    fifo
}

enum FsmSt: u4 {
    RESET = 0,
    HASH_TABLE_CLEAR = 1,
    RESTART = 2,
    FIFO_PREFILL = 3,
    FIFO_POSTFILL = 4,
    START_MATCH_0 = 5,
    START_MATCH_1 = 6,
    CONTINUE_MATCH_0 = 7,
    CONTINUE_MATCH_1 = 8,
    EMIT_SHORT_MATCH = 9,
    EMIT_FINAL_LITERALS = 10,
    EMIT_END = 11,
    ERROR = 12,
}

struct State<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32,
                MINMATCH: u32, FINAL_LITERALS: u32, FIFO_CACHE_SZ: u32,
                FIFO_IN_SZ: u32> {
    fifo_cache: uN[SYM_WIDTH][FIFO_CACHE_SZ],
    fifo_in: uN[SYM_WIDTH][FIFO_IN_SZ],
    fifo_in_count: u32,
    fifo_in_nvalid: u32,
    wp: uN[PTR_WIDTH],
    hb_all_valid: bool,
    pnew: uN[PTR_WIDTH],
    pold: uN[PTR_WIDTH],
    match_nconts: u32,
    ht_ptr: uN[HASH_BITS],
    recycle: bool,
    finalize: bool,
    fsm: FsmSt,
}

fn init_state<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32,
                MINMATCH: u32, FINAL_LITERALS: u32, FIFO_CACHE_SZ: u32,
                FIFO_IN_SZ: u32>()
                -> State<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS,
                            MINMATCH, FINAL_LITERALS, FIFO_CACHE_SZ,
                            FIFO_IN_SZ> {
    State{
        fifo_cache: uN[SYM_WIDTH][FIFO_CACHE_SZ]: [uN[SYM_WIDTH]:0, ...],
        fifo_in: uN[SYM_WIDTH][FIFO_IN_SZ]: [uN[SYM_WIDTH]:0, ...],
        fifo_in_count: u32:0,
        fifo_in_nvalid: u32:0,
        wp: uN[PTR_WIDTH]:0,
        hb_all_valid: false,
        pnew: uN[PTR_WIDTH]:0,
        pold: uN[PTR_WIDTH]:0,
        match_nconts: u32:0,
        ht_ptr: uN[HASH_BITS]:0,
        recycle: false,
        finalize: false,
        fsm: FsmSt::RESET
    }
}

pub proc encoder_base<
        SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32,
        MINMATCH: u32 = {u32:4}, FINAL_LITERALS: u32 = {u32:12},
        FIFO_CACHE_SZ: u32 = {MINMATCH - u32:1},
        FIFO_IN_SZ: u32 = {std::umax(MINMATCH, FINAL_LITERALS + u32:1)},
        HB_RAM_SZ: u32 = {u32:1 << PTR_WIDTH},
        HT_RAM_SZ: u32 = {u32:1 << HASH_BITS}> {
    plain_data: chan<PlainData<SYM_WIDTH>> in;
    encoded_data: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out;

    /// History buffer RAM
    /// Size: (1<<PTR_WIDTH) x SYM_WIDTH
    ram_hb_rd_req: chan<AbstractRamReadReq<PTR_WIDTH, 1>> out;
    ram_hb_rd_resp: chan<AbstractRamReadResp<SYM_WIDTH>> in;
    ram_hb_wr_req: chan<AbstractRamWriteReq<PTR_WIDTH, SYM_WIDTH, 1>> out;
    ram_hb_wr_comp: chan<AbstractRamWriteResp> in;

    /// Hash table RAM
    /// Size: (1<<HASH_BITS) x PTR_WIDTH
    ram_ht_rd_req: chan<AbstractRamReadReq<HASH_BITS, 1>> out;
    ram_ht_rd_resp: chan<AbstractRamReadResp<PTR_WIDTH>> in;
    ram_ht_wr_req: chan<AbstractRamWriteReq<HASH_BITS, PTR_WIDTH, 1>> out;
    ram_ht_wr_comp: chan<AbstractRamWriteResp> in;

    init {
        init_state<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS,
                    MINMATCH, FINAL_LITERALS, FIFO_CACHE_SZ, FIFO_IN_SZ>()
    }

    config (
        // Data in, tokens out
        plain_data: chan<PlainData<SYM_WIDTH>> in,
        encoded_data: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out,
        // RAM
        ram_hb_rd_req: chan<AbstractRamReadReq<PTR_WIDTH, 1>> out,
        ram_hb_rd_resp: chan<AbstractRamReadResp<SYM_WIDTH>> in,
        ram_hb_wr_req: chan<AbstractRamWriteReq<PTR_WIDTH, SYM_WIDTH, 1>> out,
        ram_hb_wr_comp: chan<AbstractRamWriteResp> in,
        ram_ht_rd_req: chan<AbstractRamReadReq<HASH_BITS, 1>> out,
        ram_ht_rd_resp: chan<AbstractRamReadResp<PTR_WIDTH>> in,
        ram_ht_wr_req: chan<AbstractRamWriteReq<HASH_BITS, PTR_WIDTH, 1>> out,
        ram_ht_wr_comp: chan<AbstractRamWriteResp> in,
    ) {
        (
            plain_data, encoded_data,
            ram_hb_rd_req, ram_hb_rd_resp, ram_hb_wr_req, ram_hb_wr_comp,
            ram_ht_rd_req, ram_ht_rd_resp, ram_ht_wr_req, ram_ht_wr_comp,
        )
    }

    next (tok: token, cur: State) {
        type EncToken = Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>;
        type EncData = PlainData<SYM_WIDTH>;
        type HbRamReadReq = AbstractRamReadReq<PTR_WIDTH, 1>;
        type HbRamReadResp = AbstractRamReadResp<SYM_WIDTH>;
        type HbRamWriteReq = AbstractRamWriteReq<PTR_WIDTH, SYM_WIDTH, 1>;
        type HtRamReadReq = AbstractRamReadReq<HASH_BITS, 1>;
        type HtRamReadResp = AbstractRamReadResp<PTR_WIDTH>;
        type HtRamWriteReq = AbstractRamWriteReq<HASH_BITS, PTR_WIDTH, 1>;

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
        let rx_symbol = do_recv && !rx.is_mark;
        let rx_mark = do_recv && rx.is_mark;
        let rx_end = rx_mark && rx.mark == Mark::END;
        let rx_error = rx_mark && dbe::is_error(rx.mark);
        let rx_reset = rx_mark && rx.mark == Mark::RESET;
        let rx_unexpected_mark = rx_mark && !(rx_end || rx_reset || rx_error);

        let upd = if cur.fsm == FsmSt::ERROR || cur.recycle {
            // Do not shift FIFOs / write HB in these states
            upd
        } else if rx_symbol {
            let wp = upd.wp + uN[PTR_WIDTH]:1;
            // Update state, shift input FIFOs
            let upd = State{
                wp: wp,
                fifo_cache:
                    fifo_shift_right(upd.fifo_cache, upd.fifo_in[0]),
                fifo_in:
                    fifo_shift_left(upd.fifo_in, rx.data),
                hb_all_valid:
                    if wp == uN[PTR_WIDTH]:0 {true} else {upd.hb_all_valid},
                fifo_in_count:
                    std::umin(upd.fifo_in_count + u32:1, FIFO_IN_SZ),
                ..upd
            };
            upd
        } else if rx_end {
            // When receiving END marker, shift input FIFO anyhow as we're
            // expected to drop the symbol at OP
            let new_count = std::umin(upd.fifo_in_count + u32:1, FIFO_IN_SZ);
            let upd = State{
                finalize: true,
                fifo_cache:
                    fifo_shift_right(upd.fifo_cache, upd.fifo_in[0]),
                fifo_in:
                    fifo_shift_left(upd.fifo_in, uN[SYM_WIDTH]:0),
                fifo_in_count: new_count,
                fifo_in_nvalid: new_count - u32:1,
                ..upd
            };
            upd
        } else if (
            cur.fsm == FsmSt::FIFO_POSTFILL
            || cur.fsm == FsmSt::EMIT_FINAL_LITERALS
        ) {
            // Feed input FIFO with 0s and shift
            let (new_count, new_nvalid) = if cur.fsm == FsmSt::FIFO_POSTFILL {
                (
                    std::umin(upd.fifo_in_count + u32:1, FIFO_IN_SZ),
                    upd.fifo_in_nvalid
                )
            } else {
                (
                    upd.fifo_in_count,
                    upd.fifo_in_nvalid - u32:1
                )
            };
            let upd = State{
                fifo_in:
                    fifo_shift_left(upd.fifo_in, uN[SYM_WIDTH]:0),
                fifo_in_count: new_count,
                fifo_in_nvalid: new_nvalid,
                ..upd
            };
            upd
        } else {
            upd
        };

        // Calculate origin pointer OP from WP
        let op = ((upd.wp as u32) - FIFO_IN_SZ + u32:1) as uN[PTR_WIDTH];

        // Calculate u32 Fibonacci hash function
        let hsh_input = (upd.fifo_in[3] as u8)
                        ++ (upd.fifo_in[2] as u8)
                        ++ (upd.fifo_in[1] as u8)
                        ++ (upd.fifo_in[0] as u8);
        let hsh32 = hsh_input * u32:2654435761;
        let hsh = (hsh32 >> (u32:32 - HASH_BITS)) as uN[HASH_BITS];

        // NOTE: HT RAM, HB RAM accesses and o_token emission all happen
        // in parallel.

        // Read HT RAM
        let (ht_rd_vld, ht_rd_req) = if cur.fsm == FsmSt::START_MATCH_0 {
            (true, ram::ReadWordReq<u32:1>(hsh))
        } else {
            (false, zero!<HtRamReadReq>())
        };
        let tok_ht = send_if(tok, ram_ht_rd_req, ht_rd_vld, ht_rd_req);
        let (tok_ht, ht_rd_resp) = recv_if(tok_ht, ram_ht_rd_resp, ht_rd_vld,
            zero!<HtRamReadResp>());

        // Write HT RAM
        let (ht_wr_vld, ht_wr_req) = if cur.fsm == FsmSt::START_MATCH_1 {
            (true, ram::WriteWordReq<u32:1>(hsh, op))
        } else if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            (true, ram::WriteWordReq<u32:1>(upd.ht_ptr, uN[PTR_WIDTH]:0))
        } else {
            (false, zero!<HtRamWriteReq>())
        };
        let tok_ht = send_if(tok, ram_ht_wr_req, ht_wr_vld, ht_wr_req);
        let (tok_ht, _) = recv_if(tok_ht, ram_ht_wr_comp, ht_wr_vld,
            zero!<AbstractRamWriteResp>());

        // Update match pointers & HT ptr used to clean hash table
        let upd = if cur.fsm == FsmSt::START_MATCH_0 {
            State {
                pold: ht_rd_resp.data,
                pnew: op,
                ..upd
            }
        } else if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            State {
                ht_ptr: upd.ht_ptr + uN[HASH_BITS]:1,
                ..upd
            }
        } else {
            upd
        };

        // Prepare to check for a match
        let (mchk_do, mchk_pos, mchk_canextend) = if (
            cur.fsm == FsmSt::START_MATCH_1
        ) {
            (
                true,
                upd.pold,
                true
            )
        } else if (cur.fsm == FsmSt::CONTINUE_MATCH_1) {
            (
                true,
                upd.pold + upd.match_nconts as uN[PTR_WIDTH] + uN[PTR_WIDTH]:1,
                upd.match_nconts < ((u32:1 << CNT_WIDTH) - u32:1)
            )
        } else {
            (
                false,
                uN[PTR_WIDTH]:0,
                false
            )
        };

        // Do not match-check unwritten (uninitialized) HB RAM locations
        let mchk_is_hb_written =
            upd.hb_all_valid
            || (mchk_pos >= uN[PTR_WIDTH]:1 && mchk_pos < upd.wp);


        // Read HB RAM
        let (hb_rd_vld, hb_rd_req) = if mchk_do && mchk_is_hb_written {
            (true, ram::ReadWordReq<u32:1>(mchk_pos))
        } else {
            (false, zero!<HbRamReadReq>())
        };
        let tok_hb = send_if(tok, ram_hb_rd_req, hb_rd_vld, hb_rd_req);
        let (tok_hb, hb_rd_resp) = recv_if(tok_hb, ram_hb_rd_resp, hb_rd_vld,
            zero!<HbRamReadResp>());

        // Write HB RAM
        let (hb_wr_vld, hb_wr_req) = if rx_symbol && cur.fsm != FsmSt::ERROR {
            (true, ram::WriteWordReq<u32:1>(upd.wp, rx.data))
        } else {
            (false, zero!<HbRamWriteReq>())
        };
        let tok_hb = send_if(tok, ram_hb_wr_req, hb_wr_vld, hb_wr_req);
        let (tok_hb, _) = recv_if(tok_hb, ram_hb_wr_comp, hb_wr_vld,
            zero!<AbstractRamWriteResp>());

        // Actually check for a match
        let is_match = if mchk_do {
            let sym = hb_rd_resp.data;
            // For match to happen, following criteria have to be met:
            // 1. pos should not point to an unwritten HB entry
            //    (see `mchk_is_hb_written` above)
            // 2. pos should not point between OP and WP inclusive
            let isold = (mchk_pos - op) > (upd.wp - op);
            // 3. sym should match current origin symbol
            let _matches = sym == upd.fifo_in[0];
            // 4. our existing matching string should not be too long
            (mchk_is_hb_written && isold && _matches && mchk_canextend)
        } else {
            false
        };

        // Update match_nconts
        let upd = State{
            match_nconts: if cur.fsm == FsmSt::START_MATCH_1 {
                u32:0
            } else if cur.fsm == FsmSt::CONTINUE_MATCH_1 {
                upd.match_nconts + u32:1
            } else if cur.fsm == FsmSt::EMIT_SHORT_MATCH {
                upd.match_nconts - u32:1
            } else {
                upd.match_nconts
            },
            ..upd
        };
        
        // Handle match termination
        let (
            is_match_terminated,
            match_len,
            match_is_long
        ) = if cur.fsm == FsmSt::CONTINUE_MATCH_1 {
            (
                !is_match || upd.finalize,
                upd.match_nconts,
                upd.match_nconts >= MINMATCH,
            )
        } else {
            (false, u32:0, false)
        };

        // Emit compressed data token
        let zero_tok = zero!<EncToken>();
        let (do_emit, encoded_tok) = if rx_reset {
            // Propagate RESET
            (true, Token{
                kind: TokenKind::MARK,
                mark: Mark::RESET,
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::ERROR {
            // Do not emit anything in error state
            (false, zero_tok)
        } else if rx_error {
            // Propagate error
            (true, Token{
                kind: TokenKind::MARK,
                mark: rx.mark,
                ..zero_tok
            })
        } else if rx_unexpected_mark {
            // Generate error
            (true, Token{
                kind: TokenKind::MARK,
                mark: Mark::ERROR_BAD_MARK,
                ..zero_tok
            })
        } else if (
            cur.fsm == FsmSt::START_MATCH_1 &&
            (!is_match || upd.finalize)
        ) {
            // Emit symbol at OP as literal
            (true, Token{
                kind: TokenKind::LITERAL,
                literal: upd.fifo_in[0],
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::CONTINUE_MATCH_1 && is_match_terminated {
            // If match is long enough, emit a single CP
            // If not, emit symbols from Cache FIFO as literals
            if match_is_long {
                let off = upd.pnew - upd.pold;
                let t = Token {
                    kind: TokenKind::COPY_POINTER,
                    copy_pointer_offset: off - uN[PTR_WIDTH]:1,
                    copy_pointer_count: (match_len - u32:1) as uN[CNT_WIDTH],
                    ..zero_tok
                };
                (true, t) 
            } else {
                (true, Token {
                    kind: TokenKind::LITERAL,
                    literal: upd.fifo_cache[upd.match_nconts - u32:1],
                    ..zero_tok
                })
            }
        } else if cur.fsm == FsmSt::EMIT_SHORT_MATCH {
            (true, Token {
                kind: TokenKind::LITERAL,
                literal: upd.fifo_cache[upd.match_nconts - u32:1],
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::EMIT_FINAL_LITERALS {
            // We dump literals from Input FIFO till nvalid becomes 0
            (true, Token {
                kind: TokenKind::LITERAL,
                literal: upd.fifo_in[u32:0],
                ..zero_tok
            })
        } else if cur.fsm == FsmSt::EMIT_END {
            (true, Token {
                kind: TokenKind::MARK,
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
            init_state<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS,
                        MINMATCH, FINAL_LITERALS, FIFO_CACHE_SZ, FIFO_IN_SZ>()
        } else if cur.fsm == FsmSt::RESTART {
            // Intra-block partial reset, keeping HB and HT intact
            State {
                fifo_in_count: u32:0,
                fifo_in_nvalid: u32:0,
                match_nconts: u32:0,
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
                FsmSt::RESET => FsmSt::HASH_TABLE_CLEAR,
                FsmSt::HASH_TABLE_CLEAR => {
                    if upd.ht_ptr == uN[HASH_BITS]:0 {
                        FsmSt::RESTART
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::RESTART => FsmSt::FIFO_PREFILL,
                FsmSt::FIFO_PREFILL => {
                    if rx_end {
                        if upd.fifo_in_nvalid > u32:0 {
                            if upd.fifo_in_count < FIFO_IN_SZ {
                                FsmSt::FIFO_POSTFILL
                            } else {
                                FsmSt::EMIT_FINAL_LITERALS
                            }
                        } else {
                            // This handles empty input blocks
                            FsmSt::EMIT_END
                        }
                    } else if upd.fifo_in_count >= FIFO_IN_SZ {
                        FsmSt::START_MATCH_0
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::START_MATCH_0 => FsmSt::START_MATCH_1,
                FsmSt::START_MATCH_1 => {
                    if upd.finalize {
                        FsmSt::EMIT_FINAL_LITERALS
                    } else if is_match {
                        FsmSt::CONTINUE_MATCH_0
                    } else {
                        FsmSt::START_MATCH_0
                    }
                },
                FsmSt::CONTINUE_MATCH_0 => FsmSt::CONTINUE_MATCH_1,
                FsmSt::CONTINUE_MATCH_1 => {
                    if is_match_terminated {
                        // Match failed or interrupted
                        if match_is_long || upd.match_nconts == u32:1 {
                            // Copy pointer or a sole literal has been
                            // emitted
                            if upd.finalize {
                                FsmSt::EMIT_FINAL_LITERALS
                            } else {
                                FsmSt::START_MATCH_0
                            }
                        } else {
                            // Still need to emit some literals to terminate
                            // the match
                            FsmSt::EMIT_SHORT_MATCH
                        }
                    } else {
                        // Continue matching
                        FsmSt::CONTINUE_MATCH_0
                    }
                },
                FsmSt::EMIT_SHORT_MATCH => {
                    if upd.match_nconts == u32:1 {
                        if upd.finalize {
                            // Finish block transcription
                            FsmSt::EMIT_FINAL_LITERALS
                        } else {
                            // Restart matching
                            FsmSt::START_MATCH_0
                        }
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::FIFO_POSTFILL => {
                    // It is worth noting that we get into this state only when
                    // finalizing very short blocks (so short that they don't
                    // even fill the input FIFO).
                    if upd.fifo_in_count == FIFO_IN_SZ {
                        FsmSt::EMIT_FINAL_LITERALS
                    } else {
                        cur.fsm
                    }
                },
                FsmSt::EMIT_FINAL_LITERALS => {
                    if upd.fifo_in_nvalid == u32:1 {
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

        // Set 'recycle' flag when we want to "stall" Input FIFO to not lose
        // the symbol at the OP, which is needed for certain state transitions
        let recycle = if (
            upd.fsm == FsmSt::START_MATCH_0
            && cur.fsm != FsmSt::START_MATCH_1
        ) {
            true
        } else if (
            upd.fsm == FsmSt::EMIT_FINAL_LITERALS
            && cur.fsm != FsmSt::EMIT_FINAL_LITERALS
            && cur.fsm != FsmSt::START_MATCH_1
        ) {
            true
        } else {
            false
        };
        
        let upd = State {
            recycle: recycle,
            ..upd
        };

        upd
    }
}

/// Version of `encoder_base` that uses RamModel
/// Intended to be used only for tests
pub proc encoder_base_modelram
    <SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32> {
    init{()}

    config (
        plain_data: chan<PlainData<SYM_WIDTH>> in,
        encoded_data: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out,
    ) {
        let (hb_rd_req_s, hb_rd_req_r) =
            chan<AbstractRamReadReq<PTR_WIDTH, 1>>;
        let (hb_rd_resp_s, hb_rd_resp_r) =
            chan<AbstractRamReadResp<SYM_WIDTH>>;
        let (hb_wr_req_s, hb_wr_req_r) =
            chan<AbstractRamWriteReq<PTR_WIDTH, SYM_WIDTH, 1>>;
        let (hb_wr_comp_s, hb_wr_comp_r) =
            chan<AbstractRamWriteResp>;
        let (ht_rd_req_s, ht_rd_req_r) =
            chan<AbstractRamReadReq<HASH_BITS, 1>>;
        let (ht_rd_resp_s, ht_rd_resp_r) =
            chan<AbstractRamReadResp<PTR_WIDTH>>;
        let (ht_wr_req_s, ht_wr_req_r) =
            chan<AbstractRamWriteReq<HASH_BITS, PTR_WIDTH, 1>>;
        let (ht_wr_comp_s, ht_wr_comp_r) =
            chan<AbstractRamWriteResp>;

        spawn ram::RamModel
            <SYM_WIDTH, {u32:1<<PTR_WIDTH}, SYM_WIDTH>
            (hb_rd_req_r, hb_rd_resp_s, hb_wr_req_r, hb_wr_comp_s);
        spawn ram::RamModel
            <PTR_WIDTH, {u32:1<<HASH_BITS}, PTR_WIDTH>
            (ht_rd_req_r, ht_rd_resp_s, ht_wr_req_r, ht_wr_comp_s);

        spawn encoder_base
            <SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS>
            (
                plain_data, encoded_data,
                hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_comp_r,
                ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_comp_r,
            );
    }

    next (tok: token, state: ()) {
    }
}

/// LZ4 encoder with 8K hash table

const LZ4_SYM_WIDTH = u32:8;
const LZ4_PTR_WIDTH = u32:16;
const LZ4_CNT_WIDTH = u32:16;
const LZ4_HASH_BITS_8K = u32:13;

pub proc encoder_8k {
    init{()}

    config (
        plain_data:
            chan<PlainData<LZ4_SYM_WIDTH>> in,
        encoded_data:
            chan<Token<LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH>> out,
        ram_hb_rd_req: chan<AbstractRamReadReq<LZ4_PTR_WIDTH, 1>> out,
        ram_hb_rd_resp: chan<AbstractRamReadResp<LZ4_SYM_WIDTH>> in,
        ram_hb_wr_req: chan<AbstractRamWriteReq<LZ4_PTR_WIDTH, LZ4_SYM_WIDTH, 1>> out,
        ram_hb_wr_comp: chan<AbstractRamWriteResp> in,
        ram_ht_rd_req: chan<AbstractRamReadReq<LZ4_HASH_BITS_8K, 1>> out,
        ram_ht_rd_resp: chan<AbstractRamReadResp<LZ4_PTR_WIDTH>> in,
        ram_ht_wr_req: chan<AbstractRamWriteReq<LZ4_HASH_BITS_8K, LZ4_PTR_WIDTH, 1>> out,
        ram_ht_wr_comp: chan<AbstractRamWriteResp> in,
    ) {
        spawn encoder_base
            <LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH, LZ4_HASH_BITS_8K>
            (
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
            chan<PlainData<LZ4_SYM_WIDTH>> in,
        encoded_data:
            chan<Token<LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH>> out,
    ) {
        spawn encoder_base_modelram
            <LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH, LZ4_HASH_BITS_8K>
            (plain_data, encoded_data);
    }

    next (tok: token, state: ()) {
    }
}

///
/// Tests
///

import xls.modules.dbe.common_test as test
import xls.modules.dbe.lz4_decoder as dec

const TST_SYMS_LEN = u32:19;
const TST_SYMS = PlainData<8>[TST_SYMS_LEN]: [
    // Check basic LT and CP generation
    PlainData{is_mark: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:2, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:1, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:2, mark: Mark::NONE},
    // Final token sequence - should stay as literals
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: false, data: u8:7, mark: Mark::NONE},
    PlainData{is_mark: true, data: u8:0, mark: Mark::END},
];

const TST_TOKS_LEN = u32:16;
const TST_TOKS =
Token<LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH>[TST_TOKS_LEN]: [
    Token{kind: TokenKind::LITERAL, literal: u8:1, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:2, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::COPY_POINTER, literal: u8:0, copy_pointer_offset: u16:1, copy_pointer_count: u16:3, mark: Mark::NONE},
    // Final literal token sequence
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::LITERAL, literal: u8:7, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::NONE},
    Token{kind: TokenKind::MARK, literal: u8:0, copy_pointer_offset: u16:0, copy_pointer_count: u16:0, mark: Mark::END},
];

#[test_proc]
proc test_encoder_8k_simple {
    term: chan<bool> out;
    recv_term_r: chan<bool> in;

    init {()}

    config(term: chan<bool> out) {
        let (send_data_s, send_data_r) =
            chan<PlainData<LZ4_SYM_WIDTH>>;
        let (enc_toks_s, enc_toks_r) =
            chan<Token<LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH>>;
        let (recv_term_s, recv_term_r) =
            chan<bool>;

        spawn test::data_sender<TST_SYMS_LEN, LZ4_SYM_WIDTH>
            (TST_SYMS, send_data_s);
        spawn encoder_8k_modelram(
            send_data_r, enc_toks_s,
        );
        spawn test::token_validator
            <TST_TOKS_LEN, LZ4_SYM_WIDTH, LZ4_PTR_WIDTH, LZ4_CNT_WIDTH>
            (TST_TOKS, enc_toks_r, recv_term_s);

        (term, recv_term_r)
    }

    next (tok: token, state: ()) {
        let (tok, recv_term) = recv(tok, recv_term_r);
        send(tok, term, recv_term);
    }
}
