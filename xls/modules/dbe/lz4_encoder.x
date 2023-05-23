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

type TokenKind = dbe::TokenKind;
type Token = dbe::Token;
type PlainData = dbe::PlainData;
type RamReq = ram::SinglePortRamReq;
type RamResp = ram::SinglePortRamResp;

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

fn WriteWordReq<NUM_PARTITIONS:u32, DATA_WIDTH:u32, ADDR_WIDTH:u32>(
        addr:uN[ADDR_WIDTH], data:uN[DATA_WIDTH]) ->
        RamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
    RamReq {
        addr: addr,
        data: data,
        write_mask: (),
        read_mask: (),
        we: true,
        re: false,
    }
}
  
fn ReadWordReq<NUM_PARTITIONS:u32, DATA_WIDTH:u32, ADDR_WIDTH:u32>(
        addr:uN[ADDR_WIDTH]) ->
        RamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
    RamReq {
        addr: addr,
        data: uN[DATA_WIDTH]:0,
        write_mask: (),
        read_mask: (),
        we: false,
        re: true,
    }
}

enum FsmSt: u4 {
    RESET = 0,
    HASH_TABLE_CLEAR = 1,
    INITIAL_FILL = 2,
    START_MATCH_0 = 3,
    START_MATCH_1 = 4,
    CONTINUE_MATCH_0 = 5,
    CONTINUE_MATCH_1 = 6,
    TERMINATE_MATCH = 7,
    DUMP_FINAL_LITERALS = 8,
}

struct State<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32,
                MINMATCH: u32, FINAL_LITERALS: u32, FIFO_CACHE_SZ: u32,
                FIFO_IN_SZ: u32> {
    fifo_cache: uN[SYM_WIDTH][FIFO_CACHE_SZ],
    fifo_in: uN[SYM_WIDTH][FIFO_IN_SZ],
    fifo_in_nvalid: u32,
    wp: uN[PTR_WIDTH],
    hb_all_valid: bool,
    pnew: uN[PTR_WIDTH],
    pold: uN[PTR_WIDTH],
    ht_ptr: uN[HASH_BITS],
    recycle: bool,
    eof: bool,
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
        fifo_in_nvalid: u32:0,
        wp: uN[PTR_WIDTH]:0,
        hb_all_valid: false,
        pnew: uN[PTR_WIDTH]:0,
        pold: uN[PTR_WIDTH]:0,
        ht_ptr: uN[HASH_BITS]:0,
        recycle: false,
        eof: false,
        fsm: FsmSt::RESET
    }
}

pub proc encoder<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32, HASH_BITS: u32,
                    MINMATCH: u32 = {u32:4}, FINAL_LITERALS: u32 = {u32:12},
                    FIFO_CACHE_SZ: u32 = {MINMATCH - u32:1},
                    FIFO_IN_SZ: u32 = {std::umax(MINMATCH, FINAL_LITERALS)},
                    HB_RAM_SZ: u32 = {u32:1 << PTR_WIDTH},
                    HT_RAM_SZ: u32 = {u32:1 << HASH_BITS}> {
    i_data: chan<PlainData<SYM_WIDTH>> in;
    o_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out;

    /// History buffer RAM
    /// Size: (1<<PTR_WIDTH) x SYM_WIDTH
    o_ram_hb_req: chan<RamReq<PTR_WIDTH, SYM_WIDTH, 1>> out;
    i_ram_hb_resp: chan<RamResp<SYM_WIDTH>> in;

    /// Hash table RAM
    /// Size: (1<<HASH_BITS) x PTR_WIDTH
    o_ram_ht_req: chan<RamReq<HASH_BITS, PTR_WIDTH, 1>> out;
    i_ram_ht_resp: chan<RamResp<PTR_WIDTH>> in;

    init {
        init_state<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS,
                    MINMATCH, FINAL_LITERALS, FIFO_CACHE_SZ, FIFO_IN_SZ>()
    }

    config (
        // Data in, tokens out
        i_data: chan<PlainData<SYM_WIDTH>> in,
        o_encoded: chan<Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>> out,
        // RAM
        o_ram_hb_req: chan<RamReq<PTR_WIDTH, SYM_WIDTH, 1>> out,
        i_ram_hb_resp: chan<RamResp<SYM_WIDTH>> in,
        o_ram_ht_req: chan<RamReq<HASH_BITS, PTR_WIDTH, 1>> out,
        i_ram_ht_resp: chan<RamResp<PTR_WIDTH>> in,
    ) {
        // let (o_ram_hb_req, i_ram_hb_req) = chan<RamReq<PTR_WIDTH, SYM_WIDTH, u32:1>>;
        // let (o_ram_hb_resp, i_ram_hb_resp) = chan<RamResp<SYM_WIDTH>>;
        // spawn ram::RamModel<SYM_WIDTH, HB_RAM_SZ, SYM_WIDTH>(
        //     i_hb_read_req, o_hb_read_resp, i_hb_write_req, o_hb_write_resp);

        // let (o_ram_ht_req, i_ram_ht_req) = chan<RamReq<HASH_BITS, PTR_WIDTH, u32:1>>;
        // let (o_ram_ht_resp, i_ram_ht_resp) = chan<RamResp<PTR_WIDTH>>;
        // spawn ram::RamModel<PTR_WIDTH, HT_RAM_SZ, PTR_WIDTH>(
        //     i_ht_read_req, o_ht_read_resp, i_ht_write_req, o_ht_write_resp);

        (
            i_data, o_encoded,
            o_ram_hb_req, i_ram_hb_resp,
            o_ram_ht_req, i_ram_ht_resp,
        )
    }

    next (tok: token, cur: State) {
        trace_fmt!("---STEP--- fsm: {}, recycle: {}", cur.fsm , cur.recycle);

        let upd = cur;

        // Read new symbol from input
        let (upd, tok) = if (!cur.recycle
                && (cur.fsm == FsmSt::INITIAL_FILL
                || cur.fsm == FsmSt::START_MATCH_0
                || cur.fsm == FsmSt::CONTINUE_MATCH_0)) {
            if upd.eof != false {
                fail!("unexpected_eof", ())
            } else {
                ()
            };

            let (tok, data) = recv(tok, i_data);
            let sym = data.sym;
            let is_eof = data.last;
            // Push symbol to the HB
            let wp = upd.wp + uN[PTR_WIDTH]:1;
            let tok = send(tok, o_ram_hb_req,
                WriteWordReq<u32:1>(wp, sym));
            let (tok, _) = recv(tok, i_ram_hb_resp);
            // Update state, shift input FIFOs
            let upd = State{
                eof: is_eof,
                wp: wp,
                fifo_cache:
                    fifo_shift_right(upd.fifo_cache, upd.fifo_in[0]),
                fifo_in:
                    fifo_shift_left(upd.fifo_in, sym),
                hb_all_valid:
                    if wp == uN[PTR_WIDTH]:0 {true} else {upd.hb_all_valid},
                fifo_in_nvalid:
                    std::umin(upd.fifo_in_nvalid + u32:1, FIFO_IN_SZ),
                ..upd
            };
            (upd, tok)
        } else {
            (upd, tok)
        };
        
        let upd = State{
            recycle: false,
            ..upd
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
        // Hash table lookup & update in two states
        let (upd, tok) = if cur.fsm == FsmSt::START_MATCH_0 {
            // HT lookup, pold, pnew update
            let tok = send(tok, o_ram_ht_req,
                ReadWordReq<u32:1, PTR_WIDTH>(hsh));
            let (tok, resp) = recv(tok, i_ram_ht_resp);
            let upd = State {
                pold: resp.data,
                pnew: op,
                ..upd
            };
            (upd, tok)
        } else if cur.fsm == FsmSt::START_MATCH_1 {
            // HT update
            let tok = send(tok, o_ram_ht_req,
                WriteWordReq<u32:1>(hsh, op));
            let (tok, _) = recv(tok, i_ram_ht_resp);
            (upd, tok)
        } else {
            (upd, tok)
        };

        // Calcualte length of current match
        let match_len = op - upd.pnew;
        let match_is_long = match_len >= (MINMATCH as uN[PTR_WIDTH]);

        // Check for match
        let (is_match, tok) = if (cur.fsm == FsmSt::START_MATCH_1
                            || cur.fsm == FsmSt::CONTINUE_MATCH_1) {
            // Fetch historical symbol from HB to check for match
            let pos = upd.pold + match_len;
            let tok = send(tok, o_ram_hb_req,
                ReadWordReq<u32:1, SYM_WIDTH>(pos));
            let (tok, resp) = recv(tok, i_ram_hb_resp);
            let sym = resp.data;
            // For match to happen, following criteria have to be met:
            // 1. pos should not point to an unwritten HB entry
            let iswritten = upd.hb_all_valid || (pos < upd.wp);
            // 2. pos should not point between OP and WP inclusive
            let isold = (pos - op) > (upd.wp - op);
            // 3. sym should match current origin symbol
            let matches = sym == upd.fifo_in[0];
            // 4. our existing matching string should not be too long
            let canextend = match_len as u32 < (u32:1 << CNT_WIDTH);
            let is_match = iswritten && isold && matches && canextend;
            (is_match, tok)
        } else {
            (false, tok)
        };

        // Emit compressed data token
        let (upd, tok) = if cur.fsm == FsmSt::START_MATCH_1 {
            if !upd.eof && !is_match {
                // Emit symbol at OP as literal
                let tok = send(tok, o_encoded, Token {
                    kind: TokenKind::LT,
                    last: false,
                    lt_sym: upd.fifo_in[0],
                    cp_off: uN[PTR_WIDTH]:0,
                    cp_cnt: uN[CNT_WIDTH]:0,
                });
                (upd, tok)
            } else {
                (upd, tok)
            }
        } else if cur.fsm == FsmSt::TERMINATE_MATCH {
            // If match is long enough, emit a single CP
            // If not, emit symbols from Cache FIFO as literals
            //let _ = assert_lt(uN[PTR_WIDTH]:0, match_len);
            if match_is_long {
                let off = upd.pnew - upd.pold;
                //let _ = assert_lt(uN[PTR_WIDTH]:0, off);
                let tok = send(tok, o_encoded, Token {
                    kind: TokenKind::CP,
                    last: false,
                    lt_sym: uN[SYM_WIDTH]:0,
                    cp_off: off - uN[PTR_WIDTH]:1,
                    cp_cnt: (match_len - uN[PTR_WIDTH]:1) as uN[CNT_WIDTH],
                });
                (upd, tok) 
            } else {
                // For literal emission we use match_len as index into
                // Cache FIFO. After each produced literal, we increment
                // pnew to decrease match_len by 1.
                let fifo_idx = match_len - uN[PTR_WIDTH]:1;
                let tok = send(tok, o_encoded, Token {
                    kind: TokenKind::LT,
                    last: false,
                    lt_sym: upd.fifo_cache[fifo_idx],
                    cp_off: uN[PTR_WIDTH]:0,
                    cp_cnt: uN[CNT_WIDTH]:0,
                });
                let upd = State {
                    pnew: upd.pnew + uN[PTR_WIDTH]:1,
                    ..upd
                };
                (upd, tok)
            }
        } else if cur.fsm == FsmSt::DUMP_FINAL_LITERALS {
            // We dump literals from Input FIFO till nvalid becomes 0
            //let _ = assert_lt(upd.fifo_in_nvalid, FIFO_IN_SZ + u32:1);
            if upd.fifo_in_nvalid > u32:0 {
                let fifo_idx = FIFO_IN_SZ - upd.fifo_in_nvalid;
                let tok = send(tok, o_encoded, Token {
                    kind: TokenKind::LT,
                    last: (upd.fifo_in_nvalid == u32:1),
                    lt_sym: upd.fifo_in[fifo_idx],
                    cp_off: uN[PTR_WIDTH]:0,
                    cp_cnt: uN[CNT_WIDTH]:0,
                });
                let upd = State {
                    fifo_in_nvalid: upd.fifo_in_nvalid - u32:1,
                    ..upd
                };
                (upd, tok)
            } else {
                (upd, tok)
            }
        } else {
            (upd, tok)
        };

        // Clear hash table (one cell at a time)
        let (upd, tok) = if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            let tok = send(tok, o_ram_ht_req,
                WriteWordReq<u32:1>(upd.ht_ptr, uN[PTR_WIDTH]:0));
            let (tok, _) = recv(tok, i_ram_ht_resp);
            let upd = State {
                ht_ptr: upd.ht_ptr + uN[HASH_BITS]:1,
                ..upd
            };
            (upd, tok)
        } else {
            (upd, tok)
        };

        // Handle reset requests
        let upd = if cur.fsm == FsmSt::RESET {
            init_state<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH, HASH_BITS,
                        MINMATCH, FINAL_LITERALS, FIFO_CACHE_SZ, FIFO_IN_SZ>()
        } else {
            upd
        };

        // State change logic
        let fsm = if cur.fsm == FsmSt::RESET {
            FsmSt::HASH_TABLE_CLEAR
        } else if cur.fsm == FsmSt::HASH_TABLE_CLEAR {
            if upd.ht_ptr == uN[HASH_BITS]:0 {
                FsmSt::INITIAL_FILL
            } else {
                upd.fsm
            }
        } else if cur.fsm == FsmSt::INITIAL_FILL {
            if upd.eof {
                FsmSt::DUMP_FINAL_LITERALS
            } else if upd.fifo_in_nvalid >= FIFO_IN_SZ {
                FsmSt::START_MATCH_0
            } else {
                upd.fsm
            }
        } else if cur.fsm == FsmSt::START_MATCH_0 {
            FsmSt::START_MATCH_1
        } else if cur.fsm == FsmSt::START_MATCH_1 {
            if upd.eof {
                FsmSt::DUMP_FINAL_LITERALS
            } else if is_match {
                FsmSt::CONTINUE_MATCH_0
            } else {
                FsmSt::START_MATCH_0
            }
        } else if cur.fsm == FsmSt::CONTINUE_MATCH_0 {
            FsmSt::CONTINUE_MATCH_1
        } else if cur.fsm == FsmSt::CONTINUE_MATCH_1 {
            if upd.eof || !is_match {
                FsmSt::TERMINATE_MATCH
            } else {
                FsmSt::CONTINUE_MATCH_0
            }
        } else if cur.fsm == FsmSt::TERMINATE_MATCH {
            if match_is_long || match_len == uN[PTR_WIDTH]:1 {
                if upd.eof {
                    FsmSt::DUMP_FINAL_LITERALS
                } else {
                    FsmSt::START_MATCH_0
                }
            } else {
                upd.fsm
            }
        } else if cur.fsm == FsmSt::DUMP_FINAL_LITERALS {
            if upd.fifo_in_nvalid == u32:0 {
                // Reset and prepare for compression of next block
                FsmSt::RESET
            } else {
                upd.fsm
            }
        } else {
            fail!("unknown_FsmSt", upd.fsm)
        };

        // Set 'recycle' flag when moving into START_MATCH_0 state from
        // any state other than START_MATCH_1.
        let recycle =
            if (upd.fsm == FsmSt::START_MATCH_0
                && cur.fsm != FsmSt::START_MATCH_1)
                { true } else { upd.recycle };
        
        let upd = State {
            recycle: recycle,
            ..upd
        };

        upd
    }
}

///
/// Temporary for testing during development
/// TBD: remove!
///

pub const LZ4C_SYM_WIDTH = u32:8;
pub const LZ4C_PTR_WIDTH = u32:16;
pub const LZ4C_CNT_WIDTH = u32:16;
pub const LZ4C_HASH_BITS = u32:13;

pub proc encoder_large {
    // Data, tokens
    i_data: chan<PlainData<LZ4C_SYM_WIDTH>> in;
    o_encoded: chan<Token<LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH>> out;
    // RAMs
    o_ram_hb_req: chan<RamReq<LZ4C_PTR_WIDTH, LZ4C_SYM_WIDTH, 1>> out;
    i_ram_hb_resp: chan<RamResp<LZ4C_SYM_WIDTH>> in;
    o_ram_ht_req: chan<RamReq<LZ4C_HASH_BITS, LZ4C_PTR_WIDTH, 1>> out;
    i_ram_ht_resp: chan<RamResp<LZ4C_PTR_WIDTH>> in;

    init{}

    config (
        i_data: chan<PlainData<LZ4C_SYM_WIDTH>> in,
        o_encoded: chan<Token<LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH>> out,
        o_ram_hb_req: chan<RamReq<LZ4C_PTR_WIDTH, LZ4C_SYM_WIDTH, 1>> out,
        i_ram_hb_resp: chan<RamResp<LZ4C_SYM_WIDTH>> in,
        o_ram_ht_req: chan<RamReq<LZ4C_HASH_BITS, LZ4C_PTR_WIDTH, 1>> out,
        i_ram_ht_resp: chan<RamResp<LZ4C_PTR_WIDTH>> in,
    ) {
        spawn encoder
            <LZ4C_SYM_WIDTH, LZ4C_PTR_WIDTH, LZ4C_CNT_WIDTH, LZ4C_HASH_BITS>
            (
                i_data, o_encoded,
                o_ram_hb_req, i_ram_hb_resp,
                o_ram_ht_req, i_ram_ht_resp
            );

        (
            i_data, o_encoded,
            o_ram_hb_req, i_ram_hb_resp,
            o_ram_ht_req, i_ram_ht_resp
        )
    }

    next(tok: token, st: ()) {
    }
}
