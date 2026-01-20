// Copyright 2024-2025 The XLS Authors
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

// This file contains implementation of MatchFinder
//
// # Match Finder Processing Logic
//
// 1. If the symbol is already in the hash table (i.e., it was seen before), then:
//     1.1 Compare the sequence in the history buffer (starting after the previous occurrence of the symbol)
//         with the current sequence in memory.
//     1.2 If a match is found (i.e., the sequences are equal up to some point),
//         and the match is at least as long as the configured minimum:
//             1.2.1 Emit a sequence (A, B, C) where:
//                 A - literals collected since the last match
//                 B - offset of the match (from the hash table)
//                 C - length of the match
//
// 2. If no valid match is found (either not long enough or the symbol is new),
//    add the current symbol to the hash table.
//
// 3. Copy all processed symbols from memory into the history buffer.
//
// 4. After processing the last symbol, if it wasn’t part of a match,
//    emit a final sequence for the remaining (trailing) literals.
// ---
//
// ## Example
//
// - Minimum match length: 3
// - Input symbols: [1, 2, 8, 5, 6, 1, 8, 5, 6, 9]
//
// 1. The first few symbols are all new, so they are added to literals.
//    - Literals: [1, 2, 8, 5, 6]
//    - Sequences: []
//
// 2. At index 5, the symbol `1` is seen again.
//    - Check for a match in the history buffer:
//      - History: [1, 2, 8, 5, 6], current: [1, 8, 5, 6...]
//      - Only the first symbol matches (match length = 1), which is below the minimum.
//    - So, `1` is added to literals.
//    - Literals: [1, 2, 8, 5, 6, 1]
//    - Sequences: []
//
// 3. At index 6, the symbol `8` is seen again.
//    - Matching sequence found: [8, 5, 6] (length = 3)
//    - A sequence is emitted:
//      - Offset: 6 (index of current `8`)
//      - Match start: index 3 (prior occurrence of `8`)
//      - Match length: 3
//    - Literals: [1, 2, 8, 5, 6, 1]
//    - Sequences: [(6, 4, 3)]
//
// 4. At index 9, symbol `9` is new and added to literals.
//    - It's the last symbol, so trailing literals are flushed.
//    - A dummy sequence is added to mark the trailing literal.
//    - Literals: [1, 2, 8, 5, 6, 1, 9]
//    - Sequences: [(6, 4, 3), (1, 0, 0)]
//
//  NOTE: assumes DATA_W <= SEQUENCE_RECORD_W

import std;

import xls.examples.ram;
import xls.modules.zstd.common;
import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.memory.mem_writer;
import xls.modules.zstd.memory.axi;
import xls.modules.zstd.memory.axi_ram_reader;
import xls.modules.zstd.memory.axi_ram_writer;
import xls.modules.zstd.history_buffer;
import xls.modules.zstd.hash_table;
import xls.modules.zstd.aligned_parallel_ram;
import xls.modules.zstd.mem_reader_simple_arbiter;
import xls.modules.zstd.mem_writer_simple_arbiter;
import xls.modules.zstd.sequence_encoder;
import xls.modules.zstd.memory.mem_writer_data_downscaler;

const KEY_WIDTH = common::SYMBOL_WIDTH;
const DEFAULT_HT_KEY_W = u32:32;
const DEFAULT_HB_DATA_W = u32:64;
const WIDER_BUS_DATA_W = u32:64;

pub struct ZstdParams<HT_SIZE_W: u32> {
    num_entries_log2: uN[HT_SIZE_W],
}

pub enum MatchFinderRespStatus : u1 {
    OK = 0,
    ERROR = 1,
}

pub struct MatchFinderReq<HT_SIZE_W: u32, ADDR_W: u32> {
    input_addr: uN[ADDR_W],
    input_size: uN[ADDR_W],
    output_lit_addr: uN[ADDR_W],
    output_seq_addr: uN[ADDR_W],
    zstd_params: ZstdParams<HT_SIZE_W>,
}

pub struct MatchFinderResp {
    status: MatchFinderRespStatus, // indicate the state of the operation
    lit_cnt: u32, // number of literals
    seq_cnt: u32, // number of sequences
}

struct MatchFinderInternalLoopState<
    ADDR_W: u32,
    HT_SIZE_W: u32,
> {
    active: bool,
    lit_addr_offset: uN[ADDR_W],
    seq_addr_offset: uN[ADDR_W],
    lit_cnt: u32,
    seq_cnt: u32,
    lit_since_last_sequence: u32,
    hb_entry_cnt: u32,

    current_input_addr: uN[ADDR_W],
    output_lit_addr: uN[ADDR_W],
    output_seq_addr: uN[ADDR_W],
    num_entries_log2: uN[HT_SIZE_W],
    remaining_size: uN[ADDR_W],
}

struct MatchFinderInternalHistoryCompareConf<
    ADDR_W: u32,
    HB_OFFSET_W: u32
>{
    current_hb_offset: uN[HB_OFFSET_W],
    current_input_addr: uN[ADDR_W],
}

struct MatchFinderInternalHistoryCompareResp{
    match_found: bool,
    match_length: u32,
    last: bool
}

struct MatchFinderInternalHistoryCompareState<
    ADDR_W: u32,
    HB_OFFSET_W: u32
> {
    active: bool,
    conf: MatchFinderInternalHistoryCompareConf<ADDR_W, HB_OFFSET_W>,
    match_length: u32,
}

struct MatchFinderInternalLiteralCopyConf<ADDR_W: u32> {
    input_addr: uN[ADDR_W],
    output_addr: uN[ADDR_W],
    size: uN[ADDR_W],
    move_literals: bool,
}

struct MatchFinderInternalLiteralCopyState<ADDR_W: u32> {
    active: bool,
    conf: MatchFinderInternalLiteralCopyConf<ADDR_W>,
}

struct MatchFinderInternalLiteralCopyResp<ADDR_W: u32> {
    addr: uN[ADDR_W]
}


proc MatchFinderInternalLiteralCopy<
    ADDR_W: u32, DATA_W: u32, HB_SIZE: u32, MIN_SEQ_LEN: u32,
    HB_DATA_W: u32 = {DEFAULT_HB_DATA_W},
    HB_OFFSET_W: u32 = {std::clog2(HB_SIZE)}
>
{
    type State = MatchFinderInternalLiteralCopyState<ADDR_W>;
    type Config = MatchFinderInternalLiteralCopyConf<ADDR_W>;
    type Resp = MatchFinderInternalLiteralCopyResp<ADDR_W>;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<WIDER_BUS_DATA_W, ADDR_W>;
    type NarrowMemWriterDataPacket = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;

    type Addr = uN[ADDR_W];

    hb_wr_req_s: chan<HistoryBufferWrReq> out;
    hb_wr_resp_r: chan<HistoryBufferWrResp> in;
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;
    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_packet_s: chan<MemWriterDataPacket> out;
    mem_wr_resp_r: chan<MemWriterResp> in;
    conf_r: chan<Config> in;
    resp_s: chan<Resp> out;

    init { zero!<State>() }

    config(
        hb_wr_req_s: chan<HistoryBufferWrReq> out,
        hb_wr_resp_r: chan<HistoryBufferWrResp> in,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_packet_s: chan<MemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
        conf_r: chan<Config> in,
        resp_s: chan<Resp> out,
    ) {
        (
            hb_wr_req_s, hb_wr_resp_r,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_packet_s, mem_wr_resp_r,
            conf_r, resp_s
        )
    }

    next(state: State) {
        const KEY_WIDTH_BYTES = KEY_WIDTH as Addr / Addr:8;
        let tok = join();
        let conf = state.conf;

        if !state.active {
            let (tok, conf) = recv(tok, conf_r);
            State { active: true, conf: conf }
        } else if conf.size == Addr:0 {
            // copy done
            let tok = send(tok, resp_s, Resp { addr: state.conf.input_addr });
            zero!<State>()
        } else {
            let tok = send(
                tok, mem_rd_req_s, MemReaderReq {
                    addr: conf.input_addr,
                    length: KEY_WIDTH_BYTES
                }
            );

            // place in history buffer
            let (tok, mem_resp) = recv(tok, mem_rd_resp_r);
            let tok = send(tok, hb_wr_req_s, HistoryBufferWrReq {
                data: mem_resp.data as uN[HB_DATA_W]
            });
            let (tok1, _) = recv(tok, hb_wr_resp_r);

            let status = if conf.move_literals {
                // place in literals memory
                let tok2 = send(tok, mem_wr_req_s, MemWriterReq {
                    addr: conf.output_addr,
                    length: KEY_WIDTH_BYTES
                });
                let tok2 = send(tok, mem_wr_packet_s, MemWriterDataPacket {
                    data: mem_resp.data as uN[HB_DATA_W],
                    length: KEY_WIDTH_BYTES,
                    last: true
                });
                let (tok2, resp) = recv(tok, mem_wr_resp_r);
                trace_fmt!("|- writing literal value {:#x} at {:#x} -> {:#x} [{}]", mem_resp.data, conf.input_addr, conf.output_addr, resp.status);
                resp.status
            } else {
                mem_writer::MemWriterRespStatus::OKAY
            };

            State {
                active: status == mem_writer::MemWriterRespStatus::OKAY,
                conf: Config{
                    input_addr: conf.input_addr + KEY_WIDTH_BYTES,
                    output_addr: conf.output_addr + KEY_WIDTH_BYTES,
                    size: conf.size - KEY_WIDTH_BYTES,
                    ..conf
                }
            }
        }

    }
}
proc MatchFinderInternalHistoryCompare<
    ADDR_W: u32, DATA_W: u32, HB_SIZE: u32, MIN_SEQ_LEN: u32,
    HB_DATA_W: u32 = {DEFAULT_HB_DATA_W},
    HB_OFFSET_W: u32 = {std::clog2(HB_SIZE)},
>{
    type State = MatchFinderInternalHistoryCompareState<ADDR_W, HB_OFFSET_W>;
    type Config = MatchFinderInternalHistoryCompareConf<ADDR_W, HB_OFFSET_W>;
    type Resp = MatchFinderInternalHistoryCompareResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<WIDER_BUS_DATA_W, ADDR_W>;
    type NarrowMemWriterDataPacket = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;
    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;

    type Addr = uN[ADDR_W];

    hb_rd_req_s: chan<HistoryBufferRdReq> out;
    hb_rd_resp_r: chan<HistoryBufferRdResp> in;
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;
    conf_r: chan<Config> in;
    resp_s: chan<Resp> out;

    init {zero!<State>()}
    config(
        hb_rd_req_s: chan<HistoryBufferRdReq> out,
        hb_rd_resp_r: chan<HistoryBufferRdResp> in,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        conf_r: chan<Config> in,
        resp_s: chan<Resp> out,
    ) {
        (
            hb_rd_req_s, hb_rd_resp_r,
            mem_rd_req_s, mem_rd_resp_r,
            conf_r, resp_s
        )
    }

    next(state: State) {
        const KEY_WIDTH_BYTES = KEY_WIDTH as Addr / Addr:8;
        const HB_DATA_BYTES = HB_DATA_W / u32:8;

        let tok = join();
        if !state.active {
            let (tok, conf) = recv(tok, conf_r);
            State {
                active: true,
                match_length: u32:1,
                conf: conf,
            }

        } else {
            let tok = send(
                tok, mem_rd_req_s, MemReaderReq {
                    addr: state.conf.current_input_addr,
                    length: KEY_WIDTH_BYTES
                }
            );
            let (tok, mem_resp) = recv(tok, mem_rd_resp_r);

            let tok = send(tok, hb_rd_req_s, HistoryBufferRdReq {
                offset: state.conf.current_hb_offset,
            });

            let (tok, hb_resp) = recv(tok, hb_rd_resp_r);

            trace_fmt!("comparing {:#x} ({:#x})=={:#x}", hb_resp.data, state.conf.current_hb_offset, mem_resp.data);

            let diff = hb_resp.data != mem_resp.data as uN[HB_DATA_W] || state.conf.current_hb_offset == uN[HB_OFFSET_W]:0;
            let match_length_increment_cornercase = if state.conf.current_hb_offset == uN[HB_OFFSET_W]:0 && hb_resp.data == mem_resp.data as uN[HB_DATA_W] { u32:1 } else { u32:0 };

            let tok = send_if(tok, resp_s, diff, Resp {
                match_found: true,
                match_length: state.match_length + match_length_increment_cornercase,
                last: mem_resp.last
            });

            State {
                active: !diff,
                match_length: u32:1 + state.match_length,
                conf: Config {
                    current_input_addr: state.conf.current_input_addr + KEY_WIDTH_BYTES,
                    current_hb_offset: state.conf.current_hb_offset - HB_DATA_BYTES as uN[HB_OFFSET_W],
                }
            }
        }

    }
}

pub proc MatchFinder<
    ADDR_W: u32, DATA_W: u32, HT_SIZE: u32, HB_SIZE: u32, MIN_SEQ_LEN: u32,
    DATA_W_LOG2: u32 = {std::clog2(DATA_W + u32:1)},

    HT_KEY_W: u32 = {DEFAULT_HT_KEY_W},
    HT_VALUE_W: u32 = {HT_KEY_W + ADDR_W},
    HT_SIZE_W: u32 = {std::clog2(HT_SIZE + u32:1)},

    HB_DATA_W: u32 = {DEFAULT_HB_DATA_W},
    HB_OFFSET_W: u32 = {std::clog2(HB_SIZE)},
> {
    type State = MatchFinderInternalLoopState<ADDR_W, HT_SIZE_W>;
    type Resp = MatchFinderResp;
    type Req = MatchFinderReq<HT_SIZE_W, ADDR_W>;
    type RespStatus = MatchFinderRespStatus;

    type HistoryCompareConfig = MatchFinderInternalHistoryCompareConf<ADDR_W, HB_OFFSET_W>;
    type HistoryCompareResp = MatchFinderInternalHistoryCompareResp;
    type HistoryCompareConfig = MatchFinderInternalHistoryCompareConf<ADDR_W, HB_OFFSET_W>;
    type CopyConfig = MatchFinderInternalLiteralCopyConf<ADDR_W>;
    type CopyResp = MatchFinderInternalLiteralCopyResp<ADDR_W>;

    type HashTableRdReq = hash_table::HashTableReadReq<HT_KEY_W, HT_SIZE, HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<HT_KEY_W, HT_VALUE_W, HT_SIZE, HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type MemReaderReq = mem_reader::MemReaderReq<ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<DATA_W, ADDR_W>;
    type MemWriterReq = mem_writer::MemWriterReq<ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<WIDER_BUS_DATA_W, ADDR_W>;
    type NarrowMemWriterDataPacket = mem_writer::MemWriterDataPacket<DATA_W, ADDR_W>;
    type MemWriterRespStatus = mem_writer::MemWriterRespStatus;

    type Addr = uN[ADDR_W];

    ht_rd_req_s: chan<HashTableRdReq> out;
    ht_rd_resp_r: chan<HashTableRdResp> in;
    ht_wr_req_s: chan<HashTableWrReq> out;
    ht_wr_resp_r: chan<HashTableWrResp> in;

    mem_wr_req_s: chan<MemWriterReq> out;
    mem_wr_packet_s: chan<MemWriterDataPacket> out;
    mem_wr_resp_r: chan<MemWriterResp> in;
    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    hb_mlen_conf_s: chan<HistoryCompareConfig> out;
    hb_mlen_resp_r: chan<HistoryCompareResp> in;
    copy_conf_s: chan<CopyConfig> out;
    copy_resp_r: chan<CopyResp> in;

    init { zero!<State>() }

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_packet_s: chan<NarrowMemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,
        ht_rd_req_s: chan<HashTableRdReq> out,
        ht_rd_resp_r: chan<HashTableRdResp> in,
        ht_wr_req_s: chan<HashTableWrReq> out,
        ht_wr_resp_r: chan<HashTableWrResp> in,
        hb_rd_req_s: chan<HistoryBufferRdReq> out,
        hb_rd_resp_r: chan<HistoryBufferRdResp> in,
        hb_wr_req_s: chan<HistoryBufferWrReq> out,
        hb_wr_resp_r: chan<HistoryBufferWrResp> in,
    ) {
        let (hb_mlen_conf_s, hb_reader_conf_r) = chan<HistoryCompareConfig, u32:1>("hb_reader_conf");
        let (hb_reader_resp_s, hb_mlen_resp_r) = chan<HistoryCompareResp, u32:1>("hb_reader_resp");
        let (copy_conf_s, copy_conf_r) = chan<CopyConfig, u32:1>("copy_conf");
        let (copy_resp_s, copy_resp_r) = chan<CopyResp, u32:1>("copy_resp");

        let (n_mem_wr_req_s, n_mem_wr_req_r) = chan<MemWriterReq, u32:1>[2]("n_req");
        let (n_mem_wr_data_s, n_mem_wr_data_r) = chan<MemWriterDataPacket, u32:1>[2]("n_data");
        let (n_mem_wr_resp_s, n_mem_wr_resp_r) = chan<MemWriterResp, u32:1>[2]("n_resp");

        let (n_mem_rd_req_s, n_mem_rd_req_r) = chan<MemReaderReq, u32:1>[3]("n_mem_rd_req");
        let (n_mem_rd_resp_s, n_mem_rd_resp_r) = chan<MemReaderResp, u32:1>[3]("n_mem_rd_resp");

        let (wider_mem_wr_packet_s, wider_mem_wr_packet_r) = chan<MemWriterDataPacket, u32:1>("wider_mem_wr_packet");

        spawn mem_writer_data_downscaler::MemWriterDataDownscaler<
            ADDR_W, WIDER_BUS_DATA_W, DATA_W,
        > (wider_mem_wr_packet_r, mem_wr_packet_s);


        spawn mem_writer_simple_arbiter::MemWriterSimpleArbiter<ADDR_W, WIDER_BUS_DATA_W, u32:2>
        (
            n_mem_wr_req_r, n_mem_wr_data_r, n_mem_wr_resp_s,
            mem_wr_req_s, wider_mem_wr_packet_s, mem_wr_resp_r,
        );

        spawn mem_reader_simple_arbiter::MemReaderSimpleArbiter<ADDR_W, DATA_W, u32:3> (
            n_mem_rd_req_r, n_mem_rd_resp_s,
            mem_rd_req_s, mem_rd_resp_r,
        );

        spawn MatchFinderInternalHistoryCompare<
            ADDR_W, DATA_W,
            HB_SIZE, MIN_SEQ_LEN,
        >
        (
            hb_rd_req_s, hb_rd_resp_r,
            n_mem_rd_req_s[0], n_mem_rd_resp_r[0],
            hb_reader_conf_r, hb_reader_resp_s,
        );

        spawn MatchFinderInternalLiteralCopy<
            ADDR_W, DATA_W,
            HB_SIZE, MIN_SEQ_LEN,
        >
        (
            hb_wr_req_s, hb_wr_resp_r,
            n_mem_rd_req_s[1], n_mem_rd_resp_r[1],
            n_mem_wr_req_s[0], n_mem_wr_data_s[0], n_mem_wr_resp_r[0],
            copy_conf_r, copy_resp_s
        );

        (
            ht_rd_req_s, ht_rd_resp_r,
            ht_wr_req_s, ht_wr_resp_r,
            n_mem_wr_req_s[1], n_mem_wr_data_s[1], n_mem_wr_resp_r[1],
            n_mem_rd_req_s[2], n_mem_rd_resp_r[2],
            req_r, resp_s,
            hb_mlen_conf_s, hb_mlen_resp_r,
            copy_conf_s, copy_resp_r
        )
    }

    next(state: State) {
        const KEY_WIDTH_BYTES = KEY_WIDTH as Addr / Addr:8;
        const HB_DATA_BYTES = HB_DATA_W / u32:8;
        const SEQUENCE_RECORD_B = sequence_encoder::SEQUENCE_RECORD_W / u32:8;
        let tok = join();

        if !state.active {
            let (tok, req) = recv(tok, req_r);
            trace_fmt!("Started encoding");
            State {
                active: true,
                output_lit_addr: req.output_lit_addr,
                output_seq_addr: req.output_seq_addr,
                current_input_addr: req.input_addr,
                num_entries_log2: req.zstd_params.num_entries_log2,
                remaining_size: req.input_size,
                ..zero!<State>()
            }
        } else if state.remaining_size == uN[ADDR_W]:0 {
            // https://datatracker.ietf.org/doc/html/rfc8878#name-sequences_section
            // When all sequences are decoded, if there are literals left in the Literals_Section, these bytes are added at the end of the block.
            trace_fmt!("Literals left ({}, {}, {})", state.lit_since_last_sequence, u32:0, u32:0);
            trace_fmt!("Encoding finished");
            let tok = send(tok, resp_s, Resp {
                status: RespStatus::OK,
                lit_cnt: state.lit_cnt,
                seq_cnt: state.seq_cnt,
            });
            zero!<State>()
        } else {
            let tok = send(
                tok, mem_rd_req_s, MemReaderReq {
                    addr: state.current_input_addr,
                    length: KEY_WIDTH_BYTES
                }
            );
            let (tok, resp) = recv(tok, mem_rd_resp_r);
            let ht_key = resp.data as uN[HT_KEY_W];

            // Step 1: read from hashtable and check if there's a match (+ corner cases)
            let tok = send(tok, ht_rd_req_s, HashTableRdReq {
                num_entries_log2: state.num_entries_log2,
                key: ht_key,
            });
            let (tok, ht_rd_resp) = recv(tok, ht_rd_resp_r);
            let matched = ht_rd_resp.is_match &&
            resp.data == (ht_rd_resp.value >> ADDR_W) as uN[DATA_W] &&
            state.current_input_addr - ht_rd_resp.value as Addr > KEY_WIDTH_BYTES;

            // Step 2: compute history buffer offset and get match length
            let offset = ((state.current_input_addr - (ht_rd_resp.value as Addr)) as uN[HB_OFFSET_W] - uN[HB_OFFSET_W]:2) * HB_DATA_BYTES as uN[HB_OFFSET_W];

            // cornercase: prevent inter-block matches
            let matched = matched && offset as u32 < HB_DATA_BYTES * state.hb_entry_cnt;

            let tok = send_if(tok, hb_mlen_conf_s, matched, HistoryCompareConfig {
                current_hb_offset: offset,
                current_input_addr: state.current_input_addr + KEY_WIDTH_BYTES,
            });
            let (tok, cmp_resp) = recv_if(tok, hb_mlen_resp_r, matched, HistoryCompareResp {
                match_length: Addr:1,
                ..zero!<HistoryCompareResp>()
            });

            // Step 3: compute parameters for sequence and write it if it's long enough
            let encode_sequence = cmp_resp.match_length >= MIN_SEQ_LEN && matched;
            let match_offset =  offset / HB_DATA_BYTES as uN[HB_OFFSET_W] + uN[HB_OFFSET_W]:2;

            let tok3 = send_if(tok, mem_wr_req_s, encode_sequence, MemWriterReq {
                addr: state.seq_addr_offset + state.output_seq_addr,
                length: SEQUENCE_RECORD_B,
            });

            let seq_bytes = sequence_encoder::serialize_sequence(
                sequence_encoder::Sequence {
                    literals_len: state.lit_since_last_sequence as u16,
                    // https://datatracker.ietf.org/doc/html/rfc8878#section-3.1.1.3.2.1.1
                    // match length code is incremented by 3 when decoding (TODO: make sure baselines are handled correctly)
                    match_len: cmp_resp.match_length as u16 - u16:3,
                    // https://datatracker.ietf.org/doc/html/rfc8878#name-sequence-execution
                    // if Offset_Value > 3, then the offset is Offset_Value - 3 => offset += 3 (we don't use repeat offsets for now)
                    offset: match_offset as u16 + u16:3
                }
            ) as uN[WIDER_BUS_DATA_W];

            let tok3 = send_if(tok3, mem_wr_packet_s, encode_sequence, MemWriterDataPacket {
                data: seq_bytes,
                length: SEQUENCE_RECORD_B,
                last: true,
            });
            let (tok3, wr_resp) = recv_if(tok3, mem_wr_resp_r, encode_sequence, zero!<MemWriterResp>());
            if encode_sequence {
                trace_fmt!("|- writing sequence ({}, {}, {}) (serialized: {:#x}), at {:#x} [{}], hb offset was {}",
                    state.lit_since_last_sequence, match_offset, cmp_resp.match_length, seq_bytes,
                    state.seq_addr_offset + state.output_seq_addr,wr_resp.status, offset
                );
            } else {};

            // Step 4: copy processed literals to history buffer (and literals memory if sequence wasn't written)
            let copy_conf = CopyConfig {
                input_addr: state.current_input_addr,
                output_addr: state.lit_addr_offset + state.output_lit_addr,
                size: cmp_resp.match_length * KEY_WIDTH_BYTES,
                move_literals: !encode_sequence
            };
            let tok4 = send(tok, copy_conf_s, copy_conf);
            let (tok4, copy_resp) = recv(tok4, copy_resp_r);

            // Step 5: write current data in hashtable
            let tok5 = send(tok, ht_wr_req_s, HashTableWrReq {
                num_entries_log2: state.num_entries_log2,
                key: ht_key,
                value: (
                    (ht_key) ++
                    (state.current_input_addr)
                ),
            });
            let (tok5, ht_wr_resp) = recv(tok, ht_wr_resp_r);
            let tok = join(tok3, tok4, tok5);
            // Step 6: compute next iteration parameters
            let address_increment = KEY_WIDTH_BYTES * cmp_resp.match_length;
            let lit_cnt_increment = if encode_sequence { uN[ADDR_W]:0 } else { cmp_resp.match_length };
            let lit_address_increment = if encode_sequence { uN[ADDR_W]:0 } else { KEY_WIDTH_BYTES * cmp_resp.match_length };
            let seq_address_increment = if encode_sequence { SEQUENCE_RECORD_B } else { uN[ADDR_W]:0 };
            let seq_cnt_increment = encode_sequence as Addr; // if encode_sequence: seq_cnt ++
            let lit_since_last_sequence = if encode_sequence { Addr:0 } else { state.lit_since_last_sequence + cmp_resp.match_length };
            let remaining_size = if state.remaining_size < address_increment { u32:0 } else { state.remaining_size - address_increment };
            let hb_entry_cnt = state.hb_entry_cnt + cmp_resp.match_length as u32;

            State {
                active: true,
                current_input_addr: copy_resp.addr,
                remaining_size: remaining_size,
                lit_addr_offset: state.lit_addr_offset + lit_address_increment,
                lit_cnt: state.lit_cnt + lit_cnt_increment,
                lit_since_last_sequence: lit_since_last_sequence,
                seq_addr_offset: state.seq_addr_offset + seq_address_increment,
                seq_cnt: state.seq_cnt + seq_cnt_increment,
                hb_entry_cnt: hb_entry_cnt,
                ..state
            }
        }
    }
}

const INST_ADDR_W = u32:32;
const INST_DATA_W = u32:64;
const INST_MIN_SEQ_LEN = u32:3;
const INST_DATA_W_LOG2 = std::clog2(INST_DATA_W + u32:1);

const INST_HT_SIZE = u32:512;
const INST_HT_SIZE_W = std::clog2(INST_HT_SIZE + u32:1);
const INST_HT_KEY_W = u32:32;
const INST_HT_VALUE_W = INST_HT_KEY_W + INST_ADDR_W; // original symbol + address
const INST_HB_DATA_W = u32:64;
const INST_HB_SIZE = u32:1024;
const INST_HB_OFFSET_W = std::clog2(INST_HB_SIZE);

proc MatchFinderInst {
    type MemReaderReq = mem_reader::MemReaderReq<INST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<INST_DATA_W, INST_ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<INST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<WIDER_BUS_DATA_W, INST_ADDR_W>;
    type NarrowMemWriterDataPacket = mem_writer::MemWriterDataPacket<INST_DATA_W, INST_ADDR_W>;

    type HashTableRdReq = hash_table::HashTableReadReq<INST_HT_KEY_W, INST_HT_SIZE, INST_HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<INST_HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<INST_HT_KEY_W, INST_HT_VALUE_W, INST_HT_SIZE, INST_HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<INST_HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<INST_HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<INST_HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type Req = MatchFinderReq<INST_HT_SIZE_W, INST_ADDR_W>;
    type Resp = MatchFinderResp;

    config (
        // Req & Resp
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,

        // Access to input
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        // Output
        mem_wr_req_s: chan<MemWriterReq> out,
        mem_wr_packet_s: chan<NarrowMemWriterDataPacket> out,
        mem_wr_resp_r: chan<MemWriterResp> in,

        // HashTable RAM interface
        ht_rd_req_s: chan<HashTableRdReq> out,
        ht_rd_resp_r: chan<HashTableRdResp> in,
        ht_wr_req_s: chan<HashTableWrReq> out,
        ht_wr_resp_r: chan<HashTableWrResp> in,

        // HistoryBuffer RAM interface
        hb_rd_req_s: chan<HistoryBufferRdReq> out,
        hb_rd_resp_r: chan<HistoryBufferRdResp> in,
        hb_wr_req_s: chan<HistoryBufferWrReq> out,
        hb_wr_resp_r: chan<HistoryBufferWrResp> in,
    ) {
        spawn MatchFinder<
            INST_ADDR_W, INST_DATA_W, INST_HT_SIZE, INST_HB_SIZE, INST_MIN_SEQ_LEN,
            INST_DATA_W_LOG2,
            INST_HT_KEY_W, INST_HT_VALUE_W, INST_HT_SIZE_W,
            INST_HB_DATA_W, INST_HB_OFFSET_W,
        >(
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_packet_s, mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        );
    }

    init {}

    next (state: ()) {}
}

const COCOTB_ADDR_W = u32:32;
const COCOTB_DATA_W = u32:64;
const COCOTB_MIN_SEQ_LEN = u32:3;
const COCOTB_DATA_W_LOG2 = std::clog2(COCOTB_DATA_W + u32:1);
const COCOTB_DEST_W = u32:8;
const COCOTB_ID_W = u32:8;

const COCOTB_HT_SIZE = u32:512;
const COCOTB_HT_SIZE_W = std::clog2(COCOTB_HT_SIZE + u32:1);
const COCOTB_HT_KEY_W = u32:32;
const COCOTB_HT_VALUE_W = COCOTB_HT_KEY_W + COCOTB_ADDR_W;
const COCOTB_HT_HASH_W = std::clog2(COCOTB_HT_SIZE);
const COCOTB_HT_RAM_DATA_W = COCOTB_HT_VALUE_W + u32:1;
const COCOTB_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const COCOTB_HT_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_HT_RAM_WORD_PARTITION_SIZE, COCOTB_HT_RAM_DATA_W);
const COCOTB_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const COCOTB_HT_RAM_INITIALIZED = true;

const COCOTB_HB_DATA_W = u32:64;
const COCOTB_HB_SIZE = u32:1024;
const COCOTB_HB_OFFSET_W = std::clog2(COCOTB_HB_SIZE);
const COCOTB_HB_RAM_NUM = u32:8;
const COCOTB_HB_RAM_SIZE = COCOTB_HB_SIZE / COCOTB_HB_RAM_NUM;
const COCOTB_HB_RAM_DATA_W = COCOTB_HB_DATA_W / COCOTB_HB_RAM_NUM;
const COCOTB_HB_RAM_ADDR_W = std::clog2(COCOTB_HB_RAM_SIZE);
const COCOTB_HB_RAM_PARTITION_SIZE = COCOTB_HB_RAM_DATA_W;
const COCOTB_HB_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_HB_RAM_PARTITION_SIZE, COCOTB_HB_RAM_DATA_W);
const COCOTB_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const COCOTB_HB_RAM_INITIALIZED = true;

const COCOTB_RAM_DATA_W = COCOTB_DATA_W;
const COCOTB_RAM_SIZE = u32:2048;
const COCOTB_RAM_ADDR_W = COCOTB_ADDR_W;
const COCOTB_RAM_PARTITION_SIZE = COCOTB_RAM_DATA_W / u32:8;
const COCOTB_RAM_NUM_PARTITIONS = ram::num_partitions(COCOTB_RAM_PARTITION_SIZE, COCOTB_RAM_DATA_W);
const COCOTB_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const COCOTB_RAM_INITIALIZED = true;
const COCOTB_RAM_ASSERT_VALID_READ = true;

proc MatchFinderCocotbInst {
    type MemReaderReq = mem_reader::MemReaderReq<COCOTB_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<COCOTB_DATA_W, COCOTB_ADDR_W>;

    type MemWriterReq = mem_writer::MemWriterReq<COCOTB_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<COCOTB_DATA_W, COCOTB_ADDR_W>;

    type HashTableRdReq = hash_table::HashTableReadReq<COCOTB_HT_KEY_W, COCOTB_HT_SIZE, COCOTB_HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<COCOTB_HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<COCOTB_HT_KEY_W, COCOTB_HT_VALUE_W, COCOTB_HT_SIZE, COCOTB_HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;
    type HashTableRamRdReq = ram::ReadReq<COCOTB_HT_HASH_W, COCOTB_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<COCOTB_HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<COCOTB_HT_HASH_W, COCOTB_HT_RAM_DATA_W, COCOTB_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<COCOTB_HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<COCOTB_HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<COCOTB_HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;
    type HistoryBufferRamRdReq = ram::ReadReq<COCOTB_HB_RAM_ADDR_W, COCOTB_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<COCOTB_HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<COCOTB_HB_RAM_ADDR_W, COCOTB_HB_RAM_DATA_W, COCOTB_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;

    type AxiAr = axi::AxiAr<COCOTB_ADDR_W, COCOTB_ID_W>;
    type AxiR = axi::AxiR<COCOTB_DATA_W, COCOTB_ID_W>;
    type AxiAw = axi::AxiAw<COCOTB_ADDR_W, COCOTB_ID_W>;
    type AxiW = axi::AxiW<COCOTB_DATA_W, COCOTB_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<COCOTB_ID_W>;

    type Req = MatchFinderReq<COCOTB_HT_SIZE_W, COCOTB_ADDR_W>;
    type Resp = MatchFinderResp;

    config (
        // Req & Resp
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        // External AXI bussies
        axi_aw_s: chan<AxiAw> out,
        axi_w_s: chan<AxiW> out,
        axi_b_r: chan<AxiB> in,
        axi_ar_s: chan<AxiAr> out,
        axi_r_r: chan<AxiR> in,
        // hash table
        ht_ram_rd_req_s: chan<HashTableRamRdReq> out,
        ht_ram_rd_resp_r: chan<HashTableRamRdResp> in,
        ht_ram_wr_req_s: chan<HashTableRamWrReq> out,
        ht_ram_wr_resp_r: chan<HashTableRamWrResp> in,
        // history buffer
        hb_ram_rd_req_s: chan<HistoryBufferRamRdReq>[8] out,
        hb_ram_rd_resp_r: chan<HistoryBufferRamRdResp>[8] in,
        hb_ram_wr_req_s: chan<HistoryBufferRamWrReq>[8] out,
        hb_ram_wr_resp_r: chan<HistoryBufferRamWrResp>[8] in,
    ) {
        let (ht_rd_req_s, ht_rd_req_r) = chan<HashTableRdReq, u32:1>("ht_rd_req");
        let (ht_rd_resp_s, ht_rd_resp_r) = chan<HashTableRdResp, u32:1>("ht_rd_resp");
        let (ht_wr_req_s, ht_wr_req_r) = chan<HashTableWrReq, u32:1>("ht_wr_req");
        let (ht_wr_resp_s, ht_wr_resp_r) = chan<HashTableWrResp, u32:1>("ht_wr_resp");
        let (hb_rd_req_s, hb_rd_req_r) = chan<HistoryBufferRdReq, u32:1>("hb_rd_req");
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<HistoryBufferRdResp, u32:1>("hb_rd_resp");
        let (hb_wr_req_s, hb_wr_req_r) = chan<HistoryBufferWrReq, u32:1>("hb_wr_req");
        let (hb_wr_resp_s, hb_wr_resp_r) = chan<HistoryBufferWrResp, u32:1>("hb_wr_resp");
        let (mem_wr_req_s, mem_wr_req_r) = chan<MemWriterReq, u32:1>("mem_wr_req");
        let (mem_wr_data_s, mem_wr_data_r) = chan<MemWriterDataPacket, u32:1>("mem_wr_data");
        let (mem_wr_resp_s, mem_wr_resp_r) = chan<MemWriterResp, u32:1>("mem_wr_resp");
        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq, u32:1>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp, u32:1>("mem_rd_resp");

        spawn hash_table::HashTable<COCOTB_HT_KEY_W, COCOTB_HT_VALUE_W, COCOTB_HT_SIZE, COCOTB_HT_SIZE_W>(
            ht_rd_req_r, ht_rd_resp_s,
            ht_wr_req_r, ht_wr_resp_s,
            ht_ram_rd_req_s, ht_ram_rd_resp_r,
            ht_ram_wr_req_s, ht_ram_wr_resp_r,
        );

        spawn history_buffer::HistoryBuffer<COCOTB_HB_SIZE, COCOTB_HB_DATA_W>(
            hb_rd_req_r, hb_rd_resp_s,
            hb_wr_req_r, hb_wr_resp_s,
            hb_ram_rd_req_s, hb_ram_rd_resp_r,
            hb_ram_wr_req_s, hb_ram_wr_resp_r,
        );

        spawn mem_reader::MemReader<
            COCOTB_DATA_W, COCOTB_ADDR_W, COCOTB_DEST_W, COCOTB_ID_W
        >(
            mem_rd_req_r, mem_rd_resp_s,
            axi_ar_s, axi_r_r,
        );

        spawn mem_writer::MemWriter<
            COCOTB_ADDR_W, COCOTB_DATA_W, COCOTB_DEST_W, COCOTB_ID_W, u32:0
        >(
            mem_wr_req_r, mem_wr_data_r,
            axi_aw_s, axi_w_s, axi_b_r,
            mem_wr_resp_s
        );

        spawn MatchFinder<
            COCOTB_ADDR_W, COCOTB_DATA_W, COCOTB_HT_SIZE, COCOTB_HB_SIZE, COCOTB_MIN_SEQ_LEN,
            COCOTB_DATA_W_LOG2,
            COCOTB_HT_KEY_W, COCOTB_HT_VALUE_W, COCOTB_HT_SIZE_W,
            COCOTB_HB_DATA_W, COCOTB_HB_OFFSET_W,
        >(
            req_r, resp_s,
            mem_rd_req_s, mem_rd_resp_r,
            mem_wr_req_s, mem_wr_data_s, mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        );
    }

    init {}
    next (state: ()) { }
}

const TEST_ADDR_W = u32:32;
const TEST_DATA_W = u32:64;
const TEST_MIN_SEQ_LEN = u32:3;
const TEST_HT_SIZE = u32:512;
const TEST_HB_SIZE = u32:1024;
const TEST_DATA_W_LOG2 = std::clog2(TEST_DATA_W + u32:1);
const TEST_DEST_W = u32:8;
const TEST_ID_W = u32:8;

const TEST_HT_KEY_W = u32:32; // TODO: hash computation doesn't work properly for smaller keys
const TEST_HT_VALUE_W = TEST_HT_KEY_W + TEST_ADDR_W; // original symbol + address
const TEST_HT_HASH_W = std::clog2(TEST_HT_SIZE);
const TEST_HT_RAM_DATA_W = TEST_HT_VALUE_W + u32:1; // value + valid
const TEST_HT_RAM_WORD_PARTITION_SIZE = u32:1;
const TEST_HT_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HT_RAM_WORD_PARTITION_SIZE, TEST_HT_RAM_DATA_W);
const TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HT_RAM_INITIALIZED = true;
const TEST_HT_SIZE_W = std::clog2(TEST_HT_SIZE + u32:1);

const TEST_HB_RAM_NUM = u32:8;
const TEST_HB_DATA_W = u32:64;
const TEST_HB_OFFSET_W = std::clog2(TEST_HB_SIZE);
const TEST_HB_RAM_SIZE = TEST_HB_SIZE / TEST_HB_RAM_NUM;
const TEST_HB_RAM_DATA_W = TEST_HB_DATA_W / TEST_HB_RAM_NUM;
const TEST_HB_RAM_ADDR_W = std::clog2(TEST_HB_RAM_SIZE);
const TEST_HB_RAM_PARTITION_SIZE = TEST_HB_RAM_DATA_W;
const TEST_HB_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_HB_RAM_PARTITION_SIZE, TEST_HB_RAM_DATA_W);
const TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_HB_RAM_INITIALIZED = true;

const TEST_RAM_DATA_W = TEST_DATA_W;
const TEST_RAM_SIZE = u32:2048;
const TEST_RAM_ADDR_W = TEST_ADDR_W;
const TEST_RAM_PARTITION_SIZE = TEST_RAM_DATA_W / u32:8;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_RAM_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_RAM_INITIALIZED = true;
const TEST_RAM_ASSERT_VALID_READ = true;

const TEST_OUTPUT_LIT_ADDR = uN[TEST_ADDR_W]:0x100;
const TEST_OUTPUT_SEQ_ADDR = uN[TEST_ADDR_W]:0x200;

const TEST_DATA = [
    u8:0x1,
    u8:0xA,
    u8:0xB,
    u8:0xC,
    u8:0x1,
    u8:0x2,
    u8:0x1,
    u8:0x4,
    u8:0xA,
    u8:0xB,
    u8:0xC,
    u8:0x5,
    u8:0xA,
    u8:0xB,
    u8:0xC,
    u8:0x9,
];

#[test_proc]
proc MatchFinderTest {
    // Memory Reader + Input
    type MemReaderReq = mem_reader::MemReaderReq<TEST_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_DATA_W, TEST_ADDR_W>;

    type InputBufferRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type InputBufferRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type InputBufferRamWrResp = ram::WriteResp;

    type OutputBufferRamRdReq = ram::ReadReq<TEST_RAM_ADDR_W, TEST_RAM_NUM_PARTITIONS>;
    type OutputBufferRamRdResp = ram::ReadResp<TEST_RAM_DATA_W>;
    type OutputBufferRamWrReq = ram::WriteReq<TEST_RAM_ADDR_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type OutputBufferRamWrResp = ram::WriteResp;

    type AxiAr = axi::AxiAr<TEST_ADDR_W, TEST_ID_W>;
    type AxiR = axi::AxiR<TEST_DATA_W, TEST_ID_W>;
    type AxiAw = axi::AxiAw<TEST_ADDR_W, TEST_ID_W>;
    type AxiW = axi::AxiW<TEST_DATA_W, TEST_RAM_NUM_PARTITIONS>;
    type AxiB = axi::AxiB<TEST_ID_W>;

    // Memory Writer

    type MemWriterReq = mem_writer::MemWriterReq<TEST_ADDR_W>;
    type MemWriterResp = mem_writer::MemWriterResp;
    type MemWriterDataPacket = mem_writer::MemWriterDataPacket<TEST_DATA_W, TEST_ADDR_W>;

    // Hash Table

    type HashTableRdReq = hash_table::HashTableReadReq<TEST_HT_KEY_W, TEST_HT_SIZE, TEST_HT_SIZE_W>;
    type HashTableRdResp = hash_table::HashTableReadResp<TEST_HT_VALUE_W>;
    type HashTableWrReq = hash_table::HashTableWriteReq<TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE, TEST_HT_SIZE_W>;
    type HashTableWrResp = hash_table::HashTableWriteResp;

    type HashTableRamRdReq = ram::ReadReq<TEST_HT_HASH_W, TEST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamRdResp = ram::ReadResp<TEST_HT_RAM_DATA_W>;
    type HashTableRamWrReq = ram::WriteReq<TEST_HT_HASH_W, TEST_HT_RAM_DATA_W, TEST_HT_RAM_NUM_PARTITIONS>;
    type HashTableRamWrResp = ram::WriteResp;

    // History Buffer

    type HistoryBufferRdReq = history_buffer::HistoryBufferReadReq<TEST_HB_OFFSET_W>;
    type HistoryBufferRdResp = history_buffer::HistoryBufferReadResp<TEST_HB_DATA_W>;
    type HistoryBufferWrReq = history_buffer::HistoryBufferWriteReq<TEST_HB_DATA_W>;
    type HistoryBufferWrResp = history_buffer::HistoryBufferWriteResp;

    type HistoryBufferRamRdReq = ram::ReadReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamRdResp = ram::ReadResp<TEST_HB_RAM_DATA_W>;
    type HistoryBufferRamWrReq = ram::WriteReq<TEST_HB_RAM_ADDR_W, TEST_HB_RAM_DATA_W, TEST_HB_RAM_NUM_PARTITIONS>;
    type HistoryBufferRamWrResp = ram::WriteResp;

    // Match Finder

    type Req = MatchFinderReq<TEST_HT_SIZE_W, TEST_ADDR_W>;
    type Resp = MatchFinderResp;

    // Other

    type NumEntriesLog2 = uN[TEST_HT_SIZE_W];
    type RamAddr = uN[TEST_RAM_ADDR_W];
    type RamMask = uN[TEST_RAM_NUM_PARTITIONS];

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;


    input_mem_wr_req_s: chan<MemWriterReq> out;
    input_mem_wr_data_s: chan<MemWriterDataPacket> out;
    input_mem_wr_resp_r: chan<MemWriterResp> in;
    output_rd_req_s: chan<OutputBufferRamRdReq> out;
    output_rd_resp_r: chan<OutputBufferRamRdResp>in;

    config(terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        // procs
        let (ht_ram_rd_req_s, ht_ram_rd_req_r) = chan<HashTableRamRdReq>("ht_ram_rd_req");
        let (ht_ram_rd_resp_s, ht_ram_rd_resp_r) = chan<HashTableRamRdResp>("ht_ram_rd_resp");
        let (ht_ram_wr_req_s, ht_ram_wr_req_r) = chan<HashTableRamWrReq>("ht_ram_wr_req");
        let (ht_ram_wr_resp_s, ht_ram_wr_resp_r) = chan<HashTableRamWrResp>("ht_ram_wr_resp");
        let (ht_rd_req_s, ht_rd_req_r) = chan<HashTableRdReq>("ht_rd_req");
        let (ht_rd_resp_s, ht_rd_resp_r) = chan<HashTableRdResp>("ht_rd_resp");
        let (ht_wr_req_s, ht_wr_req_r) = chan<HashTableWrReq>("ht_wr_req");
        let (ht_wr_resp_s, ht_wr_resp_r) = chan<HashTableWrResp>("ht_wr_resp");
        let (hb_rd_req_s, hb_rd_req_r) = chan<HistoryBufferRdReq>("hb_rd_req");
        let (hb_rd_resp_s, hb_rd_resp_r) = chan<HistoryBufferRdResp>("hb_rd_resp");
        let (hb_wr_req_s, hb_wr_req_r) = chan<HistoryBufferWrReq>("hb_wr_req");
        let (hb_wr_resp_s, hb_wr_resp_r) = chan<HistoryBufferWrResp>("hb_wr_resp");
        // rams
        let (hb_ram_rd_req_s, hb_ram_rd_req_r) = chan<HistoryBufferRamRdReq>[8]("hb_ram_rd_req");
        let (hb_ram_rd_resp_s, hb_ram_rd_resp_r) = chan<HistoryBufferRamRdResp>[8]("hb_ram_rd_resp");
        let (hb_ram_wr_req_s, hb_ram_wr_req_r) = chan<HistoryBufferRamWrReq>[8]("hb_ram_wr_req");
        let (hb_ram_wr_resp_s, hb_ram_wr_resp_r) = chan<HistoryBufferRamWrResp>[8]("hb_ram_wr_resp");
        let (input_ram_rd_req_s, input_ram_rd_req_r) = chan<InputBufferRamRdReq>("input_ram_rd_req");
        let (input_ram_rd_resp_s, input_ram_rd_resp_r) = chan<InputBufferRamRdResp>("input_ram_rd_resp");
        let (input_ram_wr_req_s, input_ram_wr_req_r) = chan<InputBufferRamWrReq>("input_ram_wr_req");
        let (input_ram_wr_resp_s, input_ram_wr_resp_r) = chan<InputBufferRamWrResp>("input_ram_wr_resp");
        let (output_rd_req_s, output_rd_req_r) = chan<OutputBufferRamRdReq>("output_rd_req");
        let (output_rd_resp_s, output_rd_resp_r) = chan<OutputBufferRamRdResp>("output_rd_resp");
        let (output_wr_req_s, output_wr_req_r) = chan<OutputBufferRamWrReq>("output_wr_req");
        let (output_wr_resp_s, output_wr_resp_r) = chan<OutputBufferRamWrResp>("output_wr_resp");
        // axi
        let (input_axi_ar_s, input_axi_ar_r) = chan<AxiAr>("input_axi_ar");
        let (input_axi_r_s, input_axi_r_r) = chan<AxiR>("input_axi_r");
        let (input_axi_aw_s, input_axi_aw_r) = chan<AxiAw>("input_axi_aw");
        let (input_axi_w_s, input_axi_w_r) = chan<AxiW>("input_axi_w");
        let (input_axi_b_s, input_axi_b_r) = chan<AxiB>("input_axi_b");
        let (output_axi_aw_s, output_axi_aw_r) = chan<AxiAw>("output_axi_aw");
        let (output_axi_w_s, output_axi_w_r) = chan<AxiW>("output_axi_w");
        let (output_axi_b_s, output_axi_b_r) = chan<AxiB>("output_axi_b");
        let (input_mem_rd_req_s, input_mem_rd_req_r) = chan<MemReaderReq>("input_mem_rd_req");
        let (input_mem_rd_resp_s,input_mem_rd_resp_r) = chan<MemReaderResp>("input_mem_rd_resp");
        let (input_mem_wr_req_s, input_mem_wr_req_r) = chan<MemWriterReq>("input_mem_wr_req");
        let (input_mem_wr_data_s, input_mem_wr_data_r) = chan<MemWriterDataPacket>("input_mem_wr_data");
        let (input_mem_wr_resp_s, input_mem_wr_resp_r) = chan<MemWriterResp>("input_mem_wr_resp");
        let (output_mem_wr_req_s, output_mem_wr_req_r) = chan<MemWriterReq>("output_mem_wr_req");
        let (output_mem_wr_data_s,output_mem_wr_data_r) = chan<MemWriterDataPacket>("output_mem_wr_data");
        let (output_mem_wr_resp_s,output_mem_wr_resp_r) = chan<MemWriterResp>("output_mem_wr_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            input_ram_rd_req_r, input_ram_rd_resp_s,
            input_ram_wr_req_r, input_ram_wr_resp_s,
        );

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_RAM_SIZE, TEST_RAM_PARTITION_SIZE,
            TEST_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_RAM_INITIALIZED,
            TEST_RAM_ASSERT_VALID_READ, TEST_RAM_ADDR_W,
        >(
            output_rd_req_r, output_rd_resp_s,
            output_wr_req_r, output_wr_resp_s
        );

        spawn ram::RamModel<
            TEST_HT_RAM_DATA_W, TEST_HT_SIZE, TEST_HT_RAM_WORD_PARTITION_SIZE,
            TEST_HT_RAM_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_HT_RAM_INITIALIZED
        >(
            ht_ram_rd_req_r, ht_ram_rd_resp_s,
            ht_ram_wr_req_r, ht_ram_wr_resp_s
        );

        unroll_for! (i, _) : (u32, ()) in u32:0..u32:8 {
            spawn ram::RamModel<
            TEST_HB_RAM_DATA_W, TEST_HB_RAM_SIZE, TEST_HB_RAM_PARTITION_SIZE,
            TEST_HB_RAM_SIMULTANEOUS_RW_BEHAVIOR, TEST_HB_RAM_INITIALIZED
            >(
                hb_ram_rd_req_r[i], hb_ram_rd_resp_s[i],
                hb_ram_wr_req_r[i], hb_ram_wr_resp_s[i],
            );
        }(());

        spawn axi_ram_reader::AxiRamReader<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_DEST_W, TEST_ID_W,
            TEST_RAM_SIZE,
        >(
            input_axi_ar_r, input_axi_r_s,
            input_ram_rd_req_s, input_ram_rd_resp_r,
        );

        spawn axi_ram_writer::AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_ID_W, TEST_RAM_SIZE, TEST_ADDR_W
        >(
            input_axi_aw_r, input_axi_w_r, input_axi_b_s,
            input_ram_wr_req_s, input_ram_wr_resp_r
        );

        spawn axi_ram_writer::AxiRamWriter<
            TEST_ADDR_W, TEST_DATA_W,
            TEST_ID_W, TEST_RAM_SIZE, TEST_ADDR_W
        >(
            output_axi_aw_r, output_axi_w_r, output_axi_b_s,
            output_wr_req_s, output_wr_resp_r
        );

        spawn mem_reader::MemReader<
        TEST_DATA_W, TEST_ADDR_W, TEST_DEST_W, TEST_ID_W,
        >(
            input_mem_rd_req_r, input_mem_rd_resp_s,
            input_axi_ar_s, input_axi_r_r,
        );

        spawn mem_writer::MemWriter<
        TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, u32:0
        >(
            output_mem_wr_req_r, output_mem_wr_data_r,
            output_axi_aw_s, output_axi_w_s, output_axi_b_r,
            output_mem_wr_resp_s
        );

        spawn mem_writer::MemWriter<
        TEST_ADDR_W, TEST_DATA_W, TEST_DEST_W, TEST_ID_W, u32:0
        >(
            input_mem_wr_req_r, input_mem_wr_data_r,
            input_axi_aw_s, input_axi_w_s, input_axi_b_r,
            input_mem_wr_resp_s
        );


        spawn hash_table::HashTable<TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE, TEST_HT_SIZE_W>(
            ht_rd_req_r, ht_rd_resp_s,
            ht_wr_req_r, ht_wr_resp_s,
            ht_ram_rd_req_s, ht_ram_rd_resp_r,
            ht_ram_wr_req_s, ht_ram_wr_resp_r,
        );

        spawn history_buffer::HistoryBuffer<TEST_HB_SIZE, TEST_HB_DATA_W>(
            hb_rd_req_r, hb_rd_resp_s,
            hb_wr_req_r, hb_wr_resp_s,
            hb_ram_rd_req_s, hb_ram_rd_resp_r,
            hb_ram_wr_req_s, hb_ram_wr_resp_r,
        );

        spawn MatchFinder<
            TEST_ADDR_W, TEST_DATA_W, TEST_HT_SIZE, TEST_HB_SIZE, TEST_MIN_SEQ_LEN,
            TEST_DATA_W_LOG2,
            TEST_HT_KEY_W, TEST_HT_VALUE_W, TEST_HT_SIZE_W,
            TEST_HB_DATA_W, TEST_HB_OFFSET_W,
        >(
            req_r, resp_s,
            input_mem_rd_req_s, input_mem_rd_resp_r,
            output_mem_wr_req_s, output_mem_wr_data_s, output_mem_wr_resp_r,
            ht_rd_req_s, ht_rd_resp_r, ht_wr_req_s, ht_wr_resp_r,
            hb_rd_req_s, hb_rd_resp_r, hb_wr_req_s, hb_wr_resp_r,
        );

        (
            terminator,
            req_s, resp_r,
            input_mem_wr_req_s, input_mem_wr_data_s, input_mem_wr_resp_r,
            output_rd_req_s, output_rd_resp_r,
        )
    }

    init { }

    next(state: ()) {

        let tok = join();

        // arrange: create input data
        let tok = send(tok, input_mem_wr_req_s, MemWriterReq {
            addr: RamAddr:0,
            length: array_size(TEST_DATA)
        });

        for ((i, test_data), tok) in enumerate(TEST_DATA) {
            let ram_wr_req = MemWriterDataPacket {
                data: test_data as uN[TEST_DATA_W],
                length: RamAddr:1,
                last: i == array_size(TEST_DATA) - u32:1
            };
            let tok = send(tok, input_mem_wr_data_s, ram_wr_req);
            trace_fmt!("[TEST] Sent #{} data to input RAM {:#x}", i + u32:1, ram_wr_req);
            tok
        }(tok);

        let (tok, _) = recv(tok, input_mem_wr_resp_r);

        let req = Req {
            input_addr: uN[TEST_ADDR_W]:0x0,
            input_size: array_size(TEST_DATA) as u32,
            output_lit_addr: TEST_OUTPUT_LIT_ADDR,
            output_seq_addr: TEST_OUTPUT_SEQ_ADDR,
            zstd_params: ZstdParams<TEST_HT_SIZE_W> {
                num_entries_log2: NumEntriesLog2:9,
            },
        };

        // act: run match finder
        let tok = send(tok, req_s, req);
        trace_fmt!("[TEST] Sent request to the MatchFinder: {:#x}", req);
        let (tok, resp) = recv(tok, resp_r);

        // assert: literals
        assert_eq(resp.lit_cnt, u32:10);
        for ((i, expected), tok) in enumerate([
            // <---         l3l2l1
            u64:0x040102010C0B0A01,
            u64:0x0905

        ]) {
            let tok = send(tok, output_rd_req_s, OutputBufferRamRdReq {
                addr: TEST_OUTPUT_LIT_ADDR / RamAddr:8 + i,
                mask: uN[8]: 0xFF
            });

            let (tok, resp) = recv(tok, output_rd_resp_r);
            if resp.data != expected {
                trace_fmt!("[TEST] {:#x} != {:#x} at {:#x}", resp.data, expected, TEST_OUTPUT_LIT_ADDR + i * KEY_WIDTH / u32:8);
            } else {};
            assert_eq(resp.data, expected);
            tok
        }(join());

        // assert: sequences
        assert_eq(resp.seq_cnt, u32:2);
        for ((i, expected), tok) in enumerate([
            // legend:
            // LL : literals_len,
            // MO : match offset,
            // ML : match len

            //    seq2     seq1
            //    MLML LLLL MOMO MLML
            u64:0x0000_0008_000A_0000,
            //                seq2
            //              LLLL MOMO
            u64:0x0000_0000_0001_0007,
        ]) {
            let tok = send(tok, output_rd_req_s, OutputBufferRamRdReq {
                addr: TEST_OUTPUT_SEQ_ADDR / RamAddr:8 + i,
                mask: uN[8]: 0xFF
            });

            let (tok, resp) = recv(tok, output_rd_resp_r);
            if resp.data != expected {
                trace_fmt!("[TEST] {:#x} != {:#x} at {:#x}", resp.data, expected, TEST_OUTPUT_LIT_ADDR + i * KEY_WIDTH / u32:8);
            } else {};
            assert_eq(resp.data, expected);
            tok
        }(join());
        send(tok, terminator, true);
    }
}
