// Copyright 2024 The XLS Authors
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

import xls.modules.zstd.memory.mem_reader;
import xls.modules.zstd.common;

type CompressionMode = common::CompressionMode;
type SequenceConf = common::SequenceConf;

enum SequenceHeaderSize : u2 {
    TWO_BYTES   = 0,
    THREE_BYTES = 1,
    FOUR_BYTES  = 2,
}

fn parse_sequence_first_byte(byte :u8) -> SequenceHeaderSize {
    if byte == u8:0 {
        SequenceHeaderSize::TWO_BYTES
    } else if byte == u8:255 {
        SequenceHeaderSize::FOUR_BYTES
    } else if byte[7:] == u1:1 {
        SequenceHeaderSize::THREE_BYTES
    } else {
        SequenceHeaderSize::TWO_BYTES
    }
}

fn extract_seq_mode(byte: u8) -> (CompressionMode, CompressionMode, CompressionMode) {
    (byte[2:4] as CompressionMode,
     byte[4:6] as CompressionMode,
     byte[6:] as CompressionMode)
}

pub fn parse_sequence_conf(header: u32) -> (SequenceConf, u3) {
    let header_size = parse_sequence_first_byte(header[:8]);
    let sum1 = (header[0:7] ++ header[8:16]) as u17;
    let sum2 = sum1 + (header[16:24] ++ u8:0) as u17 ;
    match(header_size) {
        SequenceHeaderSize::TWO_BYTES => {
            let (match_mode, offset_mode, literals_mode) = extract_seq_mode(header[8:16]);
            (SequenceConf{
                sequence_count: header[:8] as u17,
                match_mode: match_mode,
                offset_mode: offset_mode,
                literals_mode: literals_mode,
            }, u3:2)
        },
        SequenceHeaderSize::THREE_BYTES => {
            let (match_mode, offset_mode, literals_mode) = extract_seq_mode(header[16:24]);
            (SequenceConf{
                sequence_count: sum1,
                match_mode: match_mode,
                offset_mode: offset_mode,
                literals_mode: literals_mode,
            }, u3:3)
        },
        SequenceHeaderSize::FOUR_BYTES => {
            let (match_mode, offset_mode, literals_mode) = extract_seq_mode(header[24:32]);
            (SequenceConf{
                sequence_count: sum2,
                match_mode: match_mode,
                offset_mode: offset_mode,
                literals_mode: literals_mode,
            }, u3:4)
        },
        _ => (zero!<SequenceConf>(), u3:0)
        // fail!() doesn't work with quicktest, JIT failes to translate such function
        //  _ => fail!("Incorrect_header_size", zero!<SequenceConf>())
    }
}

#[quickcheck(test_count=50000)]
fn test_parse_sequence_conf(x: u32) -> bool {
    // let length = parse_sequence_first_byte(x[0:8]);
    let (seq_conf, length) = parse_sequence_conf(x);
    let byte0 = x[0:8];
    let byte1 = x[8:16];
    let byte2 = x[16:24];

    if x[0:8] < u8:128 {
        length == u3:2 && seq_conf.sequence_count == byte0 as u17
    } else if x[0:8] < u8:255 {
        length == u3:3 && seq_conf.sequence_count == (((byte0 - u8:128) as u17) << u8:8) as u17 + byte1 as u17
    } else {
        length == u3:4 && seq_conf.sequence_count == u17:0x7f00 + byte1 as u17 + ((byte2 as u17) << u8:8) as u17
    }
}


pub enum SequenceConfDecoderStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct SequenceConfDecoderReq <ADDR_W: u32> {
    addr: uN[ADDR_W],
}

pub struct SequenceConfDecoderResp {
    header: SequenceConf,
    length: u3,
    status: SequenceConfDecoderStatus,
}

pub proc SequenceConfDecoder<AXI_DATA_W: u32, AXI_ADDR_W: u32> {

    type Req = SequenceConfDecoderReq<AXI_ADDR_W>;
    type Resp = SequenceConfDecoderResp;
    type Status = SequenceConfDecoderStatus;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    init {}

    config(
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,

        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
    ) {
        (mem_rd_req_s, mem_rd_resp_r, req_r, resp_s)
    }

    next(state: ()) {
        let tok = join();

        let (tok, decode_request) = recv(tok, req_r);
        trace_fmt!("[SequenceConfDecoder] received {}", decode_request);

        let tok = send(tok, mem_rd_req_s, MemReaderReq {
            addr: decode_request.addr,
            // max number of bytes that the header can have, see RFC8878 Section 3.1.1.3.2.1.
            length: uN[AXI_ADDR_W]:4,
        });
        // TODO: handle multiple receives on mem_rd_resp_r when AXI_DATA_W < 32
        const_assert!(AXI_DATA_W >= u32:32);
        let (tok, raw) = recv(tok, mem_rd_resp_r);
        let (header, length) = parse_sequence_conf(raw.data[:32]);
        let tok = send(tok, resp_s, Resp {
            header: header,
            length: length,
            status: match (raw.status) {
                MemReaderStatus::OKAY => Status::OKAY,
                MemReaderStatus::ERROR => Status::ERROR,
                _ => fail!("literals_header_decoder_status_unreachable", Status::OKAY),
            }
        });
    }
}

const TEST_AXI_DATA_W = u32:64;
const TEST_AXI_ADDR_W = u32:32;

#[test_proc]
proc SequenceConfDecoderTest {
    type Req = SequenceConfDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = SequenceConfDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    terminator: chan<bool> out;

    mem_rd_req_r: chan<MemReaderReq> in;
    mem_rd_resp_s: chan<MemReaderResp> out;
    req_s: chan<Req> out;
    resp_r: chan<Resp> in;

    init {}

    config(terminator: chan<bool> out) {

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn SequenceConfDecoder<TEST_AXI_DATA_W, TEST_AXI_ADDR_W> (
            mem_rd_req_s, mem_rd_resp_r, req_r, resp_s
        );

        (
            terminator,
            mem_rd_req_r, mem_rd_resp_s,
            req_s, resp_r
        )
    }

    next(state: ()) {
        let tok = join();

        // test data format: raw header, expected size in bytes, expected parsed header
        let tests: (u32, u3, SequenceConf)[8] = [
            (u32:0x00_00, u3:2, SequenceConf {
                sequence_count: u17:0,
                literals_mode: CompressionMode::PREDEFINED,
                offset_mode: CompressionMode::PREDEFINED,
                match_mode: CompressionMode::PREDEFINED,
            }),
            (u32:0x6C_00, u3:2, SequenceConf {
                sequence_count: u17:0,
                literals_mode: CompressionMode::RLE,
                offset_mode: CompressionMode::COMPRESSED,
                match_mode: CompressionMode::REPEAT,
            }),
            (u32:0xE4_01, u3:2, SequenceConf {
                sequence_count: u17:0x01,
                literals_mode: CompressionMode::REPEAT,
                offset_mode: CompressionMode::COMPRESSED,
                match_mode: CompressionMode::RLE,
            }),
            (u32:0xAC_7F, u3:2, SequenceConf {
                sequence_count: u17:0x7F,
                literals_mode: CompressionMode::COMPRESSED,
                offset_mode: CompressionMode::COMPRESSED,
                match_mode: CompressionMode::REPEAT,
            }),
            (u32:0x84_0080, u3:3, SequenceConf {
                sequence_count: u17:0,
                literals_mode: CompressionMode::COMPRESSED,
                offset_mode: CompressionMode::PREDEFINED,
                match_mode: CompressionMode::RLE,
            }),
            (u32:0x18_FFFE, u3:3, SequenceConf {
                sequence_count: u17:0x7EFF,
                literals_mode: CompressionMode::PREDEFINED,
                offset_mode: CompressionMode::RLE,
                match_mode: CompressionMode::COMPRESSED,
            }),
            (u32:0x70_0000FF, u3:4, SequenceConf {
                sequence_count: u17:0x7F00,
                literals_mode: CompressionMode::RLE,
                offset_mode: CompressionMode::REPEAT,
                match_mode: CompressionMode::PREDEFINED,
            }),
            (u32:0x68_FFFFFF, u3:4, SequenceConf {
                sequence_count: u17:0x17EFF,
                literals_mode: CompressionMode::RLE,
                offset_mode: CompressionMode::COMPRESSED,
                match_mode: CompressionMode::COMPRESSED,
            }),
        ];
        const ADDR = uN[TEST_AXI_ADDR_W]:0xDEAD;

        // positive cases
        let tok = for ((_, (test_vec, expected_length, expected_header)), tok): ((u32, (u32, u3, SequenceConf)), token) in enumerate(tests) {
            send(tok, req_s, Req {
                addr: ADDR,
            });
            let (tok, req) = recv(tok, mem_rd_req_r);
            assert_eq(req, MemReaderReq {
                addr: ADDR,
                length: uN[TEST_AXI_ADDR_W]:4
            });
            let tok = send(tok, mem_rd_resp_s, MemReaderResp {
                status: MemReaderStatus::OKAY,
                data: test_vec as uN[TEST_AXI_DATA_W],
                length: uN[TEST_AXI_ADDR_W]:4,
                last: true,
            });
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, SequenceConfDecoderResp {
                header: expected_header,
                status: SequenceConfDecoderStatus::OKAY,
                length: expected_length,
            });
            tok
        }(tok);

        // negative case: AXI Error
        send(tok, req_s, Req {
            addr: ADDR,
        });
        let (tok, req) = recv(tok, mem_rd_req_r);
        assert_eq(req, MemReaderReq {
            addr: ADDR,
            length: uN[TEST_AXI_ADDR_W]:4
        });
        let tok = send(tok, mem_rd_resp_s, MemReaderResp {
            status: MemReaderStatus::ERROR,
            data: uN[TEST_AXI_DATA_W]:0,
            length: uN[TEST_AXI_ADDR_W]:0,
            last: true,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp.status, SequenceConfDecoderStatus::ERROR);

        send(join(), terminator, true);
    }
}

proc SequenceConfDecoderInst {
    type Req = SequenceConfDecoderReq<u32:16>;
    type Resp = SequenceConfDecoderResp;
    type ReaderReq = mem_reader::MemReaderReq<u32:16>;
    type ReaderResp = mem_reader::MemReaderResp<u32:64, u32:16>;

    reader_req_s: chan<ReaderReq> out;
    reader_resp_r: chan<ReaderResp> in;

    decode_req_r: chan<Req> in;
    decode_resp_s: chan<Resp> out;

    config(
        reader_req_s: chan<ReaderReq> out,
        reader_resp_r: chan<ReaderResp> in,
        decode_req_r: chan<Req> in,
        decode_resp_s: chan<Resp> out,
    ) {
        spawn SequenceConfDecoder<u32:64, u32:16>(
            reader_req_s,
            reader_resp_r,
            decode_req_r,
            decode_resp_s
        );
        (reader_req_s, reader_resp_r, decode_req_r, decode_resp_s)
    }

    init {}

    next(state: ()) {}
}
