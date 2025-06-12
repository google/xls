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

pub enum LiteralsHeaderSize : u3 {
    SINGLE_BYTE      = 0,
    TWO_BYTES        = 1,
    THREE_BYTES      = 2,
    COMP_THREE_BYTES = 4,
    COMP_FOUR_BYTES  = 5,
    COMP_FIVE_BYTES  = 6,
}

pub enum LiteralsBlockType: u3 {
    RAW        = 0,
    RLE        = 1,
    COMP       = 2,
    COMP_4     = 3,
    TREELESS   = 4,
    TREELESS_4 = 5,
}


pub fn parse_literals_header_first_byte(first_byte :u8) -> (LiteralsBlockType, LiteralsHeaderSize) {
    match (first_byte[:2], first_byte[2:4]) {
        (u2:0, u2:1) => (LiteralsBlockType::RAW, LiteralsHeaderSize::TWO_BYTES),
        (u2:0, u2:3) => (LiteralsBlockType::RAW, LiteralsHeaderSize::THREE_BYTES),
        (u2:0, _) => (LiteralsBlockType::RAW, LiteralsHeaderSize::SINGLE_BYTE),
        (u2:1, u2:1) => (LiteralsBlockType::RLE, LiteralsHeaderSize::TWO_BYTES),
        (u2:1, u2:3) => (LiteralsBlockType::RLE, LiteralsHeaderSize::THREE_BYTES),
        (u2:1, _) => (LiteralsBlockType::RLE, LiteralsHeaderSize::SINGLE_BYTE),
        (u2:2, u2:0) => (LiteralsBlockType::COMP, LiteralsHeaderSize::COMP_THREE_BYTES),
        (u2:2, u2:1) => (LiteralsBlockType::COMP_4, LiteralsHeaderSize::COMP_THREE_BYTES),
        (u2:2, u2:2) => (LiteralsBlockType::COMP_4, LiteralsHeaderSize::COMP_FOUR_BYTES),
        (u2:2, u2:3) => (LiteralsBlockType::COMP_4, LiteralsHeaderSize::COMP_FIVE_BYTES),
        (u2:3, u2:0) => (LiteralsBlockType::TREELESS, LiteralsHeaderSize::COMP_THREE_BYTES),
        (u2:3, u2:1) => (LiteralsBlockType::TREELESS_4, LiteralsHeaderSize::COMP_THREE_BYTES),
        (u2:3, u2:2) => (LiteralsBlockType::TREELESS_4, LiteralsHeaderSize::COMP_FOUR_BYTES),
        (u2:3, u2:3) => (LiteralsBlockType::TREELESS_4, LiteralsHeaderSize::COMP_FIVE_BYTES),
        _ => (LiteralsBlockType::TREELESS, LiteralsHeaderSize::COMP_THREE_BYTES),
        // fail!() doesn't work with quicktest, JIT failes to translate such function
        //_ => fail!("Should_never_be_called", (LiteralsBlockType::RAW, LiteralsHeaderSize::SINGLE_BYTE))
    }
}

#[quickcheck]
fn test_parse_literals_header_first_byte(x: u8) -> bool {
    let (literal, length) = parse_literals_header_first_byte(x);
    ((literal == LiteralsBlockType::RAW || literal == LiteralsBlockType::RLE) &&
    (length == LiteralsHeaderSize::SINGLE_BYTE || length == LiteralsHeaderSize::TWO_BYTES ||
     length == LiteralsHeaderSize::THREE_BYTES)) ||
    ((literal == LiteralsBlockType::COMP || literal == LiteralsBlockType::TREELESS) &&
    (length == LiteralsHeaderSize::COMP_THREE_BYTES)) ||
    ((literal == LiteralsBlockType::COMP_4 || literal == LiteralsBlockType::TREELESS_4) &&
    (length == LiteralsHeaderSize::COMP_THREE_BYTES || length == LiteralsHeaderSize::COMP_FOUR_BYTES ||
     length == LiteralsHeaderSize::COMP_FIVE_BYTES)
    )
}

pub struct LiteralsHeader {
    literal_type: LiteralsBlockType,
    regenerated_size: u20,
    compressed_size: u20,
}

pub fn parse_literals_header(header_raw: u40) -> (LiteralsHeader, u3, u8) {
    let (literal_type, header_size) = parse_literals_header_first_byte(header_raw[0:8]);
    let (regenerated_size, compressed_size, header_length, symbol) = match (header_size) {
        LiteralsHeaderSize::SINGLE_BYTE => (header_raw[3:8] as u20, header_raw[3:8] as u20, u3:1, header_raw[8:16]),
        LiteralsHeaderSize::TWO_BYTES => (header_raw[4:16] as u20, header_raw[4:16] as u20, u3:2, header_raw[16:24]),
        LiteralsHeaderSize::THREE_BYTES => (header_raw[4:24] as u20, header_raw[4:24] as u20, u3:3, header_raw[24:32]),
        LiteralsHeaderSize::COMP_THREE_BYTES => (header_raw[4:14] as u20, header_raw[14:24] as u20, u3:3, header_raw[24:32]),
        LiteralsHeaderSize::COMP_FOUR_BYTES => (header_raw[4:18] as u20, header_raw[18:32] as u20, u3:4, u8:0),
        LiteralsHeaderSize::COMP_FIVE_BYTES => (header_raw[4:22] as u20, header_raw[22:40] as u20, u3:5, u8:0),
        // fail!() doesn't work with quicktest, JIT failes to translate such function
        //_ => fail!("Unrecognized_header_sizeC" ,CompressedBlockSize {
        _ => (u20:0, u20:0, u3:0, u8:0),
    };
    (LiteralsHeader {
        literal_type: literal_type,
        regenerated_size: regenerated_size,
        compressed_size: match (literal_type) {
            LiteralsBlockType::RLE => u20:1,
            _ => compressed_size,
        }
    }, header_length, symbol)
}

#[quickcheck]
fn test_parse_literals_header(x: u40) -> bool {
    let (header, header_length_bytes, symbol) = parse_literals_header(x);
    let (_, header_size) = parse_literals_header_first_byte(x[0:8]);

    let length_bytes_equivalence = match (header_size) {
        LiteralsHeaderSize::SINGLE_BYTE => header_length_bytes == u3:1,
        LiteralsHeaderSize::TWO_BYTES => header_length_bytes == u3:2,
        LiteralsHeaderSize::THREE_BYTES | LiteralsHeaderSize::COMP_THREE_BYTES => header_length_bytes == u3:3,
        LiteralsHeaderSize::COMP_FOUR_BYTES => header_length_bytes == u3:4,
        LiteralsHeaderSize::COMP_FIVE_BYTES => header_length_bytes == u3:5,
        _ => false
    };
    let raw_length_equivalence = if (header.literal_type == LiteralsBlockType::RAW) {
        header.regenerated_size == header.compressed_size
    } else { true };
    let regen_comp_size_equivalence = if (header.literal_type == LiteralsBlockType::RAW || header.literal_type == LiteralsBlockType::RLE) {
        raw_length_equivalence && match(header_size) {
            LiteralsHeaderSize::SINGLE_BYTE => header.regenerated_size == x[3:8] as u20,
            LiteralsHeaderSize::TWO_BYTES => header.regenerated_size == x[4:16] as u20,
            LiteralsHeaderSize::THREE_BYTES => header.regenerated_size == x[4:24],
            _ => false
        }
    } else {
        match(header_size) {
            LiteralsHeaderSize::COMP_THREE_BYTES => {
                header.regenerated_size == x[4:14] as u20 &&
                header.compressed_size == x[14:24] as u20
            },
            LiteralsHeaderSize::COMP_FOUR_BYTES => {
                header.regenerated_size == x[4:18] as u20 &&
                header.compressed_size == x[18:32] as u20
            },
            LiteralsHeaderSize::COMP_FIVE_BYTES => {
                header.regenerated_size == x[4:22] as u20 &&
                header.compressed_size == x[22:40] as u20
            },
            _ => false
        }
    };

    let symbol_equivalence = match (header_size) {
        LiteralsHeaderSize::SINGLE_BYTE => symbol == x[8:16],
        LiteralsHeaderSize::TWO_BYTES => symbol == x[16:24],
        LiteralsHeaderSize::THREE_BYTES | LiteralsHeaderSize::COMP_THREE_BYTES => symbol == x[24:32],
        LiteralsHeaderSize::COMP_FOUR_BYTES => symbol == u8:0,
        LiteralsHeaderSize::COMP_FIVE_BYTES => symbol == u8:0,
        _ => false
    };

    length_bytes_equivalence && raw_length_equivalence && regen_comp_size_equivalence && symbol_equivalence
}

pub enum LiteralsHeaderDecoderStatus : u1 {
    OKAY = 0,
    ERROR = 1,
}

pub struct LiteralsHeaderDecoderReq <ADDR_W: u32> {
    addr: uN[ADDR_W],
}

pub struct LiteralsHeaderDecoderResp {
    header: LiteralsHeader,
    symbol: u8,
    length: u3,
    status: LiteralsHeaderDecoderStatus,
}

pub proc LiteralsHeaderDecoder<AXI_DATA_W: u32, AXI_ADDR_W: u32> {

    type Req = LiteralsHeaderDecoderReq<AXI_ADDR_W>;
    type Resp = LiteralsHeaderDecoderResp;
    type Status = LiteralsHeaderDecoderStatus;

    type MemReaderReq  = mem_reader::MemReaderReq<AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<AXI_DATA_W, AXI_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;

    mem_rd_req_s: chan<MemReaderReq> out;
    mem_rd_resp_r: chan<MemReaderResp> in;

    init {}

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        mem_rd_req_s: chan<MemReaderReq> out,
        mem_rd_resp_r: chan<MemReaderResp> in,
    ) {
        (req_r, resp_s, mem_rd_req_s, mem_rd_resp_r)
    }

    next(state: ()) {
        let tok = join();

        let (tok, decode_request) = recv(tok, req_r);
        send(tok, mem_rd_req_s, MemReaderReq {
            addr: decode_request.addr,
            // max number of bytes that the header can have, see RFC8878 Section 3.1.1.3.1.1.
            length: uN[AXI_ADDR_W]:5,
        });
        // TODO: handle multiple receives on mem_rd_resp_r when AXI_DATA_W < 40
        const_assert!(AXI_DATA_W >= u32:64);
        let (tok, raw) = recv(tok, mem_rd_resp_r);
        let (header, length, symbol) = parse_literals_header(raw.data[:40]);
        send(tok, resp_s, Resp {
            header: header,
            symbol: symbol,
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
proc LiteralsHeaderDecoderTest {
    type Req = LiteralsHeaderDecoderReq<TEST_AXI_ADDR_W>;
    type Resp = LiteralsHeaderDecoderResp;

    type MemReaderReq  = mem_reader::MemReaderReq<TEST_AXI_ADDR_W>;
    type MemReaderResp = mem_reader::MemReaderResp<TEST_AXI_DATA_W, TEST_AXI_ADDR_W>;
    type MemReaderStatus = mem_reader::MemReaderStatus;

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    mem_rd_req_r: chan<MemReaderReq> in;
    mem_rd_resp_s: chan<MemReaderResp> out;

    init {}

    config(terminator: chan<bool> out) {

        let (mem_rd_req_s, mem_rd_req_r) = chan<MemReaderReq>("mem_rd_req");
        let (mem_rd_resp_s, mem_rd_resp_r) = chan<MemReaderResp>("mem_rd_resp");

        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");

        spawn LiteralsHeaderDecoder<TEST_AXI_DATA_W, TEST_AXI_ADDR_W> (
            req_r, resp_s, mem_rd_req_s, mem_rd_resp_r
        );

        (
            terminator,
            req_s, resp_r,
            mem_rd_req_r, mem_rd_resp_s,
        )
    }

    next(state: ()) {
        let tok = join();

        // test data format: raw header, expected size in bytes, expected parsed header
        let tests: (u40, u3, LiteralsHeader, u8)[16] = [
            // 2 bits block type == RAW, 1 bit size_format == 0, 5 bits regenerated_size, symbol: 0xAA
            (u40:0b10101010_10100_0_00, u3:1, LiteralsHeader {
                literal_type: LiteralsBlockType::RAW,
                regenerated_size: u20:0b10100,
                compressed_size: u20:0b10100,
            }, u8:0xAA),
            // 2 bits block type == RAW, 2 bit size_format == 1, 12 bits regenerated_size, symbol: 0xF5
            (u40:0b11110101_101010101010_01_00, u3:2, LiteralsHeader {
                literal_type: LiteralsBlockType::RAW,
                regenerated_size: u20:0b101010101010,
                compressed_size: u20:0b101010101010,
            }, u8:0xF5),
            // 2 bits block type == RAW, 1 bit size_format == 2, 5 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_10101_0_00, u3:1, LiteralsHeader {
                literal_type: LiteralsBlockType::RAW,
                regenerated_size: u20:0b10101,
                compressed_size: u20:0b10101,
            }, u8:0xF0),
            // 2 bits block type == RAW, 2 bit size_format == 3, 20 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_10101010101010101010_11_00, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::RAW,
                regenerated_size: u20:0b10101010101010101010,
                compressed_size: u20:0b10101010101010101010,
            }, u8:0xF0),

            // 2 bits block type == RLE, 1 bit size_format == 0, 5 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_10100_0_01, u3:1, LiteralsHeader {
                literal_type: LiteralsBlockType::RLE,
                regenerated_size: u20:0b10100,
                compressed_size: u20:1,
            }, u8:0xF0),
            // 2 bits block type == RLE, 2 bits size_format == 1, 12 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_101010101010_01_01, u3:2, LiteralsHeader {
                literal_type: LiteralsBlockType::RLE,
                regenerated_size: u20:0b101010101010,
                compressed_size: u20:1,
            }, u8:0xF0),
            // 2 bits block type == RLE, 1 bit size_format == 2, 5 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_10101_0_01, u3:1, LiteralsHeader {
                literal_type: LiteralsBlockType::RLE,
                regenerated_size: u20:0b10101,
                compressed_size: u20:1,
            }, u8:0xF0),
            // 2 bits block type == RLE, 2 bits size_format == 3, 20 bits regenerated_size, symbol: 0xF0
            (u40:0b11110000_10101010101010101010_11_01, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::RLE,
                regenerated_size: u20:0b10101010101010101010,
                compressed_size: u20:1,
            }, u8:0xF0),

            // 2 bits block type == COMPRESSED, 2 bits size_format == 0, 10 bits regenerated_size and compressed_size, symbol: 0xF0
            (u40:0b11110000_1010101010_0101010101_00_10, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::COMP,
                regenerated_size: u20:0b0101010101,
                compressed_size: u20:0b1010101010,
            }, u8:0xF0),
            // 2 bits block type == COMPRESSED, 2 bits size_format == 1, 10 bits regenerated_size and compressed_size, symbol: 0xF0
            (u40:0b11110000_1010101010_0101010101_01_10, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::COMP_4,
                regenerated_size: u20:0b0101010101,
                compressed_size: u20:0b1010101010,
            }, u8:0xF0),
            // 2 bits block type == COMPRESSED, 2 bits size_format == 2, 14 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b10101010101010_01010101010101_10_10, u3:4, LiteralsHeader {
                literal_type: LiteralsBlockType::COMP_4,
                regenerated_size: u20:0b01010101010101,
                compressed_size: u20:0b10101010101010,
            }, u8:0x0),
            // 2 bits block type == COMPRESSED, 2 bits size_format == 3, 18 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b101010101010101010_010101010101010101_11_10, u3:5, LiteralsHeader {
                literal_type: LiteralsBlockType::COMP_4,
                regenerated_size: u20:0b010101010101010101,
                compressed_size: u20:0b101010101010101010,
            }, u8:0x0),

            // 2 bits block type == TREELESS, 2 bits size_format == 0, 10 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b1010101010_0101010101_00_11, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::TREELESS,
                regenerated_size: u20:0b0101010101,
                compressed_size: u20:0b1010101010,
            }, u8:0x0),
            // 2 bits block type == TREELESS, 2 bits size_format == 1, 10 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b1010101010_0101010101_01_11, u3:3, LiteralsHeader {
                literal_type: LiteralsBlockType::TREELESS_4,
                regenerated_size: u20:0b0101010101,
                compressed_size: u20:0b1010101010,
            }, u8:0x0),
            // 2 bits block type == TREELESS, 2 bits size_format == 2, 14 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b10101010101010_01010101010101_10_11, u3:4, LiteralsHeader {
                literal_type: LiteralsBlockType::TREELESS_4,
                regenerated_size: u20:0b01010101010101,
                compressed_size: u20:0b10101010101010,
            }, u8:0x0),
            // 2 bits block type == TREELESS, 2 bits size_format == 3, 18 bits regenerated_size and compressed_size, symbol: 0x0
            (u40:0b101010101010101010_010101010101010101_11_11, u3:5, LiteralsHeader {
                literal_type: LiteralsBlockType::TREELESS_4,
                regenerated_size: u20:0b010101010101010101,
                compressed_size: u20:0b101010101010101010,
            }, u8:0x0),
        ];
        const ADDR = uN[TEST_AXI_ADDR_W]:0xDEAD;

        // positive cases
        let tok = for ((_, (test_vec, expected_length, expected_header, expected_symbol)), tok): ((u32, (u40, u3, LiteralsHeader, u8)), token) in enumerate(tests) {
            send(tok, req_s, Req {
                addr: ADDR,
            });
            let (tok, req) = recv(tok, mem_rd_req_r);
            assert_eq(req, MemReaderReq {
                addr: ADDR,
                length: uN[TEST_AXI_ADDR_W]:5
            });
            let tok = send(tok, mem_rd_resp_s, MemReaderResp {
                status: MemReaderStatus::OKAY,
                data: test_vec as uN[TEST_AXI_DATA_W],
                length: uN[TEST_AXI_ADDR_W]:5,
                last: true,
            });
            let (tok, resp) = recv(tok, resp_r);
            assert_eq(resp, LiteralsHeaderDecoderResp {
                header: expected_header,
                symbol: expected_symbol,
                status: LiteralsHeaderDecoderStatus::OKAY,
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
            length: uN[TEST_AXI_ADDR_W]:5
        });
        let tok = send(tok, mem_rd_resp_s, MemReaderResp {
            status: MemReaderStatus::ERROR,
            data: uN[TEST_AXI_DATA_W]:0,
            length: uN[TEST_AXI_ADDR_W]:0,
            last: true,
        });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp.status, LiteralsHeaderDecoderStatus::ERROR);

        send(join(), terminator, true);
    }
}

proc LiteralsHeaderDecoderInst {
    type Req = LiteralsHeaderDecoderReq<u32:16>;
    type Resp = LiteralsHeaderDecoderResp;
    type ReaderReq = mem_reader::MemReaderReq<u32:16>;
    type ReaderResp = mem_reader::MemReaderResp<u32:64, u32:16>;

    decode_req_r: chan<Req> in;
    decode_resp_s: chan<Resp> out;

    reader_req_s: chan<ReaderReq> out;
    reader_resp_r: chan<ReaderResp> in;

    config(
        decode_req_r: chan<Req> in,
        decode_resp_s: chan<Resp> out,
        reader_req_s: chan<ReaderReq> out,
        reader_resp_r: chan<ReaderResp> in,
    ) {
        spawn LiteralsHeaderDecoder<u32:64, u32:16>(
            decode_req_r,
            decode_resp_s,
            reader_req_s,
            reader_resp_r
        );
        (decode_req_r, decode_resp_s, reader_req_s, reader_resp_r)
    }

    init {}

    next(state: ()) {}
}
