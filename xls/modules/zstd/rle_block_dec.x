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

import std;

import xls.modules.zstd.common;


type BlockDataPacket = common::BlockDataPacket;
type BlockPacketLength = common::BlockPacketLength;
type BlockData = common::BlockData;
type BlockSize = common::BlockSize;

type ExtendedBlockDataPacket = common::ExtendedBlockDataPacket;
type CopyOrMatchContent = common::CopyOrMatchContent;
type CopyOrMatchLength = common::CopyOrMatchLength;
type SequenceExecutorMessageType = common::SequenceExecutorMessageType;


pub enum RleBlockDecoderStatus: u1 {
    OKAY = 0,
}

pub struct RleBlockDecoderReq {
    id: u32,
    symbol: u8,
    length: BlockSize,
    last_block: bool,
}

pub struct RleBlockDecoderResp {
    status: RleBlockDecoderStatus
}

struct RleBlockDecoderState {
    req: RleBlockDecoderReq,
    req_valid: bool,
}

pub proc RleBlockDecoder<DATA_W: u32> {
    type Req = RleBlockDecoderReq;
    type Resp = RleBlockDecoderResp;
    type Output = ExtendedBlockDataPacket;

    type State = RleBlockDecoderState;

    req_r: chan<Req> in;
    resp_s: chan<Resp> out;
    output_s: chan<Output> out;

    config( req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,
    ) { (req_r, resp_s, output_s) }

    init { zero!<State>() }

    next(state: State) {
        const MAX_OUTPUT_SYMBOLS = (DATA_W / u32:8);
        const MAX_LEN = MAX_OUTPUT_SYMBOLS as uN[common::BLOCK_SIZE_WIDTH];

        let tok0 = join();

        let (tok1, req) = recv_if(tok0, req_r, !state.req_valid, state.req);

        let last = req.length <= MAX_LEN;
        let length = if last { req.length } else { MAX_LEN };
        let data = unroll_for! (i, data): (u32, uN[DATA_W]) in u32:0..MAX_OUTPUT_SYMBOLS {
            bit_slice_update(data, i * u32:8, req.symbol)
        }(uN[DATA_W]:0);

        let output = Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last,
                last_block: req.last_block,
                id: req.id,
                data: checked_cast<BlockData>(data),
                length: checked_cast<BlockPacketLength>(length),
            }
        };

        send_if(tok1, resp_s, last, zero!<Resp>());
        send(tok1, output_s, output);

        if last {
            zero!<State>()
        } else {
            let length = req.length - MAX_LEN;
            State {
                req: Req { length, ..req },
                req_valid: true,
            }
        }
    }
}


const TEST_DATA_W = u32:64;

#[test_proc]
proc RleBlockDecoderTest {
    type Req = RleBlockDecoderReq;
    type Resp = RleBlockDecoderResp;
    type Output = ExtendedBlockDataPacket;

    type Data = uN[TEST_DATA_W];
    type Length = uN[common::BLOCK_SIZE_WIDTH];

    terminator: chan<bool> out;

    req_s: chan<Req> out;
    resp_r: chan<Resp> in;
    output_r: chan<Output> in;

    config (terminator: chan<bool> out) {
        let (req_s, req_r) = chan<Req>("req");
        let (resp_s, resp_r) = chan<Resp>("resp");
        let (output_s, output_r) = chan<Output>("output");

        spawn RleBlockDecoder<TEST_DATA_W>(
            req_r, resp_s, output_s
        );

        (terminator, req_s, resp_r, output_r)
    }

    init { }

    next (state: ()) {
        let tok = join();

        let tok = send(tok, req_s, Req { id: u32:5, symbol: u8:0xAB, length: Length:0x28, last_block: true });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp { status: RleBlockDecoderStatus::OKAY });

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: false,
                last_block: true,
                id: u32:5,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:8
            }
        });

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: false,
                last_block: true,
                id: u32:5,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:8
            }
        });


        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: false,
                last_block: true,
                id: u32:5,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:8
            }
        });


        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: false,
                last_block: true,
                id: u32:5,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:8
            }
        });


        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: true,
                last_block: true,
                id: u32:5,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:8
            }
        });

        let tok = send(tok, req_s, Req { id: u32:1, symbol: u8:0xAB, length: Length:0, last_block: true });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp { status: RleBlockDecoderStatus::OKAY });

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: true,
                last_block: true,
                id: u32:1,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:0
            }
        });

        let tok = send(tok, req_s, Req { id: u32:10, symbol: u8:0xAB, length: Length:0, last_block: false });
        let (tok, resp) = recv(tok, resp_r);
        assert_eq(resp, Resp { status: RleBlockDecoderStatus::OKAY });

        let (tok, output) = recv(tok, output_r);
        assert_eq(output, Output {
            msg_type: SequenceExecutorMessageType::LITERAL,
            packet: BlockDataPacket {
                last: true,
                last_block: false,
                id: u32:10,
                data: BlockData:0xABAB_ABAB_ABAB_ABAB,
                length: BlockPacketLength:0
            }
        });

        send(tok, terminator, true);
    }
}


const INST_DATA_W = u32:64;

proc RleBlockDecoderInst {
    type Req = RleBlockDecoderReq;
    type Resp = RleBlockDecoderResp;
    type Output = ExtendedBlockDataPacket;

    type Data = uN[INST_DATA_W];
    type Length = uN[common::BLOCK_SIZE_WIDTH];

    config(
        req_r: chan<Req> in,
        resp_s: chan<Resp> out,
        output_s: chan<Output> out,
    ) {
        spawn RleBlockDecoder<INST_DATA_W>(req_r, resp_s, output_s);
    }


    init { }

    next (state: ()) {}
}
