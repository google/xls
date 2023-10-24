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

// This file contains work-in-progress ZSTD decoder implementation
// More information about ZSTD decoding can be found in:
// https://datatracker.ietf.org/doc/html/rfc8878
//
import std
import xls.modules.zstd.buffer as buff
import xls.modules.zstd.zstd_magic as magic
import xls.modules.zstd.zstd_frame_header as frame_header

type Buffer = buff::Buffer;

const DATA_WIDTH = u32:64;
const BUFFER_WIDTH = u32:128;
const ZERO_FRAME_HEADER = frame_header::ZERO_FRAME_HEADER;

enum ZstdDecoderStatus : u8 {
  IDLE = 0,
  DECODE_FRAME_HEADER = 1,
  DECODE_BLOCKS = 2,
}

struct ZstdDecoderState {
    status: ZstdDecoderStatus,
    buffer: Buffer<BUFFER_WIDTH>,
    frame_header: frame_header::FrameHeader,
}

pub proc ZstdDecoder {
    input_r: chan<bits[DATA_WIDTH]> in;
    output_s: chan<bits[DATA_WIDTH]> out;

    init {(
        ZstdDecoderState{
            status: ZstdDecoderStatus::IDLE,
            buffer: Buffer { contents: bits[BUFFER_WIDTH]:0, length: u32:0},
            frame_header: ZERO_FRAME_HEADER,
        }
    )}

    config (
        input_r: chan<bits[DATA_WIDTH]> in,
        output_s: chan<bits[DATA_WIDTH]> out,
    ) {(input_r, output_s)}

    next (tok: token, state: ZstdDecoderState) {
        let (tok, data) = recv(tok, input_r);
        trace_fmt!("received: 0x{:x}", data);

        // for now just return the received value
        let tok = send(tok, output_s, data);
        trace_fmt!("send: 0x{:x}", data);

        let buffer = buff::buffer_append(state.buffer, data);

        let state = match state.status {
            ZstdDecoderStatus::IDLE => {
                trace_fmt!("Decoding magic number!");
                let magic_result = magic::parse_magic_number(buffer);
                match magic_result.status {
                    magic::MagicStatus::OK => ZstdDecoderState {
                        status: ZstdDecoderStatus::DECODE_FRAME_HEADER,
                        buffer: magic_result.buffer,
                        frame_header: ZERO_FRAME_HEADER,
                    },
                    _ => state
                }
            },
            _ => state
        };

        let state = match state.status {
            ZstdDecoderStatus::DECODE_FRAME_HEADER => {
                trace_fmt!("Decoding frame header!");
                let frame_header_result = frame_header::parse_frame_header(state.buffer);
                match frame_header_result.status {
                    frame_header::FrameHeaderStatus::OK => ZstdDecoderState {
                        status: ZstdDecoderStatus::DECODE_BLOCKS,
                        buffer: frame_header_result.buffer,
                        frame_header: frame_header_result.header,
                    },
                    _ => state,
                }
            },
            _ => state
        };

        state
    }
}
