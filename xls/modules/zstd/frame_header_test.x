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

import std;
import xls.modules.zstd.buffer as buff;
import xls.modules.zstd.frame_header as frame_header;

type Buffer = buff::Buffer;
type FrameHeaderResult = frame_header::FrameHeaderResult;
type WindowSize = frame_header::WindowSize;

// Largest allowed WindowLog accepted by libzstd decompression function
// https://github.com/facebook/zstd/blob/v1.4.7/lib/decompress/zstd_decompress.c#L296
// Use only in C++ tests when comparing DSLX ZSTD Decoder with libzstd
pub const TEST_WINDOW_LOG_MAX_LIBZSTD = WindowSize:30;

pub fn parse_frame_header_128(buffer: Buffer<128>) -> FrameHeaderResult<128> {
    frame_header::parse_frame_header<TEST_WINDOW_LOG_MAX_LIBZSTD>(buffer)
}
