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

pub enum Mark : u4 {
    // Default initialization value used when marker field is not needed
    NONE = 0,
    // Signals end of sequence/block
    END = 1,
    // Requests reset of the processing chain
    RESET = 2,

    _ERROR_FIRST = 8,
    // Only error marks have values >= __ERROR_FIRST
    ERROR_BAD_MARK = 8,
    ERROR_INVAL_CP = 9,
}

pub fn is_error(mark: Mark) -> bool {
    (mark as u32) >= (Mark::_ERROR_FIRST as u32)
}

pub enum TokenKind : u2 {
    LT = 0, // Literal
    CP = 1, // Copy pointer
    MK = 2, // Control marker
}

pub struct Token<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    kind: TokenKind,
    // Literal fields
    lt_sym: uN[SYM_WIDTH],
    // Copy pointer fields
    cp_off: uN[PTR_WIDTH],
    cp_cnt: uN[CNT_WIDTH],
    // Marker fields
    mark: Mark
}

pub fn zero_token<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32>()
        -> Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH> {
    Token<SYM_WIDTH, PTR_WIDTH, CNT_WIDTH>{
        kind: TokenKind::MK,
        lt_sym: uN[SYM_WIDTH]:0,
        cp_off: uN[PTR_WIDTH]:0,
        cp_cnt: uN[CNT_WIDTH]:0,
        mark: Mark::NONE,
    }
}

pub struct PlainData<DATA_WIDTH: u32> {
    is_mark: bool,
    data: uN[DATA_WIDTH],    // symbol
    mark: Mark,             // marker code
}

pub fn zero_plain_data<DATA_WIDTH: u32>() -> PlainData<DATA_WIDTH> {
    PlainData<DATA_WIDTH>{
        is_mark: true,
        data: uN[DATA_WIDTH]:0,
        mark: Mark::NONE,
    }
}
