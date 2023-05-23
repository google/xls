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

pub enum TokenKind : u1 {
    LT = 0, // Literal
    CP = 1  // Copy pointer
}

pub struct Token<SYM_WIDTH: u32, PTR_WIDTH: u32, CNT_WIDTH: u32> {
    kind: TokenKind,
    last: bool,
    // Literal fields
    lt_sym: uN[SYM_WIDTH],
    // Copy pointer fields
    cp_off: uN[PTR_WIDTH],
    cp_cnt: uN[CNT_WIDTH]
}

pub struct PlainData<SYM_WIDTH: u32> {
    sym: uN[SYM_WIDTH], // symbol
    last: bool,         // last symbol in a block
}
