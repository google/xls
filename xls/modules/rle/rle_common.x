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

// This file defines RLE common data structures
//

// Structure contains uncompressed symbols.
// Structure is used as an input and an output to and from
// a preprocessing stage as well as an input to a RLE encoder.
// It is also used as an output from RLE decoder.
pub struct PlainData<SYMB_WIDTH: u32, SYMB_COUNT: u32> {
    symbols: bits[SYMB_WIDTH][SYMB_COUNT], // symbols
    symbol_valids: bits[1][SYMB_COUNT],    // symbol valid
    last: bool,                            // flush RLE
}

// Structure contains compressed (symbol, counter) pairs.
// Structure is used as an output from RLE encoder and
// as an input to RLE decoder.
pub struct CompressedData<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32, PAIR_COUNT: u32> {
    symbols: bits[SYMBOL_WIDTH][PAIR_COUNT], // symbol
    counts: bits[COUNT_WIDTH][PAIR_COUNT],   // symbol counter
    last: bool,                              // flush RLE
}
