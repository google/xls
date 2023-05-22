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

// This file implements a parametric RLE encoder
//
// The encoder uses Run Length Encoding (RLE) to compress the input stream
// with symbols (data to compress). The output contains packets of data with
// the symbols and the number of their occurrences in the input stream.

// structure containing output data a Data preprocessor
// and input to an RLE encoder
pub struct PreData<SYMB_WIDTH: u32> {
    symbol: bits[SYMB_WIDTH], // symbol
    last: bool,               // flush RLE
}

// structure containing output data from an RLE encoder
// and input to a Stream Mixer
pub struct EncData<SYMBOL_WIDTH: u32, COUNT_WIDTH: u32> {
    symbol: bits[SYMBOL_WIDTH], // symbol
    count: bits[COUNT_WIDTH],   // symbol counter
    last: bool,                 // flush RLE
}
