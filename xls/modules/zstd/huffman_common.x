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

// This file contains the implementation of Huffmann tree decoder.

import std;

pub const MAX_WEIGHT = u32:11;
pub const WEIGHT_LOG = std::clog2(MAX_WEIGHT + u32:1);
pub const MAX_SYMBOL_COUNT = u32:256;
pub const MAX_CODE_LEN = u32:12;

pub const PARALLEL_ACCESS_WIDTH = u32:8;
pub const COUNTER_WIDTH = std::clog2(PARALLEL_ACCESS_WIDTH + u32:1);

pub struct WeightPreScanMetaData {
    occurance_number: uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH],
    valid_weights:    u1[MAX_WEIGHT + u32:1],
    weights_count:    uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1],
}

// TODO: Enable once parametrics work
//pub struct WeightPreScanMetaData <
//    PARALLEL_ACCESS_WIDTH: u32,
//    COUNTER_WIDTH: u32 = {std::clog2(PARALLEL_ACCESS_WIDTH + u32:1)}
//> {
//    occurance_number: uN[COUNTER_WIDTH][PARALLEL_ACCESS_WIDTH],
//    valid_weights:   u1[MAX_WEIGHT + u32:1],
//    weights_count:   uN[COUNTER_WIDTH][MAX_WEIGHT + u32:1],
//}

pub struct WeightPreScanOutput {
    weights: uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH],
    meta_data: WeightPreScanMetaData,
}
// TODO: Use parametrics when they work
//pub struct WeightPreScanOutput<
//    PARALLEL_ACCESS_WIDTH: u32, WEIGHT_LOG: u32
//> {
//    weights: uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH],
//    meta_data: WeightPreScanMetaData,
//}

pub struct CodeBuilderToPreDecoderOutput {
    max_code_length: uN[WEIGHT_LOG],
    valid_weights:   u1[MAX_WEIGHT + u32:1],
}

pub struct CodeBuilderToDecoderOutput {
    symbol_valid: u1[PARALLEL_ACCESS_WIDTH],
    code_length:  uN[WEIGHT_LOG][PARALLEL_ACCESS_WIDTH],
    code:         uN[MAX_WEIGHT][PARALLEL_ACCESS_WIDTH],
}
