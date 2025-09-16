// Copyright 2025 The XLS Authors
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

type BlockSize = common::BlockSize;

// The simples heuristic the estimate block size for input data,
// to be used as the first stage of data splitting.
// uses:
//   * RFC-defined maximum block size,
//   * parameter-defined maximum block size,
//   * size of input data
// to provide the result BlockSize
pub fn get_block_size(
    size: u32,
    max_block_size: BlockSize,
) -> BlockSize {
    std::min(size, std::min(max_block_size, common::MAX_BLOCK_SIZE) as u32) as BlockSize
}

#[test]
fn test_get_block_size() {
    // size greater than the protocol allows to fit in a block - 129KB
    let size = (u32:129 << 10) as u32;
    assert_eq(
        get_block_size(size, common::MAX_BLOCK_SIZE),
        common::MAX_BLOCK_SIZE as BlockSize
    );

    // size greater than the selected block size, below RFC-defined limit
    let size = (u32:100 << 10) as u32; // 100KB
    let selected_block_size = (u32:90 << 10) as BlockSize; // 90KB
    assert_eq(
        get_block_size(size, selected_block_size),
        selected_block_size as BlockSize
    );

    // size smaller than the selected block size
    let size = u32:100;
    assert_eq(
        get_block_size(size, selected_block_size),
        size as BlockSize
    );

    // zero data to compress
    let size = u32:0;
    assert_eq(
        get_block_size(size, selected_block_size),
        size as BlockSize
    );
}
