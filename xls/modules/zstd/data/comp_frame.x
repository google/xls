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

pub struct DataArray<BITS_PER_WORD: u32, LENGTH: u32>{
  data: uN[BITS_PER_WORD][LENGTH],
  length: u32,
  array_length: u32
}
pub const FRAMES:DataArray<
  u32:64,
  u32:7
>[1] = [DataArray<64, 7>{
  length: u32:51,
  array_length: u32:7,
  data: uN[64][7]:[uN[64]:0x00504784fd2fb528, uN[64]:0xcf95700001150000, uN[64]:0xe17d50b989ac93c4, uN[64]:0x0daf000895a6e608, uN[64]:0xb96010b86f7602a4, uN[64]:0x05b0e051238666e8, uN[64]:0x8470e3]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:10
>[1] = [DataArray<64, 10>{
  length: u32:80,
  array_length: u32:10,
  data: uN[64][10]:[uN[64]:0xc4c4cf95cf95cf95, uN[64]:0x93c4c4c4c4c4c4c4, uN[64]:0xacc493c493c493c4, uN[64]:0xc493c493c493c489, uN[64]:0x93c493c489acc493, uN[64]:0x08e17d50b9c493c4, uN[64]:0xc4c4c4cf9595a6e6, uN[64]:0x93c493c4c4c4c4c4, uN[64]:0xc489acc493c493c4, uN[64]:0xc493c493c493c493]
}];
