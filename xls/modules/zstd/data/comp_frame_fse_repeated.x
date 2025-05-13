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
  u32:12
>[1] = [DataArray<64, 12>{
  length: u32:92,
  array_length: u32:12,
  data: uN[64][12]:[uN[64]:0x003e2484fd2fb528, uN[64]:0x02d6790000840000, uN[64]:0xff86117f06110168, uN[64]:0x0000440452dd7eff, uN[64]:0x04b6010674016531, uN[64]:0x001c00b32100001c, uN[64]:0x0100001c00e70100, uN[64]:0x00bd0100001c000d, uN[64]:0x001c003a0100001c, uN[64]:0x0100001c007f0100, uN[64]:0x00690100001d006b, uN[64]:0x993d99b6]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:8
>[1] = [DataArray<64, 8>{
  length: u32:62,
  array_length: u32:8,
  data: uN[64][8]:[uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6d6d6d6d6d6d6, uN[64]:0xd6d6656565656565, uN[64]:0xb3b3b3b3d6d6]
}];
