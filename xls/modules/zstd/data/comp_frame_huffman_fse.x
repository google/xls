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


import xls.modules.zstd.common;
type DataArray = common::DataArray;

pub const FRAMES:DataArray<
  u32:64,
  u32:50
>[1] = [DataArray<64, 50>{
  length: u32:64,
  array_length: u32:8,
  data: uN[64][50]:[uN[64]:0x007e4f84fd2fb528, uN[64]:0x00068e00017d0000, uN[64]:0xd5764f39f0080008, uN[64]:0x04000400045c4f40, uN[64]:0xcfefff3e7fefff00, uN[64]:0x5dff77afbdffef3f, uN[64]:0x1de190b0000301fb, uN[64]:0x807e83a8084e0c21, uN[64]:0, ...]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:50
>[1] = [DataArray<64, 50>{
  length: u32:126,
  array_length: u32:16,
  data: uN[64][50]:[uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e680e6e6, uN[64]:0xe6e6e6b3e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0x80e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6b3e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6b3b3e6e6, uN[64]:0xe6e6e6b3e6e6e6b3, uN[64]:0xe6e6e6e6e6e6b3e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6e6e6e6e6e6b3, uN[64]:0xb3e6e6b3b3e6b3e6, uN[64]:0xe6e6e6e6e6e6e6e6, uN[64]:0xe6e6b3e6e6b3, uN[64]:0, ...]
}];
