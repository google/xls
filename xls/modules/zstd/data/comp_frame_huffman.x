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
  length: u32:93,
  array_length: u32:12,
  data: uN[64][12]:[uN[64]:0x00704484fd2fb528, uN[64]:0xac033a0002650000, uN[64]:0x1111111111118e00, uN[64]:0x0007000700071011, uN[64]:0x131a053a5606874c, uN[64]:0x93b7146cb45c3584, uN[64]:0x06499215949aa275, uN[64]:0x0132000c0126fd3b, uN[64]:0x15a7b54443de03b8, uN[64]:0x5da6a9b37c005000, uN[64]:0x4e0200656960219d, uN[64]:0x912a65cf0b]
}];
pub const DECOMPRESSED_FRAMES:DataArray<
  u32:64,
  u32:14
>[1] = [DataArray<64, 14>{
  length: u32:112,
  array_length: u32:14,
  data: uN[64][14]:[uN[64]:0x0a03050a0305000a, uN[64]:0x0605050a0305000a, uN[64]:0x0708050a03050600, uN[64]:0x05040b0c06040c04, uN[64]:0x050a030c0b05040b, uN[64]:0x06040c0408050308, uN[64]:0x0b05040b05040b0c, uN[64]:0x0a030301050a030c, uN[64]:0x090409050a030505, uN[64]:0x040c040507020a0a, uN[64]:0x0602070b03090c06, uN[64]:0x030d0f060b030d0f, uN[64]:0x0f06040c0408050b, uN[64]:0x020909040600030d]
}];
