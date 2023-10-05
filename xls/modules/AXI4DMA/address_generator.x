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

struct RawAddressData<ADDR_WIDTH: u32> {
  base_address: bits[ADDR_WIDTH],
  burst_length: bits[8],
  burst_size: bits[3],
  burst_type: bits[2],
}

struct AlignedAddress<ADDR_wIDTH: u32, DATA_WIDTH: u32, DATA_BYTE_WIDTH = DATA_WIDTH / u32:8> {
  aligned_address: bits[ADDR_WIDTH],
  active_bytes: bits[DATA_BYTE_WIDTH],
}

struct 

pub proc AddressGenerationAndSlicing<ADDR_WIDTH: u32, DATA_WIDTH: u32> {
  input_r: chan<RawAddressData> in;
  output_r: chan<AlignedAddress> out;
  init{()}
}
