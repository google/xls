// Copyright 2020 Google LLC
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

#![cfg(let_terminator_is_semi = true)]

// Performs a table-less crc32 of the input data as in Hacker's Delight:
// https://www.hackersdelight.org/hdcodetxt/crc.c.txt (roughly flavor b)

fn crc32_one_byte_inferred(byte: u8, polynomial: u32, crc: u32) -> u32 {
  let crc = crc ^ (byte as u32);
  // 8 rounds of updates.
  for (i, crc): (u32, u32) in range(u32:0, u32:8) {
    let mask = -(crc & u32:1);
    (crc >> u32:1) ^ (polynomial & mask)
  }(crc)
}
