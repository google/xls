// Copyright 2020 The XLS Authors
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

// Performs a table-less crc32 of the input data as in Hacker's Delight:
// https://www.hackersdelight.org/hdcodetxt/crc.c.txt (roughly flavor b)
import std;

const U32_MAX = std::unsigned_max_value<u32:32>();

fn crc32_one_byte(byte: u8, polynomial: u32, crc: u32) -> u32 {
    let crc = crc ^ (byte as u32);
    // 8 rounds of updates.
    for (_, crc): (u32, u32) in range(u32:0, u32:8) {
        let mask = -(crc & u32:1);
        (crc >> u32:1) ^ (polynomial & mask)
    }(crc)
}

fn main(message: u8) -> u32 { crc32_one_byte(message, u32:0xEDB88320, U32_MAX) ^ U32_MAX }

#[test]
fn crc32_one_char() { assert_eq(u32:0x83DCEFB7, main('1')) }
