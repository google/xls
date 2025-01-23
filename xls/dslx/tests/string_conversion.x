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

// Simple verification that character strings can be lowered to IR.

fn main() -> u8[31] {
  "abcdefghijklmnopqrstuvwxyz\"\x5F\u{58}\u{4C}\u{53}"
}

#[test]
fn main_test() {

  assert_eq("\u{0}", "\0");

  // Test unicode with 1 through 6 digits.
  assert_eq("\u{E}", u8[1]:[0xE]);
  assert_eq("\u{7E}", u8[1]:[0x7E]);
  assert_eq("\u{1FF}", u8[2]: [0xC7, 0xBF]);
  assert_eq("\u{1FBF}", u8[3]: [0xE1, 0xBE, 0xBF]);
  assert_eq("\u{12FBF}", u8[4]: [0xF0, 0x92, 0xBE, 0xBF]);
  assert_eq("\u{10CB2F}", u8[4]: [0xF4, 0x8C, 0xAC, 0xAF]);

  let expected_result:u8[31] = "abcdefghijklmnopqrstuvwxyz\"_XLS";
  assert_eq(expected_result, main())
}
