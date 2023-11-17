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

fn main() -> u2 {
    let x0 = encode(u32:0x00010000);
    let x1 = encode(x0);
    encode(x1)
}

#[test]
fn encode_test() {
    assert_eq(u2:0b00, encode(u3:0b000));
    assert_eq(u2:0b00, encode(u3:0b001));
    assert_eq(u2:0b01, encode(u3:0b010));
    assert_eq(u2:0b01, encode(u3:0b011));
    assert_eq(u2:0b10, encode(u3:0b100));
    assert_eq(u2:0b10, encode(u3:0b101));
    assert_eq(u2:0b11, encode(u3:0b110));
    assert_eq(u2:0b11, encode(u3:0b111));
}
