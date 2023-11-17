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

fn main() -> u30 {
    let x0 = decode<u5>(u3:0b010);
    let x1 = decode<u16>(x0);
    decode<u30>(x1)
}

#[test]
fn decode_test() {
    assert_eq(u5:0b00001, decode<u5>(u3:0b000));
    assert_eq(u5:0b00010, decode<u5>(u3:0b001));
    assert_eq(u5:0b00100, decode<u5>(u3:0b010));
    assert_eq(u5:0b01000, decode<u5>(u3:0b011));
    assert_eq(u5:0b10000, decode<u5>(u3:0b100));
    assert_eq(u5:0b00000, decode<u5>(u3:0b101));
    assert_eq(u5:0b00000, decode<u5>(u3:0b110));
    assert_eq(u5:0b00000, decode<u5>(u3:0b111));
}
