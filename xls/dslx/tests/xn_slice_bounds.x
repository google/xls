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

const S = true;
const N = u32:32;

type MyS32 = xN[S][N];

fn from_to(x: u32) -> u8 { x[MyS32:0:MyS32:8] }

fn to(x: u32) -> u8 { x[:MyS32:8] }

fn from(x: u32) -> u8 { x[MyS32:-8:] }

fn main(x: u32) -> u8[3] { [from_to(x), to(x), from(x)] }

#[test]
fn test_main() {
    assert_eq(from_to(u32:0x12345678), u8:0x78);
    assert_eq(to(u32:0x12345678), u8:0x78);
    assert_eq(from(u32:0x12345678), u8:0x12);
}
