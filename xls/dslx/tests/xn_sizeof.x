// Copyright 2024 The XLS Authors
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

fn sizeof<S: bool, N: u32>(x: xN[S][N]) -> u32 { N }

fn main() -> (u32, u32) { (sizeof(u4:0), sizeof(s4:0)) }

#[test]
fn test_main() { assert_eq(main(), (u32:4, u32:4)); }

#[test]
fn test_sizeof() {
    assert_eq(sizeof(u1:0), u32:1);
    assert_eq(sizeof(u2:0), u32:2);
    assert_eq(sizeof(u3:0), u32:3);
    assert_eq(sizeof(u4:0), u32:4);
    assert_eq(sizeof(u5:0), u32:5);
    assert_eq(sizeof(u6:0), u32:6);
    assert_eq(sizeof(u7:0), u32:7);

    assert_eq(sizeof(s1:0), u32:1);
    assert_eq(sizeof(s2:0), u32:2);
    assert_eq(sizeof(s3:0), u32:3);
    assert_eq(sizeof(s4:0), u32:4);
    assert_eq(sizeof(s5:0), u32:5);
    assert_eq(sizeof(s6:0), u32:6);
    assert_eq(sizeof(s7:0), u32:7);
}
