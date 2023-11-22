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

fn i() -> () {
    // Block that's just a type alias.
    let w: () = { type Useless = u8; };
    w
}

fn j() -> () {
    // Trailing semi means block returns unit.
    let z: () = { u32:42; };
    z
}

fn main() -> () {
    let x: () = i();
    let y: () = j();
}

#[test]
fn test_i() { assert_eq(i(), ()) }

#[test]
fn test_j() { assert_eq(j(), ()) }
