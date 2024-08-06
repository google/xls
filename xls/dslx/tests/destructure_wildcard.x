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

#[test]
fn destructure_test() {
    let t = (u32:2, u8:3, true);
    let (_, _, v) = t;
    assert_eq(v, true)
}

#[test]
fn destructure_rest_of_tuple_test() {
    let t = (u32:2, u8:3);

    // Since this has the "correct" number of matches, this acts like a
    // wildcard, for now.
    let (a, ..) = t;
    assert_eq(u32:2, a)
}
