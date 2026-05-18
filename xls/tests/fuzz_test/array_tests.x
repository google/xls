// Copyright 2026 The XLS Authors
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

#[fuzz_test(domains=`()`)]
fn arbitrary_array(x: u32[3]) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn array_of_tuples(x: (u32, u32)[2]) -> bool {
    true
}

#[fuzz_test(domains=`((), u32:0..9)`)]
fn tuple_with_array(x: (u32[2], u32)) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn big_array(x: uN[128][3]) -> bool {
    true
}

#[fuzz_test(domains=`()`)]
fn tuple_with_big_array(x: (uN[128][2], u32)) -> bool {
    true
}
