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
fn test_arbitrary(x: u32) -> bool {
    x == x
}

#[fuzz_test(domains=`u32:10..20`)]
fn test_range(x: u32) -> bool {
    x >= u32:10 && x < u32:20
}

#[fuzz_test(domains=`[u32:5, 10, 15]`)]
fn test_element_of(x: u32) -> bool {
    x == u32:5 || x == u32:10 || x == u32:15
}
