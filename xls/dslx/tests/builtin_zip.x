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

const A = u32[3]:[1, 2, 3];
const B = u64[3]:[4, 5, 6];

fn main() -> (u32, u64) { zip(A, B)[1] }

#[test]
fn test_main() { assert_eq((u32:2, u64:5), main()) }
