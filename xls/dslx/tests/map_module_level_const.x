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

const X = u32[3]:[1, 2, 3];

fn add_one(x: u32) -> u32 { x + u32:1 }

const Y: u32[3] = map(X, add_one);

// Sample entry point that uses the constants that we can convert to IR.
fn main() -> u32 { X[0] + Y[0] }

#[test]
fn test_top_level_map_is_ok() { assert_eq(Y, u32[3]:[2, 3, 4]) }
