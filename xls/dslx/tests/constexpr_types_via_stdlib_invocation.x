// Copyright 2021 The XLS Authors
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

// Defines type aliases via inline constexpr -- this variant uses the standard
// library (so is somewhat more complex as a result, exercising import machinery
// and running the full clog2 function).
//
// Regression tests noted in https://github.com/google/xls/issues/230

import std

const VALUE_LIMIT = u32:16;
type MyType = bits[std::clog2(VALUE_LIMIT)];

fn main(x: MyType) -> MyType {
  x
}

#[test]
fn test_main() {
  assert_eq(main(u4:0xa), u4:0xa)
}
