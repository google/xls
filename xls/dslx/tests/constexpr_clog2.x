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

import std

const THING_COUNT_IN_CRUMBS = u32:2;
const THING_COUNT = THING_COUNT_IN_CRUMBS * u32:2;
const THING_BITS = std::clog2(THING_COUNT);

// TODO(https://github.com/google/xls/issues/246) We should be able to use the
// clog2 value in the type system, but as of right now we cannot.
// fn max_value() -> bits[THING_BITS] {
//   bits[THING_BITS]:-1
// }

fn main() -> u32 {
  THING_BITS
}

#[test]
fn test_main() {
  let _ = assert_eq(u32:2, THING_BITS);
  let _ = assert_eq(u32:2, main());
  ()
}
