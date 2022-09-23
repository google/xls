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

// Defines type aliases based on const bindings.
//
// Regression tests noted in https://github.com/google/xls/issues/230

// Note: faked definition specialized to reduce complexity, but still parametric
// invocation.
fn clog2<N: u32>(x: bits[N]) -> bits[N] {
  u32:4
}

const VALUE_LIMIT = u32:16;
const VALUE_LIMIT_BITS = clog2(VALUE_LIMIT);

// Use the pre-defined constant.
type MyType = bits[VALUE_LIMIT_BITS];

// Use the invocation inline.
const VALUE_LIMIT_2 = u32:16;
type MyType2 = bits[clog2(VALUE_LIMIT_2)];

fn main(x: MyType) -> MyType2 {
  x
}

#[test]
fn test_main() {
  let _ = assert_eq(VALUE_LIMIT_BITS, u32:4);
  assert_eq(main(u4:0xa), u4:0xa)
}
