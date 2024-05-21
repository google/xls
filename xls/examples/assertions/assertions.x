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

// Example of assertions being used in functions.

fn func_0<N: u32>(x: bits[N]) -> bits[N] {
    const_assert!(N == u32:32);
    assert!(x < u32:5, "x_less_than_5");

    x + u32:10
}

fn main(y: u32) -> u32 {
    if y < u32:10 {
        func_0<u32:32>(y) + u32:20
    } else if y < u32:20 {
        func_0<u32:32>(y - u32:10)
    } else {
        fail!("y_ge_than_21", u32:0)
    }
}

#[test]
fn assertion_test_ok() {
    assert_eq(main(u32:4), u32:34);
    assert_eq(main(u32:10), u32:10);
    assert_eq(main(u32:14), u32:14);
}
// Enable once https://github.com/google/xls/issues/481 is resolved.
//
// #[test]
// fn assertion_test_fail0() {
//   assert_fail(main(u32:9));
// }
//
// #[test]
// fn assertion_test_fail1() {
//   assert_fail(main(u32:15));
// }
//
// #[test]
// fn assertion_test_fail2() {
//   assert_fail(main(u32:21));
// }
