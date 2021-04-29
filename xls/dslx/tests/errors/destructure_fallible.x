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

// Test case that showed an IR conversion issue in the development of
// synthesizable fail!() support -- f() is fallible, and the destructuring in
// main() had an issue as a result.

fn f(x: u8) -> (u8, u8, u8) {
  let (a, b) = match x {
    _ => fail!((u8:0, u8:0))
  };
  (a, x, b)
}

fn main(x: u8) -> () {
  let t: (u8, u8, u8) = f(x);
  let (a, b, c) = t;
  ()
}

#![test]
fn main_test() {
  let x: u8 = u8:0;
  let result = main(x);
  assert_eq((), result)
}
