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

fn hello() {
  let x = u8:0xF0;
  let _ = trace_fmt!("Hello world!");
  let _ = trace_fmt!("x is {}, {:#x} in hex and {:#b} in binary", x, x, x);
  let y = u32:17;
  let _ = trace_fmt!("y is 32'd{:d}, 32'h{:x} and 32'b{:b}", y, y, y);
  ()
}

fn main() {
  hello()
}

#[test]
fn hello_test() { hello() }
