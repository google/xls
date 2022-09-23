// Copyright 2022 The XLS Authors
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
pub struct MyStruct {
  a: s32,
  b: u64,
}

pub const MY_STRUCT = MyStruct { a: s32:0, b: u64:0xbeef } ;
pub const MY_OTHER_STRUCT = MyStruct { a: s32:1, b: u64:0xfeed } ;

pub fn impl<param: MyStruct>() -> u64 {
  param.b
}

pub fn other_impl<x: MyStruct, y: MyStruct>() -> u64{
  (x.a + y.a) as u64 + x.b + y.b
}

fn main() -> u64{
  impl<MY_STRUCT>() + other_impl<MY_STRUCT, MY_OTHER_STRUCT>()
}

#[test]
fn main_test() {
  let _ = assert_eq(u64:0xbeef, impl<MY_STRUCT>());
  let _ = assert_eq(u64:0xfeed, impl<MY_OTHER_STRUCT>());
  let _ = assert_eq(u64:0x1bddd, other_impl<MY_STRUCT, MY_OTHER_STRUCT>());
  ()
}
