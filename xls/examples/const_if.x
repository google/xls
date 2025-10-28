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

#![feature(type_inference_v2)]

fn _const_if_differing_types<A: u32>(a : u32) -> u32 {
  let data = const if A == u32:0 {
    u16:0
  } else if A == u32:1 {
    u32:0
  } else {
    u8:0
  };

  data as u32
}

fn const_if_differing_types(a: u32) -> u32 {
  _const_if_differing_types<u32:1>(a)
}

fn my_function(a: u32, b: u32) -> u32 {
  a * b
}

fn const_if(a: u32) -> u32 {
  const if true {
    a + u32:2
  } else {
    my_function(a, u32:10)
  }
}

#[test]
fn const_if_test() {
  assert_eq(const_if(u32:0), u32:2);
  assert_eq(const_if(u32:2), u32:4);
}

fn normal_if(a: u32) -> u32 {
  if true {
    a + u32:2
  } else {
    my_function(a, u32:10)
  }
}

#[test]
fn normal_if_test() {
  assert_eq(normal_if(u32:0), u32:2);
  assert_eq(normal_if(u32:2), u32:4);
}
