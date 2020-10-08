// Copyright 2020 The XLS Authors
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

const FALSE = bool:false;

pub fn my_lsb_uses_const(x: u3) -> u1 {
  // Force the function to try to use the enclosing scope by referring to FALSE
  // here.
  let f: bool = FALSE;
  let y: u1 = std::lsb(x);
  y || f
}

pub fn my_lsb(x: u3) -> u1 {
  std::lsb(x)
}

pub struct Point {
  x: u32,
  y: u32,
}

pub enum MyEnum : u8 {
  FOO = 42,
  BAR = 64,
}
