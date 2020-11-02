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

type MyTuple = (u8,);

fn f<N: u32>(xs: MyTuple[N]) -> u32 {
  N
}

fn main() -> u32 {
  let xs: MyTuple[3] = MyTuple[3]:[(u8:0,), (u8:1,), (u8:2,)];
  f(xs)
}

test main {
  assert_eq(u32:3, main())
}
