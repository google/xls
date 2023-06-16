// Copyright 2023 The XLS Authors
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

pub fn replicate
  <NumCopies : u32, N : u32,
  L:u32 = {NumCopies * N},
  OneLess:u32 = {NumCopies - u32:1}>(
  source : uN[N])
  -> uN[L] {
  if NumCopies > u32:1 {
    source ++ replicate<OneLess>(source)
  } else {
    source
  }
}

#[test]
fn test_replicate() {
  assert_eq(
    replicate<u32:1>(u3:0b101), u3:0b001);
  ()
}
