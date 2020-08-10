// Copyright 2020 Google LLC
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

// Contains a for loop that iterates up to a CONST value.

const FOO = u32:3;

fn f() -> u32 {
  for (i, accum): (u32, u32) in range(u32:0, FOO) {
    accum+i
  }(u32:0)
}

test f {
  assert_eq(f(), u32:3)
}
