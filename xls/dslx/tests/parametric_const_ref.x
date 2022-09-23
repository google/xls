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

const DIM = u32:3;

fn f<N: u32>(_: u32[N]) -> u32[DIM] {
  u32[DIM]:[N, N, N]
}

#[test]
fn f_test() {
  assert_eq(u32[3]:[2, 2, 2], f(u32[2]:[0, ...]))
}
