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

fn p<N: u32>(_: bits[N]) -> u8 {
  N as u8
}

fn f() -> u8 {
  match false {
    // TODO(cdleary): 2020-08-05 Turn this match arm into a wildcard match when
    // https://github.com/google/xls/issues/75 is resolved.
    false => p(u8:0),
    _ => u8:0
  }
}

#![test]
fn t() {
  assert_eq(u8:8, f())
}
