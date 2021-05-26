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

// Simple smoke test driver that the cover! builtin can parse & interpret.
#![test]
fn cover_smoke() {
  let cond = u1:1;
  cover!("my_coverpoint", cond)
}

// Driver for IR conversion initial check.
fn main() {
  let cond = u1:1;
  cover!("my_coverpoint", cond)
}
