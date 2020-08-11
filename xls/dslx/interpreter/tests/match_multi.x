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

fn match_multi(x: u32) -> u32 {
  match x {
    u32:24 | u32:42 => u32:42;
    _ => u32:64
  }
}

test match_multi {
  let _ = assert_eq(u32:42, match_multi(u32:24));
  let _ = assert_eq(u32:42, match_multi(u32:42));
  let _ = assert_eq(u32:64, match_multi(u32:41));
  ()
}
