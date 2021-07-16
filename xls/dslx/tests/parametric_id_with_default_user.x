// Copyright 2021 The XLS Authors
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

// A module that *uses* the parametric function "id" that has a default
// parametric value.

import xls.dslx.tests.mod_parametric_id_with_default

fn main() -> (u5, u8) {
  let x: u5 = mod_parametric_id_with_default::id();
  let y: u8 = mod_parametric_id_with_default::id<u32:8>();
  (x, y)
}

#![test]
fn test_main() {
  assert_eq((u5:31, u8:31), main())
}
