// Copyright 2026 The XLS Authors
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

import xls.build_rules.tests.configured_value_dep_a;
import xls.build_rules.tests.configured_value_dep_b;

pub fn compute() -> u32 {
  let local_val = configured_value_or<u32>("shared_key", 3);
  configured_value_dep_a::VAL + configured_value_dep_b::VAL + local_val
}

#[test]
fn test_configured_values_scoped_and_entry_override() {
  assert_eq(compute(), 450);
}
