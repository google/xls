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

const CONST_1 = u32:666;
const CONST_2 = CONST_1;

const fn const_get() -> u32 { CONST_2 }
const fn const_adder() -> u32 { CONST_1 + CONST_2 }

fn main() -> u32 {
    const_adder() + const_get() + u32::MAX
}

#[test]
fn can_add_const() {
    assert_eq(const_adder(), u32:1332);
}
