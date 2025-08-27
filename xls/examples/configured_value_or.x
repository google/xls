// Copyright 2025 The XLS Authors
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

// Simple demonstration of configured_values_or usage. Overwrite the default
// values for bool, u32, s32, and enum types.

enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
}

fn main() -> (bool, u32, s32, MyEnum, bool, u32, s32, MyEnum) {
    let b_default = configured_value_or<bool>("b_default", false);
    let u_default = configured_value_or<u32>("u32_default", u32:42);
    let s_default = configured_value_or<s32>("s32_default", s32:-100);
    let e_default = configured_value_or<MyEnum>("enum_default", MyEnum::C);
    let b_override = configured_value_or<bool>("b_override", false);
    let u_override = configured_value_or<u32>("u32_override", u32:42);
    let s_override = configured_value_or<s32>("s32_override", s32:-100);
    let e_override = configured_value_or<MyEnum>("enum_override", MyEnum::C);
    (b_default, u_default, s_default, e_default, b_override, u_override, s_override, e_override)
}
