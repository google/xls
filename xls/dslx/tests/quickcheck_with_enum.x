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

enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
}

fn is_valid(x: MyEnum) -> bool { x == MyEnum::A || x == MyEnum::B || x == MyEnum::C }

#[quickcheck(exhaustive)]
fn prop_enum_is_valid(x: MyEnum) -> bool { is_valid(x) }
