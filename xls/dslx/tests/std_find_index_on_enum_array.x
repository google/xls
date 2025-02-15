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

//! Validates that we can find the index of a value in an enum array.
//!
//! This is what is required since we cannot currently write routines that are
//! parameterized on "either some kind of bits or some kind of enum"
//! values.

import std;

type MyEnumRaw = u2;

enum MyEnum : MyEnumRaw {
    Available = 0,
    BusyForRead = 1,
    BusyForWrite = 2,
}

fn my_enum_to_underlying(x: MyEnum) -> MyEnumRaw { x as MyEnumRaw }

const COUNT: u32 = u32:3;

fn main(x: MyEnum[COUNT]) -> (bool, u32) {
    std::find_index(map(x, my_enum_to_underlying), my_enum_to_underlying(MyEnum::Available))
}

#[test]
fn main_test() {
    let x: MyEnum[COUNT] = [MyEnum::Available, MyEnum::BusyForRead, MyEnum::BusyForWrite];
    assert_eq(main(x), (true, u32:0));

    let y: MyEnum[COUNT] = [MyEnum::BusyForRead, MyEnum::BusyForWrite, MyEnum::Available];
    assert_eq(main(y), (true, u32:2));

    let z: MyEnum[COUNT] = [MyEnum::BusyForRead, MyEnum::BusyForWrite, MyEnum::BusyForRead];
    assert_eq(main(z), (false, u32:0));
}
