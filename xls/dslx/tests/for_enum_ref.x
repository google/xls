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

enum MyEnum : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
}

fn main(x: u2) -> MyEnum {
    for (_, _): (u4, MyEnum) in u4:0..u4:4 {
        match x {
            u2:0 => MyEnum::A,
            u2:1 => MyEnum::B,
            u2:2 => MyEnum::C,
            u2:3 => MyEnum::D,
        }
    }(MyEnum::A)
}

#[test]
fn test_main() {
    assert_eq(MyEnum::A, main(u2:0));
    assert_eq(MyEnum::B, main(u2:1));
    assert_eq(MyEnum::C, main(u2:2));
    assert_eq(MyEnum::D, main(u2:3));
}
