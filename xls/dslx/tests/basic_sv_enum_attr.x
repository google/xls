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

#[sv_type("cool_enum")]
enum MyEnum : u2 {
    FOO = 0,
    BAR = 1,
}

fn main(xs: MyEnum[2], i: u1) -> MyEnum { xs[i] }

#[test]
fn main_test() {
    let xs: MyEnum[2] = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];
    assert_eq(MyEnum::FOO, main(xs, u1:0));
    assert_eq(MyEnum::BAR, main(xs, u1:1));
}
