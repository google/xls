// Copyright 2023 The XLS Authors
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

type MyType = u2;
type StructLike = (MyType[2]);

fn f(s: StructLike) -> StructLike {
    let updated: StructLike = ([s.0[u32:0] + MyType:1, s.0[u32:1] + MyType:1],);
    updated
}

#[test]
fn t_test() {
    let s: StructLike = (MyType[2]:[MyType:0, MyType:1],);
    let s_2 = f(s);
    assert_eq(s_2, (u2[2]:[u2:1, u2:2],))
}
