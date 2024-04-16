// Copyright 2024 The XLS Authors
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

pub fn fast_if<N: u32>(cond: bool, arg1: uN[N], arg2: uN[N]) -> uN[N] {
    let mask = if cond { !bits[N]:0 } else { bits[N]:0 };
    (arg1 & mask) | (arg2 & !mask)
}

pub fn fast_if_tuple_2<N: u32, M: u32>
    (cond: bool, arg1: (uN[N], uN[M]), arg2: (uN[N], uN[M])) -> (uN[N], uN[M]) {
    (fast_if(cond, arg1.0, arg2.0), fast_if(cond, arg1.1, arg2.1))
}

pub fn fast_if_tuple_3<N: u32, M: u32, O: u32>
    (cond: bool, arg1: (uN[N], uN[M], uN[O]), arg2: (uN[N], uN[M], uN[O]))
    -> (uN[N], uN[M], uN[O]) {
    (fast_if(cond, arg1.0, arg2.0), fast_if(cond, arg1.1, arg2.1), fast_if(cond, arg1.2, arg2.2))
}

#[test]
fn fast_if_test() {
    assert_eq(if true { u32:1 } else { u32:5 }, fast_if(true, u32:1, u32:5));
    assert_eq(if false { u32:1 } else { u32:5 }, fast_if(false, u32:1, u32:5));

    assert_eq(
        if true {
            trace_fmt!("if: true");
            u16:74
        } else {
            trace_fmt!("if: false");
            u16:32
        },
        fast_if(true, {
            trace_fmt!("fast_if: true");
            u16:74
        }, {
            trace_fmt!("fast_if: false");
            u16:32
        })
    );
}
