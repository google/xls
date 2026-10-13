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

fn main(a: u32[3]) -> u32 {
    let result: u32 = for (value, accum): (u32, u32) in a {
        accum + value
    }(u32:0);
    result
}

#[test]
fn for_over_array_test() {
    let a: u32[3] = u32[3]:[1, 2, 3];
    assert_eq(u32:6, main(a))
}

fn sum_pairs(pairs: (u32, u32)[2]) -> u32 {
    for ((a, b), acc): ((u32, u32), u32) in pairs {
        acc + a + b
    }(u32:0)
}

#[test]
fn sum_pairs_test() {
    let pairs = [(u32:1, u32:2), (u32:3, u32:4)];
    assert_eq(sum_pairs(pairs), u32:10);
}

fn sum_nested_tuples(items: ((u32, u32, u32), (u32, u32))[2]) -> u32 {
    for (((a, _, c), (_, e)), acc) in items {
        acc + a + c + e
    }(u32:0)
}

#[test]
fn sum_nested_tuples_test() {
    let items =
        [((u32:1, u32:100, u32:2), (u32:200, u32:3)), ((u32:4, u32:300, u32:5), (u32:400, u32:6))];
    assert_eq(sum_nested_tuples(items), u32:21);
}

fn sum_pairs_nested_for(pairs: (u32, u32)[2]) -> u32 {
    for ((a, b), acc): ((u32, u32), u32) in pairs {
        for (i, inner_acc): (u32, u32) in u32:0..u32:2 {
            inner_acc + a + b + i
        }(acc)
    }(u32:0)
}

#[test]
fn sum_pairs_nested_for_test() {
    let pairs = [(u32:1, u32:2), (u32:3, u32:4)];
    assert_eq(sum_pairs_nested_for(pairs), u32:22);
}
