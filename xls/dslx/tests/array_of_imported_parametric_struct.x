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

// Test to verify that we can instantiate and use an array of imported
// parametric structs, to exercise https://github.com/google/xls/issues/1030

import xls.dslx.tests.parametric_import;

const GLOB_ARR = parametric_import::Type<5, 10>[2]:[
    parametric_import::Type { x: u5:1, y: u10:2 }, parametric_import::Type { x: u5:3, y: u10:4 },
];

fn main() -> parametric_import::Type<u32:5, u32:10> { GLOB_ARR[0] }

#[test]
fn test_global() {
    assert_eq(GLOB_ARR[0].x, u5:1);
    assert_eq(GLOB_ARR[0].y, u10:2);
    assert_eq(GLOB_ARR[1].x, u5:3);
    assert_eq(GLOB_ARR[1].y, u10:4);
}

#[test]
fn test_local() {
    let arr = parametric_import::Type<u32:10, u32:20>[2]:[
        parametric_import::Type { x: u10:1, y: u20:2 },
        parametric_import::Type { x: u10:3, y: u20:4 },
    ];
    assert_eq(arr[0].x, u10:1);
    assert_eq(arr[0].y, u20:2);
    assert_eq(arr[1].x, u10:3);
    assert_eq(arr[1].y, u20:4);
}
