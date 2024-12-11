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

struct Point { x: u32, y: u32 }

impl Point {
    const MY_CONST = u32:5;

    fn zero() -> Self { Point { x: u32:0, y: u32:0 } }

    fn area(self: Self) -> u32 { self.x * self.y }
}

fn main() -> u32 { Point::MY_CONST }

#[test]
fn use_impl_const() {
    type PointAlias = Point;
    assert_eq(PointAlias::MY_CONST, u32:5);
}

#[test]
fn test_area() {
    let p = Point { x: u32:2, y: u32:4 };
    assert_eq(p.area(), u32:8);
}

#[test]
fn test_zero() {
    let p = Point::zero();
    assert_eq(p.x, u32:0);
}
