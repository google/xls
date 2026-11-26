// Copyright 2026 The XLS Authors
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

// A derived `default()` zeroes plain fields and calls `default()` on nested
// structs, so their custom defaults are kept, unlike with `zero!`.

struct Color { r: u8, g: u8, b: u8 }

// Defaults to white rather than the all-zeros black.
impl Color {
    fn default() -> Self { Color { r: u8:255, g: u8:255, b: u8:255 } }
}

#[derive(Default)]
struct Pixel { x: u16, y: u16, color: Color }

fn origin() -> Pixel { Pixel::default() }

#[test]
fn test_default() {
    let white = Color { r: u8:255, g: u8:255, b: u8:255 };
    assert_eq(origin(), Pixel { x: u16:0, y: u16:0, color: white });
    assert_eq(zero!<Pixel>().color, Color { r: u8:0, g: u8:0, b: u8:0 });
}
