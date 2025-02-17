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

pub fn is_x(x: u32) -> bool {
    x > u32:0
}

pub fn is_y(y: u32) -> (bool, u32) {
    (y > u32:0, y)
}

pub fn main(x: u32, y: u32) -> bool {
    let is: bool = is_x(x);
    let (is, y): (bool, u32) = is_y(y);
    let is_z: bool = is_x && is_y;
    is_z
}
