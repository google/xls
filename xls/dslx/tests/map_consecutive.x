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

// A test that does two consecutive calls to the map() builtin.

fn id(x: u32) -> u32 { x }

#[test]
fn test_main() {
    let x = u32[1]:[0];
    let y0 = map(x, id);
    let y1 = map(x, id);
    assert_eq(y0, y1)
}
