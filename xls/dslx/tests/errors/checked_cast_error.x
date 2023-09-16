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

fn main(x : u32) -> u16 {
  checked_cast<u16>(x)
}

#[test]
fn test_main() {
  assert_eq(main(u32:0), u16:0);
  assert_eq(main(u32:5), u16:5);
  assert_eq(main(u32:0xffff), u16:0xffff);
  assert_eq(main(u32:0x1ffff), u16:0xffff);
}
