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

#[test]
fn bit_slice_test() {
  let _ = assert_eq(u2:0b11, bit_slice(u6:0b100111, u6:0, u2:0));
  let _ = assert_eq(u2:0b11, bit_slice(u6:0b100111, u6:1, u2:0));
  let _ = assert_eq(u2:0b01, bit_slice(u6:0b100111, u6:2, u2:0));
  let _ = assert_eq(u2:0b00, bit_slice(u6:0b100111, u6:3, u2:0));

  let _ = assert_eq(u3:0b111, bit_slice(u6:0b100111, u6:0, u3:0));
  let _ = assert_eq(u3:0b011, bit_slice(u6:0b100111, u6:1, u3:0));
  let _ = assert_eq(u3:0b001, bit_slice(u6:0b100111, u6:2, u3:0));
  let _ = assert_eq(u3:0b100, bit_slice(u6:0b100111, u6:3, u3:0));
  ()
}
