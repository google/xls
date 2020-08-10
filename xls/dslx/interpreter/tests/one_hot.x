// Copyright 2020 Google LLC
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

test one_hot {
  // LSb has priority.
  let _ = assert_eq(u4:0b1000, one_hot(u3:0b000, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b001, true)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b010, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b011, true)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b100, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b101, true)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b110, true)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b111, true)) in
  // MSb has priority.
  let _ = assert_eq(u4:0b1000, one_hot(u3:0b000, false)) in
  let _ = assert_eq(u4:0b0001, one_hot(u3:0b001, false)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b010, false)) in
  let _ = assert_eq(u4:0b0010, one_hot(u3:0b011, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b100, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b101, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b110, false)) in
  let _ = assert_eq(u4:0b0100, one_hot(u3:0b111, false)) in
  ()
}
