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

// Verifies that trace!() can show up in a DSLX sequence w/o the
// IR converter complaining.
fn main() -> u3 {
  let x0 = clz(u3:0b111);
  let _ = trace!(x0);
  x0
}

#[test]
fn trace_test() {
  let x0 = clz(u3:0b011);
  let _ = trace!(x0);
  let x1 = (x0 as u8) * u8:3;
  let _ = trace!(x1);
  ()
}
