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

// Tests for various sizes of binary operations.
// Simple test of these ops and that wide data types work...
// and as a starting point for debugging they don't.
fn main32() -> sN[32] {
  let x = sN[32]:1000 in
  let y = sN[32]:-1000 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}

fn main1k() -> sN[1024] {
  let x = sN[1024]:1 in
  let y = sN[1024]:-3 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}

fn main() -> sN[128] {
  let x = sN[128]:1 in
  let y = sN[128]:-3 in
  let add = x + y in
  let mul = add * y in
  let shll = mul << y in
  let shra = mul >>> x in
  let shrl = mul >> x in
  let sub = shrl - y in
  sub / y
}
