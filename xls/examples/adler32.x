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

// Perform a (sequential) Adler-32 checksum over a single input byte.

// Helper Routine: Get remainder from a 32-bit divide.
fn mod(dividend: u32, divisor: u32) -> u32 {
  let quotient: u32 = u32:0 in
  let remainder: u32 = dividend in
  let term: u64 = u64:1 << u64:32 in
  let product: u64 = (divisor as u64) << u64:32 in
  let (quotient, remainder, product, term) =
     for (i, (quotient, remainder, product, term)):
             (u32, (u32, u32, u64, u64))
     in range(u32:0, u32:32) {
       let product = product >> u64:1 in
       let term = term >> u64:1 in
       let (new_q, new_r) : (u32, u32) =
         match product <= (remainder as u64) {
           true => (quotient + (term as u32), remainder - (product as u32));
           _ => (quotient, remainder);
         } in
       (new_q, new_r, product, term)
    }((quotient, remainder, product, term)) in
  remainder
}

fn adler32_seq(buf: u8) -> u32 {
  let a = u32:1 in
  let b = u32:0 in
  // Iterate only over input of length 1, for now.
  let (a, b) = for (i, (a, b)): (u8, (u32, u32)) in range(u8:0, u8:1) {
    let a = mod(a + (buf as u32), u32:65521) in
    let b = mod(b + a, u32:65521) in
    (a, b)
  }((a, b)) in
  (b << u32:16) | a
}

fn main(message: u8) -> u32 {
  adler32_seq(message)
}

test adler32_one_char {
  let _ = assert_eq(u32:0x0010001, main(u8:0x00)) in  // dec 0
  let _ = assert_eq(u32:0x0310031, main(u8:0x30)) in  // '0'
  let _ = assert_eq(u32:0x0620062, main(u8:0x61)) in  // 'a'
  let _ = assert_eq(u32:0x07f007f, main(u8:0x7e)) in  // '~' (dec 126)
  let _ = assert_eq(u32:0x0800080, main(u8:0x7f)) in  // 'DEL' (dec 127)
  assert_eq(u32:0x1000100, main(u8:0xFf))             // dec 255
}
