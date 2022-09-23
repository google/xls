// Copyright 2021 The XLS Authors
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

// Test for GitHub issue #304 - syntax error calling a parametric
// function from a parametric function.
// #1: Test invoking a local parametric.
fn local_clog2<N: u32>(x: bits[N]) -> bits[N] {
  if x >= bits[N]:1 {
    (N as bits[N]) - clz(x-bits[N]:1)
  } else {
    bits[N]:0
  }
}

fn dot_product_local<BITCOUNT: u32, LENGTH: u32,
    IDX_BITS: u32 = local_clog2<u32:32>(LENGTH) + u32:1>
  (a: bits[BITCOUNT][LENGTH], b: bits[BITCOUNT][LENGTH]) -> bits[BITCOUNT]{
  let _ = trace!(BITCOUNT);
  let _ = trace!(LENGTH);
  let _ = trace!(IDX_BITS);

  for(idx, acc): (bits[IDX_BITS], bits[BITCOUNT]) in bits[IDX_BITS]:0..LENGTH as bits[IDX_BITS] {
    let _ = trace!(idx);
    let partial_product = a[idx] * b[idx];
    acc + partial_product
  } (u32:0)
}

#[test]
fn parametric_call_local_parametric() {
  let a = [u32:0, u32:1, u32:2, u32:3];
  let b = [u32:4, u32:5, u32:6, u32:7];
  let actual = dot_product_local(a, b);
  let expected = u32:38;
  let _ = assert_eq(actual, expected);
  ()
}

// #2: Test invoking a imported ("ColonRef") parametric.
import std

fn dot_product_modref<BITCOUNT: u32, LENGTH: u32,
    IDX_BITS: u32 = std::clog2<u32:32>(LENGTH) + u32:1>
  (a: bits[BITCOUNT][LENGTH], b: bits[BITCOUNT][LENGTH]) -> bits[BITCOUNT]{
  let _ = trace!(IDX_BITS);

  for(idx, acc): (bits[IDX_BITS], bits[BITCOUNT])
    in bits[IDX_BITS]:0..LENGTH as bits[IDX_BITS] {

    let partial_product = a[idx] * b[idx];
    acc + partial_product
  } (u32:0)
}

#[test]
fn parametric_call_modref_parametric() {
  let a = [u32:0, u32:1, u32:2, u32:3];
  let b = [u32:4, u32:5, u32:6, u32:7];
  let actual = dot_product_modref(a, b);
  let expected = u32:38;
  let _ = assert_eq(actual, expected);
  ()
}
