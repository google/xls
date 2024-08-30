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

import std;

// ---
//
// SHA algorithm based on the description in:
//
// https://en.wikipedia.org/wiki/SHA-2#Pseudocode
//
// We attempt to mirror the pseudocode presented there fairly directly for ease
// of reproducing correct results.

pub type Digest = (u32, u32, u32, u32, u32, u32, u32, u32);

fn sha256_chunk_w_table(chunk: bits[512]) -> u32[64] {
    // Seed the "w" table with the message chunk.
    let w_init: u32[64] = (chunk ++ bits[1536]:0) as u32[64];

    // Build up the remaining values of the "w" table.
    // TODO(b/149962183): Make range go from 16 - 64 once counted for
    // ranges can start at values other than 0.
    let w: u32[64] = for (i, w): (u32, u32[64]) in range(u32:0, u32:48) {
        let w_im15: u32 = w[i + u32:16 - u32:15];
        let s_0: u32 = std::rrot(w_im15, u32:7) ^ std::rrot(w_im15, u32:18) ^ (w_im15 >> u32:3);
        let w_im2: u32 = w[i + u32:16 - u32:2];
        let s_1: u32 = std::rrot(w_im2, u32:17) ^ std::rrot(w_im2, u32:19) ^ (w_im2 >> u32:10);
        let value: u32 = w[i + u32:16 - u32:16] + s_0 + w[i + u32:16 - u32:7] + s_1;
        update(w, i + u32:16, value)
    }(w_init);
    w
}

// Evolves the digest for a single chunk in the overall message being
// SHA256-hashed.
fn sha256_chunk(chunk: bits[512], digest_init: Digest) -> Digest {
    let w: u32[64] = sha256_chunk_w_table(chunk);

    // The constant "K" table of addends.
    const K = u32[64]:[
        u32:0x428a2f98, u32:0x71374491, u32:0xb5c0fbcf, u32:0xe9b5dba5, u32:0x3956c25b,
        u32:0x59f111f1, u32:0x923f82a4, u32:0xab1c5ed5, u32:0xd807aa98, u32:0x12835b01,
        u32:0x243185be, u32:0x550c7dc3, u32:0x72be5d74, u32:0x80deb1fe, u32:0x9bdc06a7,
        u32:0xc19bf174, u32:0xe49b69c1, u32:0xefbe4786, u32:0x0fc19dc6, u32:0x240ca1cc,
        u32:0x2de92c6f, u32:0x4a7484aa, u32:0x5cb0a9dc, u32:0x76f988da, u32:0x983e5152,
        u32:0xa831c66d, u32:0xb00327c8, u32:0xbf597fc7, u32:0xc6e00bf3, u32:0xd5a79147,
        u32:0x06ca6351, u32:0x14292967, u32:0x27b70a85, u32:0x2e1b2138, u32:0x4d2c6dfc,
        u32:0x53380d13, u32:0x650a7354, u32:0x766a0abb, u32:0x81c2c92e, u32:0x92722c85,
        u32:0xa2bfe8a1, u32:0xa81a664b, u32:0xc24b8b70, u32:0xc76c51a3, u32:0xd192e819,
        u32:0xd6990624, u32:0xf40e3585, u32:0x106aa070, u32:0x19a4c116, u32:0x1e376c08,
        u32:0x2748774c, u32:0x34b0bcb5, u32:0x391c0cb3, u32:0x4ed8aa4a, u32:0x5b9cca4f,
        u32:0x682e6ff3, u32:0x748f82ee, u32:0x78a5636f, u32:0x84c87814, u32:0x8cc70208,
        u32:0x90befffa, u32:0xa4506ceb, u32:0xbef9a3f7, u32:0xc67178f2,
    ];

    // Compute the digest using the "w" table over 64 "rounds".
    let (a, b, c, d, e, f, g, h): Digest = for (i, (a, b, c, d, e, f, g, h)): (u32, Digest) in
        range(u32:0, u32:64) {
        let S1 = std::rrot(e, u32:6) ^ std::rrot(e, u32:11) ^ std::rrot(e, u32:25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = h + S1 + ch + K[i] + w[i];
        let S0 = std::rrot(a, u32:2) ^ std::rrot(a, u32:13) ^ std::rrot(a, u32:22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = S0 + maj;
        let (h, g, f) = (g, f, e);
        let e = d + temp1;
        let (d, c, b) = (c, b, a);
        let a = temp1 + temp2;
        (a, b, c, d, e, f, g, h)
    }(digest_init);

    // The new digest mixes together values from the original digest with the
    // derived values (a, b, c, ...) we've computed.
    let (h0, h1, h2, h3, h4, h5, h6, h7): Digest = digest_init;
    (h0 + a, h1 + b, h2 + c, h3 + d, h4 + e, h5 + f, h6 + g, h7 + h)
}

// Returns the number of bits required to add on to turn bit_count into a
// multiple of 512.
fn compute_pad_bits(bit_count: u32) -> u32 {
    std::round_up_to_nearest(bit_count, u32:512) - bit_count
}

#[test]
fn compute_pad_bits_test() {
    assert_eq(u32:511, compute_pad_bits(u32:1));
    assert_eq(u32:1, compute_pad_bits(u32:511));
    assert_eq(u32:0, compute_pad_bits(u32:512));
    assert_eq(u32:511, compute_pad_bits(u32:513));
    assert_eq(u32:0, compute_pad_bits(u32:1024));
}

// The SHA algorithm tells us to precondition our input by tacking on a
// trailing "stop" bit, padding out with zeros, and appending the length as a
// 64-bit quantity such that the resulting number of bits is a multiple of 512.
fn pad_to_512b_chunk<I: u32, P: u32 = {compute_pad_bits(I + u32:65)}, R: u32 = {I + u32:65 + P}>
    (x: bits[I]) -> bits[R] {
    let stop_bit: bits[1] = bits[1]:1;
    x ++ stop_bit ++ bits[P]:0 ++ I as bits[64]
}

pub fn sha256(message: bits[512]) -> Digest {
    let digest_init: Digest = (
        u32:0x6a09e667, u32:0xbb67ae85, u32:0x3c6ef372, u32:0xa54ff53a, u32:0x510e527f,
        u32:0x9b05688c, u32:0x1f83d9ab, u32:0x5be0cd19,
    );

    // TODO(leary): 2019-03-19 Commenting this out for now to avoid needing a
    // 'structural' for loop in IR conversion.
    //
    //for (chunk, digest): (bits[512], Digest) in message {
    //  let new_digest: Digest = sha256_chunk(chunk, digest);
    //  new_digest
    //}(digest_init)
    sha256_chunk(message, digest_init)
}

pub fn main(message: bits[512]) -> Digest { sha256(message) }

#[test]
fn sha256_empty_payload_test() {
    let chunk: bits[512] = u1:0b1 ++ bits[511]:0;
    let digest: Digest = sha256(chunk);
    assert_eq(
        (
            u32:0xe3b0c442, u32:0x98fc1c14, u32:0x9afbf4c8, u32:0x996fb924, u32:0x27ae41e4,
            u32:0x649b934c, u32:0xa495991b, u32:0x7852b855,
        ), digest)
}

#[test]
fn sha256_abc_test() {
    let message = u8[3]:['a', 'b', 'c'];
    let chunk = pad_to_512b_chunk(message as u24);
    let digest: Digest = sha256(chunk);
    assert_eq(
        (
            u32:0xba7816bf, u32:0x8f01cfea, u32:0x414140de, u32:0x5dae2223, u32:0xb00361a3,
            u32:0x96177a9c, u32:0xb410ff61, u32:0xf20015ad,
        ), digest)
}
