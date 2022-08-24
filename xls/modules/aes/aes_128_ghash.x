// Copyright 2022 The XLS Authors
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
//
// XLS implementation of the GHASH subroutine of AES-GCM.
import xls.modules.aes.aes_128
import xls.modules.aes.aes_128_common

type Block = aes_128_common::Block;
type Key = aes_128_common::Key;

const ZERO_BLOCK = Block:[
    u32:0 as u8[4],
    u32:0 as u8[4],
    u32:0 as u8[4],
    u32:0 as u8[4],
];

// Simply linearizes a block of data.
fn block_to_u128(x: Block) -> uN[128] {
    let x = x[0][0] ++ x[0][1] ++ x[0][2] ++ x[0][3] ++
            x[1][0] ++ x[1][1] ++ x[1][2] ++ x[1][3] ++
            x[2][0] ++ x[2][1] ++ x[2][2] ++ x[2][3] ++
            x[3][0] ++ x[3][1] ++ x[3][2] ++ x[3][3];
    x
}

// Blockifies a uN[128].
fn u128_to_block(x: uN[128]) -> Block {
    let x = x as u8[16];
    Block:[
        [x[0], x[1], x[2], x[3]],
        [x[4], x[5], x[6], x[7]],
        [x[8], x[9], x[10], x[11]],
        [x[12], x[13], x[14], x[15]],
    ]
}

// Computes the multiplication of x and y under GF(128) with the GCM-specific
// modulus u128:0xe10...0 (or in the field defined by x^128 + x^7 + x^2 + x + 1).
// A better implementation would use a pre-programmed lookup table for product
// components, e.g., a 4-bit lookup table for each 4-bit chunk of A * B, where B
// is a pre-set "hash key", referred to as "H" in the literature.
fn gf128_mul(x: Block, y: Block) -> Block {
    let x = block_to_u128(x);
    let y = block_to_u128(y);

    let r = u8:0b11100001 ++ uN[120]:0;
    // TODO(rspringer): Can't currently select an element from an array or
    // tuple resulting from a for loop.
    let z_v = for (i, (last_z, last_v)) in range(u32:0, u32:128) {
        let z = if (x >> (u32:127 - i)) as u1 == u1:0 { last_z } else { last_z ^ last_v };
        let v = if last_v[0:1] == u1:0 { last_v >> 1 } else { (last_v >> 1) ^ r };
        (z, v)
    }((uN[128]:0, y));

    u128_to_block(z_v.0)
}

#![test]
fn gf128_mul_test() {
    // Test vectors were constructed by evaluation against a reference.
    // Mul by zero.
    let a = Block:[
        [u8:0x03, u8:0x02, u8:0x01, u8:0x00],
        [u8:0x07, u8:0x06, u8:0x05, u8:0x04],
        [u8:0x0b, u8:0x0a, u8:0x09, u8:0x08],
        [u8:0x0f, u8:0x0e, u8:0x0d, u8:0x0c],
    ];
    let b = Block:[
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
    ];
    let _ = assert_eq(gf128_mul(a, b), b);

    // By the identity.
    let key = Key:[u32:0, u32:0, u32:0, u32:0];
    let a = aes_128::aes_encrypt(key, ZERO_BLOCK);
    let b = Block:[
        [u8:0x80, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
    ];
    let z = gf128_mul(a, b);
    let _ = assert_eq(z, a);

    // By two.
    let key = Key:[u32:0, u32:0, u32:0, u32:0];
    let a = aes_128::aes_encrypt(key, ZERO_BLOCK);
    let a = Block:[
        [u8:0x66, u8:0xe9, u8:0x4b, u8:0xd4],
        [u8:0xef, u8:0x8a, u8:0x2c, u8:0x3b],
        [u8:0x88, u8:0x4c, u8:0xfa, u8:0x59],
        [u8:0xca, u8:0x34, u8:0x2b, u8:0x2e],
    ];
    let b = Block:[
        [u8:0x40, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
    ];
    let expected = Block:[
        [u8:0x33, u8:0x74, u8:0xa5, u8:0xea],
        [u8:0x77, u8:0xc5, u8:0x16, u8:0x1d],
        [u8:0xc4, u8:0x26, u8:0x7d, u8:0x2c],
        [u8:0xe5, u8:0x1a, u8:0x15, u8:0x97],
    ];
    let z = gf128_mul(a, b);
    let _ = assert_eq(z, expected);

    // Verify commutativity.
    let z = gf128_mul(b, a);
    let _ = assert_eq(z, expected);

    // Mul of complicated values.
    let b = Block:[
        [u8:0x80, u8:0x55, u8:0xaa, u8:0x55],
        [u8:0x55, u8:0xaa, u8:0x55, u8:0xaa],
        [u8:0xa5, u8:0x5a, u8:0xa5, u8:0x5a],
        [u8:0x5a, u8:0xa5, u8:0x5a, u8:0x01],
    ];
    let expected = Block:[
        [u8:0x3b, u8:0xfe, u8:0x59, u8:0x12],
        [u8:0xce, u8:0xdc, u8:0xff, u8:0x05],
        [u8:0x88, u8:0x0f, u8:0xd7, u8:0x14],
        [u8:0xcd, u8:0x72, u8:0xa1, u8:0x93],
    ];
    let z = gf128_mul(a, b);
    let _ = assert_eq(z, expected);

    // Verify commutativity _harder_.
    let z = gf128_mul(b, a);
    assert_eq(z, expected)
}

