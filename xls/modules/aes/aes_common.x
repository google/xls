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

// Implementation of the AES-128 cipher.
//
// Note that throughout, "row" and "column" seem to be swapped: that's because
// DSLX is a row-major language, whereas AES is described in a column-major
// manner.
import xls.modules.aes.constants;

pub const MAX_KEY_BITS = u32:256;
pub const MAX_KEY_BYTES = MAX_KEY_BITS >> 3;
pub const MAX_KEY_WORDS = MAX_KEY_BYTES >> 2;

pub const KEY_WORD_BITS = u32:32;
pub const BLOCK_BITS = u32:128;
pub const BLOCK_BYTES = BLOCK_BITS >> 3;

pub type Block = u8[4][4];
pub type InitVector = uN[96];
pub type Key = u8[MAX_KEY_BYTES];
pub type KeyWord = uN[KEY_WORD_BITS];
pub type RoundKey = u32[4];

// 192 bit keys aren't currently supported, due to the complexity of the
// mod operation (needed in create_key_schedule).
pub enum KeyWidth : u2 {
    KEY_128 = 0,
    KEY_256 = 2,
}

pub const ZERO_KEY = Key:[u8:0, ...];

pub const ZERO_BLOCK = Block:[
    u32:0 as u8[4],
    u32:0 as u8[4],
    u32:0 as u8[4],
    u32:0 as u8[4],
];

pub fn rot_word(word: KeyWord) -> KeyWord {
    let bytes = word as u8[4];
    let bytes = u8[4]:[bytes[1], bytes[2], bytes[3], bytes[0]];
    bytes as KeyWord
}

pub fn sub_word(word: KeyWord) -> KeyWord {
    let bytes = word as u8[4];
    let bytes = u8[4]:[constants::S_BOX[bytes[0]], constants::S_BOX[bytes[1]],
                       constants::S_BOX[bytes[2]], constants::S_BOX[bytes[3]]];
    bytes as KeyWord
}

pub fn add_round_key(block: Block, key: RoundKey) -> Block {
    let key_0 = key[0] as u8[4];
    let key_1 = key[1] as u8[4];
    let key_2 = key[2] as u8[4];
    let key_3 = key[3] as u8[4];
    Block:[
        [block[0][0] ^ key_0[0], block[0][1] ^ key_0[1],
         block[0][2] ^ key_0[2], block[0][3] ^ key_0[3]],
        [block[1][0] ^ key_1[0], block[1][1] ^ key_1[1],
         block[1][2] ^ key_1[2], block[1][3] ^ key_1[3]],
        [block[2][0] ^ key_2[0], block[2][1] ^ key_2[1],
         block[2][2] ^ key_2[2], block[2][3] ^ key_2[3]],
        [block[3][0] ^ key_3[0], block[3][1] ^ key_3[1],
         block[3][2] ^ key_3[2], block[3][3] ^ key_3[3]],
    ]
}

#[test]
fn test_add_round_key() {
    let key = RoundKey: [
        u8:0x30 ++ u8:0x31 ++ u8:0x32 ++ u8:0x33,
        u8:0x34 ++ u8:0x35 ++ u8:0x36 ++ u8:0x37,
        u8:0x38 ++ u8:0x39 ++ u8:0x61 ++ u8:0x62,
        u8:0x63 ++ u8:0x64 ++ u8:0x65 ++ u8:0x66,
    ];
    let block = Block:[
        [u8:0, u8:1, u8:2, u8:3],
        [u8:0xff, u8:0xaa, u8:0x55, u8:0x00],
        [u8:0xa5, u8:0x5a, u8:0x5a, u8:0xa5],
        [u8:3, u8:2, u8:1, u8:0],
    ];
    let expected = Block:[
        [u8:0x30, u8:0x30, u8:0x30, u8:0x30],
        [u8:0xcb, u8:0x9f, u8:0x63, u8:0x37],
        [u8:0x9d, u8:0x63, u8:0x3b, u8:0xc7],
        [u8:0x60, u8:0x66, u8:0x64, u8:0x66],
    ];
    assert_eq(add_round_key(block, key), expected)
}

// Performs the "SubBytes" step. Replaces each byte of the input with the
// corresponding byte from the S-box.
pub fn sub_bytes(block: Block) -> Block {
    Block:[
        [constants::S_BOX[block[0][0]], constants::S_BOX[block[0][1]],
         constants::S_BOX[block[0][2]], constants::S_BOX[block[0][3]]],
        [constants::S_BOX[block[1][0]], constants::S_BOX[block[1][1]],
         constants::S_BOX[block[1][2]], constants::S_BOX[block[1][3]]],
        [constants::S_BOX[block[2][0]], constants::S_BOX[block[2][1]],
         constants::S_BOX[block[2][2]], constants::S_BOX[block[2][3]]],
        [constants::S_BOX[block[3][0]], constants::S_BOX[block[3][1]],
         constants::S_BOX[block[3][2]], constants::S_BOX[block[3][3]]],
    ]
}

#[test]
fn test_sub_bytes() {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0xff, u8:0xfe, u8:0xfd, u8:0xfc],
        [u8:0xa5, u8:0x5a, u8:0xaa, u8:0x55],
        [u8:0xde, u8:0xad, u8:0xbe, u8:0xef],
    ];
    let expected = Block:[
        [u8:0x63, u8:0x7c, u8:0x77, u8:0x7b],
        [u8:0x16, u8:0xbb, u8:0x54, u8:0xb0],
        [u8:0x06, u8:0xbe, u8:0xac, u8:0xfc],
        [u8:0x1d, u8:0x95, u8:0xae, u8:0xdf],
    ];
    assert_eq(sub_bytes(input), expected)
}

// Performs the "ShiftRows" step. Rotates row N to the left by N spaces.
pub fn shift_rows(block: Block) -> Block {
    Block:[
        u8[4]:[block[0][0], block[1][1], block[2][2], block[3][3]],
        u8[4]:[block[1][0], block[2][1], block[3][2], block[0][3]],
        u8[4]:[block[2][0], block[3][1], block[0][2], block[1][3]],
        u8[4]:[block[3][0], block[0][1], block[1][2], block[2][3]],
    ]
}

#[test]
fn test_shift_rows() {
    let input = Block:[
        [u8:0, u8:1, u8:2, u8:3],
        [u8:4, u8:5, u8:6, u8:7],
        [u8:8, u8:9, u8:10, u8:11],
        [u8:12, u8:13, u8:14, u8:15],
    ];
    let expected = Block:[
        [u8:0, u8:5, u8:10, u8:15],
        [u8:4, u8:9, u8:14, u8:3],
        [u8:8, u8:13, u8:2, u8:7],
        [u8:12, u8:1, u8:6, u8:11],
    ];
    assert_eq(shift_rows(input), expected)
}

// Performs multiplication of the input by 2 in GF(2^8). See "MixColumns" below.
fn gfmul2(input: u8) -> u8 {
    let result = input << 1;
    if input & u8:0x80 != u8:0 { result ^ u8:0x1b } else { result }
}

// Performs multiplication of the input by 3 in GF(2^8). See "MixColumns" below.
fn gfmul3(input: u8) -> u8 {
    let result = gfmul2(input);
    result ^ input
}

// Performs multiplication of the input by 9 in GF(2^8). See "MixColumns" below.
// TODO(rspringer): Consider implementing the math instead of a lookup; not sure
// what the computation vs. area cost would be.
fn gfmul9(input: u8) -> u8 {
    constants::GF_MUL_9_TBL[input]
}

// Performs multiplication of the input by 11 in GF(2^8). See "MixColumns" below.
// TODO(rspringer): Consider implementing the math instead of a lookup; not sure
// what the computation vs. area cost would be.
fn gfmul11(input: u8) -> u8 {
    constants::GF_MUL_11_TBL[input]
}

// Performs multiplication of the input by 13 in GF(2^8). See "MixColumns" below.
// TODO(rspringer): Consider implementing the math instead of a lookup; not sure
// what the computation vs. area cost would be.
fn gfmul13(input: u8) -> u8 {
    constants::GF_MUL_13_TBL[input]
}

// Performs multiplication of the input by 14 in GF(2^8). See "MixColumns" below.
// TODO(rspringer): Consider implementing the math instead of a lookup; not sure
// what the computation vs. area cost would be.
fn gfmul14(input: u8) -> u8 {
    constants::GF_MUL_14_TBL[input]
}

// See "MixColumns" below. This implements transformation of one column.
fn mix_column(col: u8[4]) -> u8[4]{
    u8[4]:[
        gfmul2(col[0]) ^ gfmul3(col[1]) ^ col[2] ^ col[3],
        col[0] ^ gfmul2(col[1]) ^ gfmul3(col[2]) ^ col[3],
        col[0] ^ col[1] ^ gfmul2(col[2]) ^ gfmul3(col[3]),
        gfmul3(col[0]) ^ col[1] ^ col[2] ^ gfmul2(col[3]),
    ]
}

// Implements the "MixColumns" step: multiplies a column of the state by the matrix:
// [[2 ,3, 1, 1]
//  [1, 2, 3, 1]
//  [1, 1, 2, 3]
//  [3, 1, 1, 2]]
// within GF(2*8) (the 256-element Galois field). In this field, addition is XOR.
// Multiplication by 1 is the identity.
// Multiplication by 2 is left shifting by one and XORing the result by 0x1b
// if it overflows.
// Multiplication by 3 is multiplying by 2 and XORing that with the identity.
pub fn mix_columns(block: Block) -> Block {
    let col0 = mix_column(u8[4]:[block[0][0], block[0][1], block[0][2], block[0][3]]);
    let col1 = mix_column(u8[4]:[block[1][0], block[1][1], block[1][2], block[1][3]]);
    let col2 = mix_column(u8[4]:[block[2][0], block[2][1], block[2][2], block[2][3]]);
    let col3 = mix_column(u8[4]:[block[3][0], block[3][1], block[3][2], block[3][3]]);
    Block:[
        [col0[0], col0[1], col0[2], col0[3]],
        [col1[0], col1[1], col1[2], col1[3]],
        [col2[0], col2[1], col2[2], col2[3]],
        [col3[0], col3[1], col3[2], col3[3]],
    ]
}

// Verification values were generated from from the BoringSSL implementation,
// commit efd09b7e.
#[test]
fn test_mix_columns() {
    let input = Block:[
        [u8:0xdb, u8:0xf2, u8:0xc6, u8:0x2d],
        [u8:0x13, u8:0x0a, u8:0xc6, u8:0x26],
        [u8:0x53, u8:0x22, u8:0xc6, u8:0x31],
        [u8:0x45, u8:0x5c, u8:0xc6, u8:0x4c],
    ];
    let expected = Block:[
        [u8:0x4b, u8:0x58, u8:0xc9, u8:0x18],
        [u8:0xd8, u8:0x70, u8:0xe4, u8:0xb5],
        [u8:0x37, u8:0x77, u8:0xb5, u8:0x73],
        [u8:0xe4, u8:0xe0, u8:0x5a, u8:0xcd],
    ];
    assert_eq(mix_columns(input), expected)
}

pub fn inv_sub_bytes(block: Block) -> Block {
    Block:[
        [constants::INV_S_BOX[block[0][0]], constants::INV_S_BOX[block[0][1]],
         constants::INV_S_BOX[block[0][2]], constants::INV_S_BOX[block[0][3]]],
        [constants::INV_S_BOX[block[1][0]], constants::INV_S_BOX[block[1][1]],
         constants::INV_S_BOX[block[1][2]], constants::INV_S_BOX[block[1][3]]],
        [constants::INV_S_BOX[block[2][0]], constants::INV_S_BOX[block[2][1]],
         constants::INV_S_BOX[block[2][2]], constants::INV_S_BOX[block[2][3]]],
        [constants::INV_S_BOX[block[3][0]], constants::INV_S_BOX[block[3][1]],
         constants::INV_S_BOX[block[3][2]], constants::INV_S_BOX[block[3][3]]],
    ]
}

#[test]
fn test_inv_sub_bytes() {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0xff, u8:0xfe, u8:0xfd, u8:0xfc],
        [u8:0xa5, u8:0x5a, u8:0xaa, u8:0x55],
        [u8:0xde, u8:0xad, u8:0xbe, u8:0xef],
    ];
    assert_eq(inv_sub_bytes(sub_bytes(input)), input)
}

pub fn inv_shift_rows(block: Block) -> Block {
    Block:[
        u8[4]:[block[0][0], block[3][1], block[2][2], block[1][3]],
        u8[4]:[block[1][0], block[0][1], block[3][2], block[2][3]],
        u8[4]:[block[2][0], block[1][1], block[0][2], block[3][3]],
        u8[4]:[block[3][0], block[2][1], block[1][2], block[0][3]],
    ]
}

#[test]
fn test_inv_shift_rows() {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0x4, u8:0x5, u8:0x6, u8:0x7],
        [u8:0x8, u8:0x9, u8:0xa, u8:0xb],
        [u8:0xc, u8:0xd, u8:0xe, u8:0xf],
    ];
    assert_eq(inv_shift_rows(shift_rows(input)), input)
}

pub fn inv_mix_column(col: u8[4]) -> u8[4]{
    u8[4]:[
        gfmul14(col[0]) ^ gfmul11(col[1]) ^ gfmul13(col[2]) ^ gfmul9(col[3]),
        gfmul9(col[0]) ^ gfmul14(col[1]) ^ gfmul11(col[2]) ^ gfmul13(col[3]),
        gfmul13(col[0]) ^ gfmul9(col[1]) ^ gfmul14(col[2]) ^ gfmul11(col[3]),
        gfmul11(col[0]) ^ gfmul13(col[1]) ^ gfmul9(col[2]) ^ gfmul14(col[3]),
    ]
}

pub fn inv_mix_columns(block: Block) -> Block {
    let col0 = inv_mix_column(u8[4]:[block[0][0], block[0][1], block[0][2], block[0][3]]);
    let col1 = inv_mix_column(u8[4]:[block[1][0], block[1][1], block[1][2], block[1][3]]);
    let col2 = inv_mix_column(u8[4]:[block[2][0], block[2][1], block[2][2], block[2][3]]);
    let col3 = inv_mix_column(u8[4]:[block[3][0], block[3][1], block[3][2], block[3][3]]);
    Block:[
        [col0[0], col0[1], col0[2], col0[3]],
        [col1[0], col1[1], col1[2], col1[3]],
        [col2[0], col2[1], col2[2], col2[3]],
        [col3[0], col3[1], col3[2], col3[3]],
    ]
}

#[test]
fn test_inv_mix_columns() {
    let input = Block:[
        [u8:0xdb, u8:0xf2, u8:0xc6, u8:0x2d],
        [u8:0x13, u8:0x0a, u8:0xc6, u8:0x26],
        [u8:0x53, u8:0x22, u8:0xc6, u8:0x31],
        [u8:0x45, u8:0x5c, u8:0xc6, u8:0x4c],
    ];
    assert_eq(inv_mix_columns(mix_columns(input)), input)
}

// Until GitHub issue #629 is resolved, this MUST NOT be called in AOT-compiled
// code!
pub fn trace_block(block: Block) {
    let bytes0 = block[0];
    let bytes1 = block[1];
    let bytes2 = block[2];
    let bytes3 = block[3];
    trace_fmt!(
        "0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
        bytes0[0], bytes0[1], bytes0[2], bytes0[3],
        bytes1[0], bytes1[1], bytes1[2], bytes1[3],
        bytes2[0], bytes2[1], bytes2[2], bytes2[3],
        bytes3[0], bytes3[1], bytes3[2], bytes3[3]);
}

// Until GitHub issue #629 is resolved, this MUST NOT be called in AOT-compiled
// code!
pub fn trace_key(key: Key) {
    let key = key as uN[256] as u32[8];
    trace_fmt!(
        "0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x} 0x{:x}",
        key[0], key[1], key[2], key[3],
        key[4], key[5], key[6], key[7]);
}

// Convenience function to XOR two blocks.
pub fn xor_block(a: Block, b: Block) -> Block {
    Block:[
        (a[0] as u32 ^ b[0] as u32) as u8[4],
        (a[1] as u32 ^ b[1] as u32) as u8[4],
        (a[2] as u32 ^ b[2] as u32) as u8[4],
        (a[3] as u32 ^ b[3] as u32) as u8[4],
    ]
}
