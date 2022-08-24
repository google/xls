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
import xls.modules.aes.aes_128_common
import xls.modules.aes.constants

type KeyWord = aes_128_common::KeyWord;
type Key = aes_128_common::Key;
type KeySchedule = aes_128_common::KeySchedule;
type Block = aes_128_common::Block;

// Produces a key whose bytes are in the same order as the incoming byte stream,
// e.g.,
// "abcd..." (i.e., 0x61, 0x62, 0x63, 0x64...) -> [0x61626364, ...]
pub fn bytes_to_key(bytes: u8[16]) -> Key {
    Key:[bytes[0] ++ bytes[1] ++ bytes[2] ++ bytes[3],
         bytes[4] ++ bytes[5] ++ bytes[6] ++ bytes[7],
         bytes[8] ++ bytes[9] ++ bytes[10] ++ bytes[11],
         bytes[12] ++ bytes[13] ++ bytes[14] ++ bytes[15]]
}

fn rot_word(word: KeyWord) -> KeyWord {
    let bytes = word as u8[4];
    let bytes = u8[4]:[bytes[1], bytes[2], bytes[3], bytes[0]];
    bytes as KeyWord
}

fn sub_word(word: KeyWord) -> KeyWord {
    let bytes = word as u8[4];
    let bytes = u8[4]:[constants::S_BOX[bytes[0]], constants::S_BOX[bytes[1]],
                       constants::S_BOX[bytes[2]], constants::S_BOX[bytes[3]]];
    bytes as KeyWord
}

// Creates the set of keys used for each [of the 9] rounds of encryption.
pub fn create_key_schedule(key : Key) -> KeySchedule {
    let sched = KeySchedule:[key, ... ];
    let (sched, _) = for (round, (sched, last_word)) : (u32, (KeySchedule, KeyWord))
            in range(u32:1, aes_128_common::NUM_ROUNDS + u32:1) {
        let word0 = sched[round - u32:1][0] ^ sub_word(rot_word(last_word)) ^ constants::R_CON[round - 1];
        let word1 = sched[round - u32:1][1] ^ word0;
        let word2 = sched[round - u32:1][2] ^ word1;
        let word3 = sched[round - u32:1][3] ^ word2;
        let sched = update(sched, round, Key:[word0, word1, word2, word3]);
        (sched, word3)
    }((sched, sched[0][3]));
    sched
}

// Verifies we produce a correct schedule for a given input key.
// Verification values were generated from from the BoringSSL implementation,
// commit efd09b7e.
#![test]
fn test_key_schedule() {
    let key =
        bytes_to_key(
            u8[16]:[u8:0x66, u8:0x65, u8:0x64, u8:0x63, u8:0x62, u8:0x61, u8:0x39, u8:0x38,
                    u8:0x37, u8:0x36, u8:0x35, u8:0x34, u8:0x33, u8:0x32, u8:0x31, u8:0x30]);
    let sched = create_key_schedule(key);
    let _ = assert_eq(sched[0], key);
    let _ = assert_eq(
        sched[1],
        bytes_to_key(
            u8[16]:[u8:0x44, u8:0xa2, u8:0x60, u8:0xa0, u8:0x26, u8:0xc3, u8:0x59, u8:0x98,
                    u8:0x11, u8:0xf5, u8:0x6c, u8:0xac, u8:0x22, u8:0xc7, u8:0x5d, u8:0x9c]));
    let _ = assert_eq(
        sched[2],
        bytes_to_key(
            u8[16]:[u8:0x80, u8:0xee, u8:0xbe, u8:0x33, u8:0xa6, u8:0x2d, u8:0xe7, u8:0xab,
                    u8:0xb7, u8:0xd8, u8:0x8b, u8:0x07, u8:0x95, u8:0x1f, u8:0xd6, u8:0x9b]));
    // Since round key N depends on round key N-1, we'll just jump ahead to
    // the last one.
    let _ = assert_eq(
        sched[10],
        bytes_to_key(
            u8[16]:[u8:0x71, u8:0x21, u8:0x06, u8:0xf1, u8:0x59, u8:0xd1, u8:0x69, u8:0x25,
                    u8:0x03, u8:0x21, u8:0xe7, u8:0xaa, u8:0xb9, u8:0xae, u8:0x62, u8:0xe0]));
    ()
}

pub fn add_round_key(block: Block, key: Key) -> Block {
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

#![test]
fn test_add_round_key() {
    let key = Key: [
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

#![test]
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

#![test]
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
#![test]
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

// Performs AES encryption of the given block. Features such as GCM or CBC mode
// additions would be performed outside this core routine itself.
pub fn aes_encrypt(key: Key, block: Block) -> Block {
    let round_keys = create_key_schedule(key);
    let block = add_round_key(block, round_keys[0]);

    let block = for (i, block): (u32, Block) in range(u32:1, u32:10) {
        let block = sub_bytes(block);
        let block = shift_rows(block);
        let block = mix_columns(block);
        let block = add_round_key(block, round_keys[i]);
        block
    }(block);
    let block = sub_bytes(block);
    let block = shift_rows(block);
    let block = add_round_key(block, round_keys[10]);
    block
}

#![test]
fn test_aes_encrypt() {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0x4, u8:0x5, u8:0x6, u8:0x7],
        [u8:0x8, u8:0x9, u8:0xa, u8:0xb],
        [u8:0xc, u8:0xd, u8:0xe, u8:0xf],
    ];
    let key = Key:[
        u8:0x0 ++ u8:0x1 ++ u8:0x2 ++ u8:0x3,
        u8:0x4 ++ u8:0x5 ++ u8:0x6 ++ u8:0x7,
        u8:0x8 ++ u8:0x9 ++ u8:0xa ++ u8:0xb,
        u8:0xc ++ u8:0xd ++ u8:0xe ++ u8:0xf,
    ];
    let expected = Block:[
        u8[4]:[u8:0x0a, u8:0x94, u8:0x0b, u8:0xb5],
        u8[4]:[u8:0x41, u8:0x6e, u8:0xf0, u8:0x45],
        u8[4]:[u8:0xf1, u8:0xc3, u8:0x94, u8:0x58],
        u8[4]:[u8:0xc6, u8:0x53, u8:0xea, u8:0x5a],
    ];
    let actual = aes_encrypt(key, input);
    assert_eq(actual, expected)
}

fn inv_sub_bytes(block: Block) -> Block {
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

#![test]
fn test_inv_sub_bytes(block: Block) {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0xff, u8:0xfe, u8:0xfd, u8:0xfc],
        [u8:0xa5, u8:0x5a, u8:0xaa, u8:0x55],
        [u8:0xde, u8:0xad, u8:0xbe, u8:0xef],
    ];
    assert_eq(inv_sub_bytes(sub_bytes(input)), input)
}

fn inv_shift_rows(block: Block) -> Block {
    Block:[
        u8[4]:[block[0][0], block[3][1], block[2][2], block[1][3]],
        u8[4]:[block[1][0], block[0][1], block[3][2], block[2][3]],
        u8[4]:[block[2][0], block[1][1], block[0][2], block[3][3]],
        u8[4]:[block[3][0], block[2][1], block[1][2], block[0][3]],
    ]
}

#![test]
fn test_inv_shift_rows(block: Block) {
    let input = Block:[
        [u8:0x0, u8:0x1, u8:0x2, u8:0x3],
        [u8:0x4, u8:0x5, u8:0x6, u8:0x7],
        [u8:0x8, u8:0x9, u8:0xa, u8:0xb],
        [u8:0xc, u8:0xd, u8:0xe, u8:0xf],
    ];
    assert_eq(inv_shift_rows(shift_rows(input)), input)
}

fn inv_mix_column(col: u8[4]) -> u8[4]{
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

#![test]
fn test_inv_mix_columns() {
    let input = Block:[
        [u8:0xdb, u8:0xf2, u8:0xc6, u8:0x2d],
        [u8:0x13, u8:0x0a, u8:0xc6, u8:0x26],
        [u8:0x53, u8:0x22, u8:0xc6, u8:0x31],
        [u8:0x45, u8:0x5c, u8:0xc6, u8:0x4c],
    ];
    assert_eq(inv_mix_columns(mix_columns(input)), input)
}

pub fn aes_decrypt(key: Key, block: Block) -> Block {
    let round_keys = create_key_schedule(key);

    let block = add_round_key(block, round_keys[10]);
    let block = inv_shift_rows(block);
    let block = inv_sub_bytes(block);

    let block = for (i, block): (u32, Block) in range(u32:1, u32:10) {
        let block = add_round_key(block, round_keys[u32:10 - i]);
        let block = inv_mix_columns(block);
        let block = inv_shift_rows(block);
        let block = inv_sub_bytes(block);
        block
    }(block);

    let block = add_round_key(block, round_keys[0]);
    block
}

#![test]
fn test_aes_decrypt() {
    let input = Block:[
        [u8:0xb6, u8:0x7f, u8:0x5e, u8:0x7f],
        [u8:0x22, u8:0x7c, u8:0xa2, u8:0xfc],
        [u8:0xf8, u8:0x80, u8:0x99, u8:0xba],
        [u8:0x1f, u8:0x14, u8:0x68, u8:0x29],
    ];
    let key = Key:[
        u8:0x50 ++ u8:0xb7 ++ u8:0x1d ++ u8:0x6e,
        u8:0x7f ++ u8:0x04 ++ u8:0x59 ++ u8:0x23,
        u8:0x5b ++ u8:0xc2 ++ u8:0x7d ++ u8:0x93,
        u8:0x0a ++ u8:0x30 ++ u8:0x9e ++ u8:0xa8,
    ];
    let expected = Block:[
        u8[4]:[u8:0x0a, u8:0x94, u8:0x0b, u8:0xb5],
        u8[4]:[u8:0x41, u8:0x6e, u8:0xf0, u8:0x45],
        u8[4]:[u8:0xf1, u8:0xc3, u8:0x94, u8:0x58],
        u8[4]:[u8:0xc6, u8:0x53, u8:0xea, u8:0x5a],
    ];
    assert_eq(aes_decrypt(key, aes_encrypt(key, input)), input)
}
