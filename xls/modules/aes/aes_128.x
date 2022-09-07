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
import xls.modules.aes.aes_common
import xls.modules.aes.constants

const NUM_ROUNDS = u32:10;

type Block = aes_common::Block;
type KeyWord = aes_common::KeyWord;
type Key = aes_128_common::Key;
type KeySchedule = aes_common::RoundKey[NUM_ROUNDS + u32:1];
type RoundKey = aes_common::RoundKey;

// Creates the set of keys used for each [of the 9] rounds of encryption.
pub fn create_key_schedule(key : Key) -> KeySchedule {
    let sched = KeySchedule:[key, ... ];
    let (sched, _) = for (round, (sched, last_word)) : (u32, (KeySchedule, KeyWord))
            in range(u32:1, NUM_ROUNDS + u32:1) {
        let word0 = sched[round - u32:1][0] ^
            aes_common::sub_word(aes_common::rot_word(last_word)) ^
            constants::R_CON[round - 1];
        let word1 = sched[round - u32:1][1] ^ word0;
        let word2 = sched[round - u32:1][2] ^ word1;
        let word3 = sched[round - u32:1][3] ^ word2;
        let sched = update(sched, round, RoundKey:[word0, word1, word2, word3]);
        (sched, word3)
    }((sched, sched[0][3]));
    sched
}

// Verifies we produce a correct schedule for a given input key.
// Verification values were generated from from the BoringSSL implementation,
// commit efd09b7e.
#![test]
fn test_key_schedule() {
    let key = Key:[
        u8:0x66 ++ u8:0x65 ++ u8:0x64 ++ u8:0x63,
        u8:0x62 ++ u8:0x61 ++ u8:0x39 ++ u8:0x38,
        u8:0x37 ++ u8:0x36 ++ u8:0x35 ++ u8:0x34,
        u8:0x33 ++ u8:0x32 ++ u8:0x31 ++ u8:0x30,
    ];
    let sched = create_key_schedule(key);
    let _ = assert_eq(sched[0], key);
    let _ = assert_eq(
        sched[1],
        Key:[
            u8:0x44 ++ u8:0xa2 ++ u8:0x60 ++ u8:0xa0,
            u8:0x26 ++ u8:0xc3 ++ u8:0x59 ++ u8:0x98,
            u8:0x11 ++ u8:0xf5 ++ u8:0x6c ++ u8:0xac,
            u8:0x22 ++ u8:0xc7 ++ u8:0x5d ++ u8:0x9c]);
    let _ = assert_eq(
        sched[2],
        Key:[
            u8:0x80 ++ u8:0xee ++ u8:0xbe ++ u8:0x33,
            u8:0xa6 ++ u8:0x2d ++ u8:0xe7 ++ u8:0xab,
            u8:0xb7 ++ u8:0xd8 ++ u8:0x8b ++ u8:0x07,
            u8:0x95 ++ u8:0x1f ++ u8:0xd6 ++ u8:0x9b]);
    // Since round key N depends on round key N-1, we'll just jump ahead to
    // the last one.
    let _ = assert_eq(
        sched[10],
        Key:[
            u8:0x71 ++ u8:0x21 ++ u8:0x06 ++ u8:0xf1,
            u8:0x59 ++ u8:0xd1 ++ u8:0x69 ++ u8:0x25,
            u8:0x03 ++ u8:0x21 ++ u8:0xe7 ++ u8:0xaa,
            u8:0xb9 ++ u8:0xae ++ u8:0x62 ++ u8:0xe0]);
    ()
}

// Performs AES encryption of the given block.
pub fn aes_encrypt(key: Key, block: Block) -> Block {
    let round_keys = create_key_schedule(key);
    let block = aes_common::add_round_key(block, round_keys[0]);

    let block = for (i, block): (u32, Block) in range(u32:1, u32:10) {
        let block = aes_common::sub_bytes(block);
        let block = aes_common::shift_rows(block);
        let block = aes_common::mix_columns(block);
        let block = aes_common::add_round_key(block, round_keys[i]);
        block
    }(block);
    let block = aes_common::sub_bytes(block);
    let block = aes_common::shift_rows(block);
    let block = aes_common::add_round_key(block, round_keys[10]);
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

pub fn aes_decrypt(key: Key, block: Block) -> Block {
    let round_keys = create_key_schedule(key);

    let block = aes_common::add_round_key(block, round_keys[NUM_ROUNDS]);
    let block = aes_common::inv_shift_rows(block);
    let block = aes_common::inv_sub_bytes(block);

    let block = for (i, block): (u32, Block) in range(u32:1, NUM_ROUNDS) {
        let block = aes_common::add_round_key(block, round_keys[NUM_ROUNDS - i]);
        let block = aes_common::inv_mix_columns(block);
        let block = aes_common::inv_shift_rows(block);
        let block = aes_common::inv_sub_bytes(block);
        block
    }(block);

    let block = aes_common::add_round_key(block, round_keys[0]);
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
