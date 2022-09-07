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
import xls.modules.aes.aes_256_common
import xls.modules.aes.aes_common
import xls.modules.aes.constants

const KEY_WORDS = aes_256_common::KEY_WORDS;
const NUM_ROUNDS = u32:14;

type Block = aes_common::Block;
type Key = aes_256_common::Key;
type KeySchedule = aes_common::RoundKey[NUM_ROUNDS + u32:1];
type RoundKey = aes_common::RoundKey;

pub fn create_key_schedule(key: Key) -> KeySchedule {
    // With a 256 bit key, the first two round keys are the two halves of the
    // input key.
    let sched = KeySchedule:[
        [key[0], key[1], key[2], key[3]],
        [key[4], key[5], key[6], key[7]],
        [u32:0, u32:0, u32:0, u32:0],
        ...
    ];

    let last_word = key[KEY_WORDS - u32:1];
    let (sched, _) = for (round, (sched, last_word)) in range(u32:2, NUM_ROUNDS + u32:1) {
        let word0 = if round & u32:1 == u32:0 {
            sched[round - u32:2][0] ^
                aes_common::sub_word(aes_common::rot_word(last_word)) ^
                constants::R_CON[(round / u32:2) - u32:1]
        } else {
            sched[round - u32:2][0] ^ aes_common::sub_word(last_word)
        };
        let word1 = sched[round - u32:2][1] ^ word0;
        let word2 = sched[round - u32:2][2] ^ word1;
        let word3 = sched[round - u32:2][3] ^ word2;
        let sched = update(sched, round, RoundKey:[word0, word1, word2, word3]);
        (sched, word3)
    }((sched, last_word));

    sched
}

#![test]
fn test_create_key_schedule() {
    let key = Key:[
        u8:0xac ++ u8:0x75 ++ u8:0x4a ++ u8:0x55,
        u8:0x99 ++ u8:0x93 ++ u8:0x7e ++ u8:0x79,
        u8:0x9e ++ u8:0x7c ++ u8:0x46 ++ u8:0x97,
        u8:0xb1 ++ u8:0xdd ++ u8:0x57 ++ u8:0x14,
        u8:0xeb ++ u8:0x11 ++ u8:0xda ++ u8:0x40,
        u8:0xb1 ++ u8:0x8c ++ u8:0xa8 ++ u8:0x29,
        u8:0x15 ++ u8:0xaa ++ u8:0x4a ++ u8:0xd5,
        u8:0xde ++ u8:0x3c ++ u8:0x83 ++ u8:0x35,
    ];
    let sched = create_key_schedule(key);
    let expected_0 = RoundKey:[
        u8:0xac ++ u8:0x75 ++ u8:0x4a ++ u8:0x55,
        u8:0x99 ++ u8:0x93 ++ u8:0x7e ++ u8:0x79,
        u8:0x9e ++ u8:0x7c ++ u8:0x46 ++ u8:0x97,
        u8:0xb1 ++ u8:0xdd ++ u8:0x57 ++ u8:0x14,
    ];
    let _ = assert_eq(expected_0, sched[0]);

    let expected_1 = RoundKey:[
        u8:0xeb ++ u8:0x11 ++ u8:0xda ++ u8:0x40,
        u8:0xb1 ++ u8:0x8c ++ u8:0xa8 ++ u8:0x29,
        u8:0x15 ++ u8:0xaa ++ u8:0x4a ++ u8:0xd5,
        u8:0xde ++ u8:0x3c ++ u8:0x83 ++ u8:0x35,
    ];
    let _ = assert_eq(expected_1, sched[1]);

    let expected_2 = RoundKey:[
        u8:0x46 ++ u8:0x99 ++ u8:0xdc ++ u8:0x48,
        u8:0xdf ++ u8:0x0a ++ u8:0xa2 ++ u8:0x31,
        u8:0x41 ++ u8:0x76 ++ u8:0xe4 ++ u8:0xa6,
        u8:0xf0 ++ u8:0xab ++ u8:0xb3 ++ u8:0xb2,
    ];
    let _ = assert_eq(expected_2, sched[2]);

    let expected_3 = RoundKey:[
        u8:0x67 ++ u8:0x73 ++ u8:0xb7 ++ u8:0x77,
        u8:0xd6 ++ u8:0xff ++ u8:0x1f ++ u8:0x5e,
        u8:0xc3 ++ u8:0x55 ++ u8:0x55 ++ u8:0x8b,
        u8:0x1d ++ u8:0x69 ++ u8:0xd6 ++ u8:0xbe,
    ];
    let _ = assert_eq(expected_3, sched[3]);

    let expected_4 = RoundKey:[
        u8:0xbd ++ u8:0x6f ++ u8:0x72 ++ u8:0xec,
        u8:0x62 ++ u8:0x65 ++ u8:0xd0 ++ u8:0xdd,
        u8:0x23 ++ u8:0x13 ++ u8:0x34 ++ u8:0x7b,
        u8:0xd3 ++ u8:0xb8 ++ u8:0x87 ++ u8:0xc9,
    ];
    let _ = assert_eq(expected_4, sched[4]);

    // ...and jump to the last, since results build on each other.
    let expected_14 = RoundKey:[
        u8:0xb1 ++ u8:0xb3 ++ u8:0xe8 ++ u8:0x0b,
        u8:0x25 ++ u8:0xfc ++ u8:0x6d ++ u8:0x57,
        u8:0xe0 ++ u8:0xe6 ++ u8:0x9d ++ u8:0xe7,
        u8:0xc4 ++ u8:0x70 ++ u8:0x42 ++ u8:0xf7,
    ];
    let _ = assert_eq(expected_14, sched[14]);
    ()
}

pub fn encrypt(key: Key, block: Block) -> Block {
    let round_keys = create_key_schedule(key);
    let block = aes_common::add_round_key(block, round_keys[0]);

    let block = for (i, block): (u32, Block) in range(u32:1, NUM_ROUNDS) {
        let block = aes_common::sub_bytes(block);
        let block = aes_common::shift_rows(block);
        let block = aes_common::mix_columns(block);
        let block = aes_common::add_round_key(block, round_keys[i]);
        block
    }(block);
    let block = aes_common::sub_bytes(block);
    let block = aes_common::shift_rows(block);
    let block = aes_common::add_round_key(block, round_keys[NUM_ROUNDS]);
    block
}

#![test]
fn test_encrypt() {
    let key = Key:[
        u8:0xcc ++ u8:0x6d ++ u8:0xe8 ++ u8:0x07,
        u8:0xd0 ++ u8:0x19 ++ u8:0x80 ++ u8:0xc9,
        u8:0x15 ++ u8:0x96 ++ u8:0xcc ++ u8:0xa8,
        u8:0x77 ++ u8:0x40 ++ u8:0x6b ++ u8:0x95,
        u8:0x8f ++ u8:0xdb ++ u8:0x3f ++ u8:0xe2,
        u8:0xac ++ u8:0x25 ++ u8:0xed ++ u8:0x7d,
        u8:0xa0 ++ u8:0x5b ++ u8:0x44 ++ u8:0x92,
        u8:0x1a ++ u8:0x63 ++ u8:0x15 ++ u8:0x0f,
    ];
    let plaintext = Block:[
        [u8:0x43, u8:0x6b, u8:0x12, u8:0xb4],
        [u8:0x6f, u8:0xfa, u8:0xb4, u8:0x60],
        [u8:0x63, u8:0xc8, u8:0x44, u8:0xba],
        [u8:0x6d, u8:0x1c, u8:0xeb, u8:0xf8],
    ];
    let expected = Block:[
        [u8:0x60, u8:0x06, u8:0xe9, u8:0x6a],
        [u8:0x72, u8:0xd2, u8:0x8a, u8:0xb8],
        [u8:0xdf, u8:0x50, u8:0xcf, u8:0xf6],
        [u8:0x75, u8:0xc9, u8:0x3d, u8:0x94],
    ];
    assert_eq(expected, encrypt(key, plaintext))
}

pub fn decrypt(key: Key, block: Block) -> Block {
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
fn test_decrypt() {
    let key = Key:[
        u8:0xcc ++ u8:0x6d ++ u8:0xe8 ++ u8:0x07,
        u8:0xd0 ++ u8:0x19 ++ u8:0x80 ++ u8:0xc9,
        u8:0x15 ++ u8:0x96 ++ u8:0xcc ++ u8:0xa8,
        u8:0x77 ++ u8:0x40 ++ u8:0x6b ++ u8:0x95,
        u8:0x8f ++ u8:0xdb ++ u8:0x3f ++ u8:0xe2,
        u8:0xac ++ u8:0x25 ++ u8:0xed ++ u8:0x7d,
        u8:0xa0 ++ u8:0x5b ++ u8:0x44 ++ u8:0x92,
        u8:0x1a ++ u8:0x63 ++ u8:0x15 ++ u8:0x0f,
    ];
    let plaintext = Block:[
        [u8:0x43, u8:0x6b, u8:0x12, u8:0xb4],
        [u8:0x6f, u8:0xfa, u8:0xb4, u8:0x60],
        [u8:0x63, u8:0xc8, u8:0x44, u8:0xba],
        [u8:0x6d, u8:0x1c, u8:0xeb, u8:0xf8],
    ];
    assert_eq(plaintext, decrypt(key, encrypt(key, plaintext)))
}
