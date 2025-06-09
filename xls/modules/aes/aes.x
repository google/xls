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
import std;
import xls.modules.aes.aes_common;
import xls.modules.aes.constants;

const MAX_NUM_ROUNDS = u32:14;

type Block = aes_common::Block;
type Key = u8[aes_common::MAX_KEY_BYTES];
type KeyWidth = aes_common::KeyWidth;
type RoundKey = aes_common::RoundKey;
type KeySchedule = aes_common::RoundKey[MAX_NUM_ROUNDS + u32:1];

fn get_num_rounds(key_width: KeyWidth) -> u32 {
    // TODO(rspringer): Static assert goes here.
    if key_width == KeyWidth::KEY_128 { u32:10 } else { u32:14 }
}

pub fn create_key_schedule(key: u8[aes_common::MAX_KEY_BYTES], key_width: KeyWidth)
    -> aes_common::RoundKey[MAX_NUM_ROUNDS + u32:1] {
    const NUM_SCHED_WORDS = u32:4 * (MAX_NUM_ROUNDS + u32:1);

    let key_words = if key_width == KeyWidth::KEY_128 { u32:4 } else { u32:8 };
    let round_shift = if key_width == KeyWidth::KEY_128 { u32:2 } else { u32:3 };
    let sched_words = (get_num_rounds(key_width) + u32:1) << 2;
    let sched = u32[NUM_SCHED_WORDS]:[u32:0, ...];

    let key = key as uN[aes_common::MAX_KEY_BITS] as u32[aes_common::MAX_KEY_WORDS];
    let (sched, _) = for (word_idx, (sched, last_word)) in u32:0..NUM_SCHED_WORDS {
        let a = word_idx < key_words;
        let b = word_idx >= key_words && std::mod_pow2(word_idx, key_words) == u32:0;
        // Condition "C" is actually:
        //  word_idx > key_words && key_words > u32:6 && word_idx % key_words == 4,
        // but since we don't support 192-bit keys, we can force-mod2 the modulo.
        let c = word_idx >= key_words && key_words > u32:6 && std::mod_pow2(word_idx, key_words) == u32:4;

        // Only one condition can be true, but this is a lot cleaner than an
        // if/else chain.
        let r_con_idx = (word_idx >> round_shift) - u32:1;
        let r_con = if r_con_idx < u32:10 { constants::R_CON[r_con_idx] } else { u32:0 };
        let word = match(a, b, c) {
            (true, _, _) => key[word_idx],
            (_, true, _) => sched[word_idx - key_words] ^
                aes_common::sub_word(aes_common::rot_word(last_word)) ^ r_con,
            (_, _, true) => sched[word_idx - key_words] ^ aes_common::sub_word(last_word),
            _            => sched[word_idx - key_words] ^ last_word,
        };

        let sched = if word_idx > sched_words {
            sched
        } else {
            update(sched, word_idx, word)
        };
        (sched, word)
    }((sched, u32:0));

    // This should be a cast, not a for loop, but casting to two-dimensional
    // arrays is not yet supported.
    let final_sched = RoundKey[MAX_NUM_ROUNDS + u32:1]:[u32[4]:[u32:0, ...], ...];
    let final_sched = for (i, final_sched) in u32:0..(MAX_NUM_ROUNDS + u32:1) {
        let start_idx = i * u32:4;
        update(final_sched, i,
               [sched[start_idx], sched[start_idx + u32:1],
                sched[start_idx + u32:2], sched[start_idx + u32:3]])
    }(final_sched);
    final_sched
}

#[test]
fn test_create_key_schedule_256() {
    let key = u8[32]:[
        u8:0xac, u8:0x75, u8:0x4a, u8:0x55,
        u8:0x99, u8:0x93, u8:0x7e, u8:0x79,
        u8:0x9e, u8:0x7c, u8:0x46, u8:0x97,
        u8:0xb1, u8:0xdd, u8:0x57, u8:0x14,
        u8:0xeb, u8:0x11, u8:0xda, u8:0x40,
        u8:0xb1, u8:0x8c, u8:0xa8, u8:0x29,
        u8:0x15, u8:0xaa, u8:0x4a, u8:0xd5,
        u8:0xde, u8:0x3c, u8:0x83, u8:0x35,
    ];
    let sched = create_key_schedule(key, KeyWidth::KEY_256);
    let expected_0 = RoundKey:[
        u8:0xac ++ u8:0x75 ++ u8:0x4a ++ u8:0x55,
        u8:0x99 ++ u8:0x93 ++ u8:0x7e ++ u8:0x79,
        u8:0x9e ++ u8:0x7c ++ u8:0x46 ++ u8:0x97,
        u8:0xb1 ++ u8:0xdd ++ u8:0x57 ++ u8:0x14,
    ];
    assert_eq(expected_0, sched[0]);

    let expected_1 = RoundKey:[
        u8:0xeb ++ u8:0x11 ++ u8:0xda ++ u8:0x40,
        u8:0xb1 ++ u8:0x8c ++ u8:0xa8 ++ u8:0x29,
        u8:0x15 ++ u8:0xaa ++ u8:0x4a ++ u8:0xd5,
        u8:0xde ++ u8:0x3c ++ u8:0x83 ++ u8:0x35,
    ];
    assert_eq(expected_1, sched[1]);

    let expected_2 = RoundKey:[
        u8:0x46 ++ u8:0x99 ++ u8:0xdc ++ u8:0x48,
        u8:0xdf ++ u8:0x0a ++ u8:0xa2 ++ u8:0x31,
        u8:0x41 ++ u8:0x76 ++ u8:0xe4 ++ u8:0xa6,
        u8:0xf0 ++ u8:0xab ++ u8:0xb3 ++ u8:0xb2,
    ];
    assert_eq(expected_2, sched[2]);

    let expected_3 = RoundKey:[
        u8:0x67 ++ u8:0x73 ++ u8:0xb7 ++ u8:0x77,
        u8:0xd6 ++ u8:0xff ++ u8:0x1f ++ u8:0x5e,
        u8:0xc3 ++ u8:0x55 ++ u8:0x55 ++ u8:0x8b,
        u8:0x1d ++ u8:0x69 ++ u8:0xd6 ++ u8:0xbe,
    ];
    assert_eq(expected_3, sched[3]);

    let expected_4 = RoundKey:[
        u8:0xbd ++ u8:0x6f ++ u8:0x72 ++ u8:0xec,
        u8:0x62 ++ u8:0x65 ++ u8:0xd0 ++ u8:0xdd,
        u8:0x23 ++ u8:0x13 ++ u8:0x34 ++ u8:0x7b,
        u8:0xd3 ++ u8:0xb8 ++ u8:0x87 ++ u8:0xc9,
    ];
    assert_eq(expected_4, sched[4]);

    // ...and jump to the last, since results build on each other.
    let expected_14 = RoundKey:[
        u8:0xb1 ++ u8:0xb3 ++ u8:0xe8 ++ u8:0x0b,
        u8:0x25 ++ u8:0xfc ++ u8:0x6d ++ u8:0x57,
        u8:0xe0 ++ u8:0xe6 ++ u8:0x9d ++ u8:0xe7,
        u8:0xc4 ++ u8:0x70 ++ u8:0x42 ++ u8:0xf7,
    ];
    assert_eq(expected_14, sched[14]);
}

// Verifies we produce a correct schedule for a given input key.
// Verification values were generated from from the BoringSSL implementation,
// commit efd09b7e.
#[test]
fn test_key_schedule_128() {
    let key = u8[32]:[
        u8:0x66, u8:0x65, u8:0x64, u8:0x63,
        u8:0x62, u8:0x61, u8:0x39, u8:0x38,
        u8:0x37, u8:0x36, u8:0x35, u8:0x34,
        u8:0x33, u8:0x32, u8:0x31, u8:0x30,
        u8:0x00, u8:0x00, u8:0x00, u8:0x00,
        u8:0x00, u8:0x00, u8:0x00, u8:0x00,
        u8:0x00, u8:0x00, u8:0x00, u8:0x00,
        u8:0x00, u8:0x00, u8:0x00, u8:0x00,
    ];
    let sched = create_key_schedule(key, KeyWidth::KEY_128);
    // Oof.
    assert_eq(sched[0], (key as uN[256] >> 128) as uN[128] as u32[4]);
    assert_eq(
        sched[1],
        RoundKey:[
            u8:0x44 ++ u8:0xa2 ++ u8:0x60 ++ u8:0xa0,
            u8:0x26 ++ u8:0xc3 ++ u8:0x59 ++ u8:0x98,
            u8:0x11 ++ u8:0xf5 ++ u8:0x6c ++ u8:0xac,
            u8:0x22 ++ u8:0xc7 ++ u8:0x5d ++ u8:0x9c]);
    assert_eq(
        sched[2],
        RoundKey:[
            u8:0x80 ++ u8:0xee ++ u8:0xbe ++ u8:0x33,
            u8:0xa6 ++ u8:0x2d ++ u8:0xe7 ++ u8:0xab,
            u8:0xb7 ++ u8:0xd8 ++ u8:0x8b ++ u8:0x07,
            u8:0x95 ++ u8:0x1f ++ u8:0xd6 ++ u8:0x9b]);
    // Since round key N depends on round key N-1, we'll just jump ahead to
    // the last one.
    assert_eq(
        sched[10],
        RoundKey:[
            u8:0x71 ++ u8:0x21 ++ u8:0x06 ++ u8:0xf1,
            u8:0x59 ++ u8:0xd1 ++ u8:0x69 ++ u8:0x25,
            u8:0x03 ++ u8:0x21 ++ u8:0xe7 ++ u8:0xaa,
            u8:0xb9 ++ u8:0xae ++ u8:0x62 ++ u8:0xe0]);
}

pub fn encrypt(key: Key, key_width: KeyWidth, block: Block) -> Block {
    let num_rounds = get_num_rounds(key_width);

    let round_keys = create_key_schedule(key, key_width);
    let block = aes_common::add_round_key(block, round_keys[0]);

    let block = for (round, last_block): (u32, Block) in u32:1..MAX_NUM_ROUNDS {
        let block = aes_common::sub_bytes(last_block);
        let block = aes_common::shift_rows(block);
        let block = aes_common::mix_columns(block);
        let block = aes_common::add_round_key(block, round_keys[round]);
        if round < num_rounds { block } else { last_block }
    }(block);
    let block = aes_common::sub_bytes(block);
    let block = aes_common::shift_rows(block);
    let block = aes_common::add_round_key(block, round_keys[num_rounds]);
    block
}

#[test]
fn test_encrypt_256() {
    let key = Key:[
        u8:0xcc, u8:0x6d, u8:0xe8, u8:0x07,
        u8:0xd0, u8:0x19, u8:0x80, u8:0xc9,
        u8:0x15, u8:0x96, u8:0xcc, u8:0xa8,
        u8:0x77, u8:0x40, u8:0x6b, u8:0x95,
        u8:0x8f, u8:0xdb, u8:0x3f, u8:0xe2,
        u8:0xac, u8:0x25, u8:0xed, u8:0x7d,
        u8:0xa0, u8:0x5b, u8:0x44, u8:0x92,
        u8:0x1a, u8:0x63, u8:0x15, u8:0x0f,
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
    assert_eq(expected, encrypt(key, KeyWidth::KEY_256, plaintext))
}

#[test]
fn test_encrypt_128() {
    let key = Key:[
        u8:0xb9, u8:0x6e, u8:0xe5, u8:0xe2,
        u8:0x8d, u8:0x4e, u8:0x6a, u8:0x22,
        u8:0xdb, u8:0x12, u8:0xea, u8:0xf5,
        u8:0x63, u8:0xf2, u8:0x05, u8:0x29,
        ...
    ];
    let plaintext = Block:[
        [u8:0xf5, u8:0x18, u8:0xa0, u8:0xad],
        [u8:0x5a, u8:0x0a, u8:0x8a, u8:0xfa],
        [u8:0x65, u8:0x60, u8:0x44, u8:0x68],
        [u8:0x0f, u8:0xc7, u8:0x4b, u8:0x3d],
    ];
    let expected = Block:[
        u8[4]:[u8:0x57, u8:0xcf, u8:0x83, u8:0x0a],
        u8[4]:[u8:0xdf, u8:0x69, u8:0x22, u8:0x7f],
        u8[4]:[u8:0x6c, u8:0x3a, u8:0xda, u8:0xab],
        u8[4]:[u8:0x51, u8:0xad, u8:0x76, u8:0x19],
    ];
    let actual = encrypt(key, KeyWidth::KEY_128, plaintext);
    assert_eq(actual, expected)
}

pub fn decrypt(key: Key, key_width: KeyWidth, block: Block) -> Block {
    let num_rounds = get_num_rounds(key_width);
    let round_keys = create_key_schedule(key, key_width);

    let block = aes_common::add_round_key(block, round_keys[num_rounds]);
    let block = aes_common::inv_shift_rows(block);
    let block = aes_common::inv_sub_bytes(block);

    let block = for (i, last_block): (u32, Block) in u32:1..MAX_NUM_ROUNDS {
        let round = num_rounds - i;
        let block = if i < num_rounds {
            let block = aes_common::add_round_key(last_block, round_keys[round]);
            let block = aes_common::inv_mix_columns(block);
            let block = aes_common::inv_shift_rows(block);
            let block = aes_common::inv_sub_bytes(block);
            block
        } else {
            last_block
        };
        block
    }(block);

    let block = aes_common::add_round_key(block, round_keys[0]);
    block
}

#[test]
fn test_decrypt_256() {
    let key = Key:[
        u8:0xcc, u8:0x6d, u8:0xe8, u8:0x07,
        u8:0xd0, u8:0x19, u8:0x80, u8:0xc9,
        u8:0x15, u8:0x96, u8:0xcc, u8:0xa8,
        u8:0x77, u8:0x40, u8:0x6b, u8:0x95,
        u8:0x8f, u8:0xdb, u8:0x3f, u8:0xe2,
        u8:0xac, u8:0x25, u8:0xed, u8:0x7d,
        u8:0xa0, u8:0x5b, u8:0x44, u8:0x92,
        u8:0x1a, u8:0x63, u8:0x15, u8:0x0f,
    ];
    let plaintext = Block:[
        [u8:0x43, u8:0x6b, u8:0x12, u8:0xb4],
        [u8:0x6f, u8:0xfa, u8:0xb4, u8:0x60],
        [u8:0x63, u8:0xc8, u8:0x44, u8:0xba],
        [u8:0x6d, u8:0x1c, u8:0xeb, u8:0xf8],
    ];
    assert_eq(plaintext,
              decrypt(key, KeyWidth::KEY_256, encrypt(key, KeyWidth::KEY_256, plaintext)))
}

#[test]
fn test_decrypt_128() {
    let plaintext = Block:[
        [u8:0xb6, u8:0x7f, u8:0x5e, u8:0x7f],
        [u8:0x22, u8:0x7c, u8:0xa2, u8:0xfc],
        [u8:0xf8, u8:0x80, u8:0x99, u8:0xba],
        [u8:0x1f, u8:0x14, u8:0x68, u8:0x29],
    ];
    let key = Key:[
        u8:0x50, u8:0xb7, u8:0x1d, u8:0x6e,
        u8:0x7f, u8:0x04, u8:0x59, u8:0x23,
        u8:0x5b, u8:0xc2, u8:0x7d, u8:0x93,
        u8:0x0a, u8:0x30, u8:0x9e, u8:0xa8,
        ...
    ];
    assert_eq(plaintext,
              decrypt(key, KeyWidth::KEY_128, encrypt(key, KeyWidth::KEY_128, plaintext)))
}
