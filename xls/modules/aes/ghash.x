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
// XLS implementation of the GHASH subroutine of the Galois counter mode of operation for
// block ciphers, as described in NIST Special Publication 800-38D: "Recommendation for Block
// Cipher Modes of Operation: Galois/Counter Mode (GCM) and GMAC".
import xls.modules.aes.aes;
import xls.modules.aes.aes_common;

type Block = aes_common::Block;
type Key = aes_common::Key;

const ZERO_BLOCK = aes_common::ZERO_BLOCK;

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
pub fn gf128_mul(x: Block, y: Block) -> Block {
    let x = block_to_u128(x);
    let y = block_to_u128(y);

    let r = u8:0b11100001 ++ uN[120]:0;
    // TODO(rspringer): Can't currently select an element from an array or
    // tuple resulting from a for loop.
    let z_v = for (i, (last_z, last_v)) in u32:0..u32:128 {
        let z = if (x >> (u32:127 - i)) as u1 == u1:0 { last_z } else { last_z ^ last_v };
        let v = if last_v[0:1] == u1:0 { last_v >> 1 } else { (last_v >> 1) ^ r };
        (z, v)
    }((uN[128]:0, y));

    u128_to_block(z_v.0)
}

#[test]
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
    assert_eq(gf128_mul(a, b), b);

    // By the identity.
    let key = aes_common::Key:[u8:0, ...];
    let a = aes::encrypt(key, aes_common::KeyWidth::KEY_128, ZERO_BLOCK);
    let b = Block:[
        [u8:0x80, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
    ];
    let z = gf128_mul(a, b);
    assert_eq(z, a);

    // By two.
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
    assert_eq(z, expected);

    // Verify commutativity.
    let z = gf128_mul(b, a);
    assert_eq(z, expected);

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
    assert_eq(z, expected);

    // Verify commutativity _harder_.
    let z = gf128_mul(b, a);
    assert_eq(z, expected)
}

// Describes the inputs to the GHASH function/proc.
// TODO(rspringer): Make more flexible: enable partial AAD and ciphertext blocks.
pub struct Command {
    // The number of complete blocks of additional authentication data (AAD).
    aad_blocks: u32,

    // The number of complete blocks of ciphertext.
    ctxt_blocks: u32,

    // The hash key to use for tag generation: accepted as an input rather than
    // being computed to avoid introducing an AES block here.
    hash_key: Block,
}

// The current step/state of the GHASH block's FSM.
enum Step : u2 {
    IDLE = 0,
    RECV_INPUT = 1,
    HASH_LENGTHS = 2,
    INVALID = 3,
}

// The carried state of the GHASH proc.
pub struct State {
    // The current FSM step, as above.
    step: Step,

    // The current command being processed.
    command: Command,

    // The number of data blocks left to process.
    input_blocks_left: u32,

    // The output of the last state of the proc. Once all blocks plus the final
    // "lengths" block have been hashed, this is the output of the proc.
    last_tag: Block,
}

// Returns the state values to use during a proc "tick": either a new command
// or the carried state, depending on the FSM step/state.
fn get_current_state(state: State, command: Command) -> State {
    let command = if state.step == Step::IDLE { command } else { state.command };
    let input_blocks_left =
        if state.step == Step::IDLE { command.aad_blocks + command.ctxt_blocks }
        else { state.input_blocks_left };
    let last_tag = if state.step == Step::IDLE { ZERO_BLOCK } else { state.last_tag };
    let init_step =
        if input_blocks_left != u32:0 {
            Step::RECV_INPUT
        } else {
            Step::HASH_LENGTHS
        };
    let step = if state.step == Step::IDLE { init_step } else { state.step };

    State {
        step: step,
        command: command,
        input_blocks_left: input_blocks_left,
        last_tag: last_tag,
    }
}

// Calculates the authentication tag for the Galois Counter Mode of operation
// for block ciphers.
// Since input streams can be of ~arbitrary length, this must be implemented
// as a proc instead of as a fixed function. When idle, this proc accepts a new
// command and will consume a block of input, either AAD or ciphertext, per
// "tick" (read from the same channel). Once complete, the resulting tag will be
// sent on the provided output channel.
pub proc ghash {
    command_in: chan<Command> in;
    input_in: chan<Block> in;
    tag_out: chan<Block> out;

    init {
        State {
            step: Step::IDLE,
            command: Command {
                aad_blocks: u32:0,
                ctxt_blocks: u32:0,
                hash_key: ZERO_BLOCK,
            },
            input_blocks_left: u32:0,
            last_tag: ZERO_BLOCK,
        }
    }


    config(command_in: chan<Command> in, input_in: chan<Block> in,
           tag_out: chan<Block> out) {
        (command_in, input_in, tag_out)
    }

    next(state: State) {
        let (tok, command) = recv_if(
            join(), command_in, state.step == Step::IDLE, zero!<Command>());
        let state = get_current_state(state, command);

        // Get the current working block and update block counts.
        let (tok, input_block) = recv_if(
          tok, input_in, state.step == Step::RECV_INPUT, zero!<Block>());
        let block =
            if state.step == Step::RECV_INPUT {
                input_block
            } else {
                let x = ((state.command.aad_blocks * u32:128) as u64) as u32[2];
                let y = ((state.command.ctxt_blocks * u32:128) as u64) as u32[2];
                Block:[x[0] as u8[4], x[1] as u8[4], y[0] as u8[4], y[1] as u8[4]]
            };

        // Will underflow when state.step == Step::HASH_LENGTHS, but it doesn't matter.
        let input_blocks_left = state.input_blocks_left - u32:1;

        let last_tag = gf128_mul(aes_common::xor_block(state.last_tag, block), state.command.hash_key);
        let tok = send_if(tok, tag_out, state.step == Step::HASH_LENGTHS, last_tag);

        let new_step = match (state.step, input_blocks_left == u32:0) {
            (  Step::RECV_INPUT, false) => Step::RECV_INPUT,
            (  Step::RECV_INPUT,  true) => Step::HASH_LENGTHS,
            (Step::HASH_LENGTHS,     _) => Step::IDLE,
            // TODO(rspringer): Turn this info a fail!() when we can pass that
            // through IR optimization.
            _ => Step::INVALID,
        };

        State {
            step: new_step,
            command: state.command,
            input_blocks_left: input_blocks_left,
            last_tag: last_tag,
        }
    }
}

// General test of GHASH operation. Verified against the Go cipher package's
// GCM implementation.
#[test_proc]
proc ghash_test {
    command_out: chan<Command> out;
    data_out: chan<Block> out;
    tag_in: chan<Block> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (data_s, data_r) = chan<Block>("data");
        let (tag_s, tag_r) = chan<Block>("tag");

        spawn ghash(command_r, data_r, tag_s);
        (command_s, data_s, tag_r, terminator)
    }

    next(state: ()) {
        // Test 1: single AAD block, single ctxt block.
        let key = Key:[u8:0, ...];
        // No real need to use AES here, but no reason to _remove_ it, either.
        let hash_key = aes::encrypt(key, aes_common::KeyWidth::KEY_128, ZERO_BLOCK);

        let aad = Block:[
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let ctxt = Block:[
            [u8:0x40, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let command = Command {
            aad_blocks: u32:1,
            ctxt_blocks: u32:1,
            hash_key: hash_key,
        };
        let tok = send(join(), command_out, command);
        let tok = send(tok, data_out, aad);
        let tok = send(tok, data_out, ctxt);
        let (tok, tag) = recv(tok, tag_in);

        let expected = Block:[
            [u8:0x6c, u8:0xfa, u8:0x83, u8:0xe9],
            [u8:0x9f, u8:0xa1, u8:0x90, u8:0xe0],
            [u8:0xfa, u8:0xe9, u8:0x0d, u8:0x93],
            [u8:0x9e, u8:0x5e, u8:0x57, u8:0x14],
        ];
        assert_eq(tag, expected);

        // Test 2: two AAD blocks, three ctxt blocks. Random data.
        let key = Key:[
            u8:0xdf, u8:0x38, u8:0x8b, u8:0x6d, u8:0x4b, u8:0x99, u8:0x52, u8:0xd6,
            u8:0x42, u8:0x40, u8:0x7b, u8:0xea, u8:0xb3, u8:0x7a, u8:0x01, u8:0xc9,
            ...
        ];
        let hash_key = aes::encrypt(key, aes_common::KeyWidth::KEY_128, ZERO_BLOCK);

        let command = Command {
            aad_blocks: u32:2,
            ctxt_blocks: u32:3,
            hash_key: hash_key,
        };
        let tok = send(tok, command_out, command);

        let aad_0 = Block:[
            [u8:0xd7, u8:0xdb, u8:0xe2, u8:0x80],
            [u8:0x39, u8:0xf8, u8:0x76, u8:0xa9],
            [u8:0xd0, u8:0x9f, u8:0xe0, u8:0xcb],
            [u8:0xbd, u8:0xa4, u8:0xc3, u8:0x5a],       ];
        let aad_1 = Block:[
            [u8:0x57, u8:0x66, u8:0x1d, u8:0xb5],
            [u8:0x4f, u8:0x04, u8:0xe2, u8:0x9d],
            [u8:0x22, u8:0xf0, u8:0xda, u8:0x06],
            [u8:0x23, u8:0xb6, u8:0x95, u8:0x61],
        ];
        let tok = send(tok, data_out, aad_0);
        let tok = send(tok, data_out, aad_1);

        let ctxt_0 = Block:[
            [u8:0x20, u8:0x76, u8:0xbc, u8:0x70],
            [u8:0x41, u8:0x01, u8:0xba, u8:0x75],
            [u8:0x08, u8:0x20, u8:0xa9, u8:0x47],
            [u8:0x43, u8:0xb1, u8:0x38, u8:0x4a],
        ];
        let ctxt_1 = Block:[
            [u8:0xc8, u8:0xd9, u8:0xd5, u8:0xdd],
            [u8:0x19, u8:0xef, u8:0x72, u8:0x40],
            [u8:0x98, u8:0x72, u8:0x55, u8:0x49],
            [u8:0xbe, u8:0xbd, u8:0x33, u8:0xe5],
        ];
        let ctxt_2 = Block:[
            [u8:0x98, u8:0x60, u8:0x48, u8:0xf5],
            [u8:0xe0, u8:0x37, u8:0x03, u8:0x01],
            [u8:0xd8, u8:0x4e, u8:0x38, u8:0xdd],
            [u8:0x0b, u8:0x11, u8:0xdc, u8:0x8e],
        ];
        let tok = send(tok, data_out, ctxt_0);
        let tok = send(tok, data_out, ctxt_1);
        let tok = send(tok, data_out, ctxt_2);
        let (tok, tag) = recv(tok, tag_in);

        let expected = Block:[
            [u8:0x67, u8:0xd4, u8:0xb1, u8:0xfb],
            [u8:0x65, u8:0x20, u8:0x5f, u8:0x0a],
            [u8:0x0a, u8:0xb8, u8:0xb1, u8:0xc2],
            [u8:0x9d, u8:0x13, u8:0xae, u8:0x52],
        ];
        assert_eq(tag, expected);

        send(tok, terminator, true);
    }
}
