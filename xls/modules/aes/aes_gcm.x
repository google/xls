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

// Implements the GCM (Galois counter mode) mode of operation ("GCM mode" is a
// bit redundant) using the AES block cipher, as described in NIST SP 800-38D,
// https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf
// (link accessible as of 2022-08-24).
//
// TODOs:
//  - Support partial blocks of plaintext and AAD (additional authenticated data).
//  - Parameterize the proc in terms of number of CTR units.
import std;
import xls.modules.aes.aes;
import xls.modules.aes.aes_common;
import xls.modules.aes.aes_ctr;
import xls.modules.aes.ghash;

type Block = aes_common::Block;
type InitVector = aes_common::InitVector;
type Key = aes_common::Key;
type KeyWidth = aes_common::KeyWidth;

// Describes an encryption operation to be performed.
struct Command {
    // True to encrypt, false to decrypt.
    encrypt: bool,
    // The number of [complete] input message blocks to process.
    msg_blocks: u32,
    // The number of [complete] AAD blocks to process.
    aad_blocks: u32,
    // The AES encryption key.
    key: Key,
    // The key width.
    key_width: aes_common::KeyWidth,
    // The CTR mode initialization vector.
    iv: InitVector,
}

// The proc's FSM state.
enum Step : u3 {
    // Not processing any message; waiting for the next command.
    IDLE = 0,
    // Reading AAD from the input channel.
    READ_AAD = 1,
    // Reading message data from the input channel.
    READ_MSG = 2,
    // Finalizing the authentication tag.
    HASH_LENGTHS = 3,
    // Invalid. Should never be used.
    INVALID = 4,
}

// The full state of the proc.
struct State {
    // The FSM state, as above.
    step: Step,

    // The command currently being processed.
    command: Command,

    // The number of AAD blocks left to receive.
    aad_blocks_left: u32,

    // The number of input message blocks left to receive.
    msg_blocks_left: u32,
}

// Returns the State structure to be used for the current "tick", whose
// contents depend on a possible input command and the current state of the proc.
fn get_current_state(state: State, command: Command) -> State {
    let command = if state.step == Step::IDLE { command } else { state.command };
    let aad_blocks_left = if state.step == Step::IDLE { command.aad_blocks }
        else { state.aad_blocks_left };
    let msg_blocks_left = if state.step == Step::IDLE { command.msg_blocks }
        else { state.msg_blocks_left };

    let step =
        if state.step == Step::IDLE {
            if command.aad_blocks != u32:0 {
                Step::READ_AAD
            } else {
                if command.msg_blocks != u32:0 {
                    Step::READ_MSG
                } else {
                    Step::HASH_LENGTHS
                }
            }
        } else { state.step };
    State {
        step: step,
        command: command,
        aad_blocks_left: aad_blocks_left,
        msg_blocks_left: msg_blocks_left,
    }
}

// Constructs the command to send to the CTR unit when a new command is received.
fn get_ctr_command(gcm_command: Command) -> aes_ctr::Command {
    aes_ctr::Command {
        msg_bytes: gcm_command.msg_blocks * aes_common::BLOCK_BYTES,
        key: gcm_command.key,
        key_width: gcm_command.key_width,
        iv: gcm_command.iv,

        // The initial counter is 2 because counter value 0 is used to create
        // the hash key and counter value 1 is used as the final XOR of the
        // auth tag.
        initial_ctr: u32:2,

        ctr_stride: u32:1,
    }
}

// Constructs the command to send to the GHASH unit when a new command is received.
fn get_ghash_command(gcm_command: Command) -> ghash::Command {
    ghash::Command {
        aad_blocks: gcm_command.aad_blocks,
        ctxt_blocks: gcm_command.msg_blocks,
        hash_key: aes::encrypt(gcm_command.key, gcm_command.key_width, aes_common::ZERO_BLOCK),
    }
}

// Creates the block used for final masking of the authentication tag.
fn create_ctr_block(iv: InitVector) -> Block {
    let ctr_block = (iv ++ u32:1) as u8[16];
    Block:[
        [ctr_block[0], ctr_block[1], ctr_block[2], ctr_block[3]],
        [ctr_block[4], ctr_block[5], ctr_block[6], ctr_block[7]],
        [ctr_block[8], ctr_block[9], ctr_block[10], ctr_block[11]],
        [ctr_block[12], ctr_block[13], ctr_block[14], ctr_block[15]],
    ]
}

// The actual proc that implements AES-GCM.
// This uses a number (currently one) of CTR units to encrypt the message data
// and a single GHASH unit to compute the authentication tag. This proc makes use
// of three external channels:
//  - One for receiving command descriptors (defined above).
//  - One for reciving input data: AAD blocks followed by plaintext blocks.
//  - One for sending output data: ciphertext blocks followed by the single
//    authentication tag block.
//
// General operation: The proc reads in a command, then reads in the indicated
// number of blocks of AAD and plaintext. Each message block goes to
// the/a CTR unit, and each AAD block goes to the GHASH unit. Each chunk of
// ciphertext (coming out of CTR) is sent both back to the user and
// into GHASH to finish computing the auth tag.
// Once all ciphertext blocks have been sent to GHASH, the auth tag is
// read out and sent to the user, which completes processing of the command.
proc aes_gcm {
    command_in: chan<Command> in;
    data_in: chan<Block> in;
    data_out: chan<Block> out;

    // CTR mode data.
    ctr_cmd_out: chan<aes_ctr::Command> out;
    ctr_input_out: chan<Block> out;
    ctr_result_in: chan<Block> in;

    // GHASH data.
    ghash_cmd_out: chan<ghash::Command> out;
    ghash_input_out: chan<Block> out;
    ghash_tag_in: chan<Block> in;

    init {
        State {
            step: Step::IDLE,
            command: Command {
                encrypt: false,
                msg_blocks: u32:0,
                aad_blocks: u32:0,
                key: Key:[u8:0, ...],
                key_width: KeyWidth::KEY_128,
                iv: InitVector:0,
            },
            msg_blocks_left: u32:0,
            aad_blocks_left: u32:0,
        }
    }

    config(command_in: chan<Command> in,
           data_r: chan<Block> in,
           data_s: chan<Block> out) {
        let (ctr_cmd_s, ctr_cmd_r) = chan<aes_ctr::Command>("ctr_cmd");
        let (ctr_input_s, ctr_input_r) = chan<Block>("ctr_input");
        let (ctr_result_s, ctr_result_r) = chan<Block>("ctr_result");
        spawn aes_ctr::aes_ctr(ctr_cmd_r, ctr_input_r, ctr_result_s);

        let (ghash_cmd_s, ghash_cmd_r) = chan<ghash::Command>("ghash_cmd");
        let (ghash_input_s, ghash_input_r) = chan<Block>("ghash_input");
        let (ghash_tag_s, ghash_tag_r) = chan<Block>("ghash_tag");
        spawn ghash::ghash(ghash_cmd_r, ghash_input_r, ghash_tag_s);

        (command_in, data_r, data_s,
         ctr_cmd_s, ctr_input_s, ctr_result_r,
         ghash_cmd_s, ghash_input_s, ghash_tag_r)
    }

    next(state: State) {
        let (tok, command) = recv_if(
            join(), command_in, state.step == Step::IDLE, zero!<Command>());
        let ctr_command = get_ctr_command(command);
        let ghash_command = get_ghash_command(command);

        let last_step = state.step;
        let state = get_current_state(state, command);
        // Only send a new command if we transition from idle to active.
        let send_ctr_command = last_step == Step::IDLE && state.msg_blocks_left != u32:0;
        let tok0 = send_if(tok, ctr_cmd_out, send_ctr_command, ctr_command);
        let tok1 = send_if(tok, ghash_cmd_out, last_step == Step::IDLE, ghash_command);
        let tok = join(tok0, tok1);

        // Send the block we read to the appropriate place: if it's an AAD block,
        // it goes to GHASH. If it's a msg block, it goes to CTR.
        let (tok, input_block) = recv_if(
            tok, data_in, state.step == Step::READ_AAD || state.step == Step::READ_MSG,
            zero!<Block>());
        let aad_blocks_left =
            if state.step == Step::READ_AAD {
                if state.aad_blocks_left <= u32:1 { u32:0 } else { state.aad_blocks_left - u32:1 }
            } else {
                state.aad_blocks_left
            };
        let msg_blocks_left =
            if state.step == Step::READ_MSG {
                if state.msg_blocks_left <= u32:1 { u32:0 } else { state.msg_blocks_left - u32:1 }
            } else {
                state.msg_blocks_left
            };
        let tok1 = send_if(tok, ghash_input_out, state.step == Step::READ_AAD, input_block);
        let tok0 = send_if(tok, ctr_input_out, state.step == Step::READ_MSG, input_block);
        let tok = join(tok0, tok1);

        // We could slightly better performance if we delayed reading from CTR by one "tick"
        // after sending it data, but it's unclear if it'd be worth the complexity.
        let (tok, ctr_block) =
             recv_if(tok, ctr_result_in, state.step == Step::READ_MSG, zero!<Block>());
        // Ciphertext always goes to GHASH.
        let ghash_block = if state.command.encrypt { ctr_block } else { input_block };
        let tok0 = send_if(tok, ghash_input_out, state.step == Step::READ_MSG, ghash_block);
        let tok1 = send_if(tok, data_out, state.step == Step::READ_MSG, ctr_block);
        let tok = join(tok0, tok1);

        // Once we've read all outputs from CTR, we just need to read the last block from GHASH.
        let (tok, tag) = recv_if(
            tok, ghash_tag_in, state.step == Step::HASH_LENGTHS, zero!<Block>());

        // Finally, XOR the GHASH'ed tag with counter 1.
        let ctr_block = create_ctr_block(state.command.iv);
        let tag = aes_common::xor_block(
            tag, aes::encrypt(state.command.key, state.command.key_width, ctr_block));
        let tok = send_if(tok, data_out, state.step == Step::HASH_LENGTHS, tag);

        let aad_remaining = aad_blocks_left != u32:0;
        let msg_remaining = msg_blocks_left != u32:0;
        let next_step = match(state.step, aad_remaining, msg_remaining) {
            (Step::READ_AAD,  true,     _) => Step::READ_AAD,
            (Step::READ_AAD, false,  true) => Step::READ_MSG,
            (Step::READ_AAD, false, false) => Step::HASH_LENGTHS,
            (Step::READ_MSG,     _,  true) => Step::READ_MSG,
            (Step::READ_MSG,     _, false) => Step::HASH_LENGTHS,
            (Step::HASH_LENGTHS, _,     _) => Step::IDLE,
            // TODO(rspringer): Turn this info a fail!() when we can pass that
            // through IR optimization.
            _ => Step::INVALID,
        };

        State {
            step: next_step,
            command: state.command,
            msg_blocks_left: msg_blocks_left,
            aad_blocks_left: aad_blocks_left,
        }
    }
}

// Tests encryption of a single block of plaintext with a single block of AAD.
// 256-bit testing is run via the cc_test.
#[test_proc]
proc aes_gcm_test_smoke_128 {
    command_out: chan<Command> out;
    input_out: chan<Block> out;
    result_out: chan<Block> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (input_s, input_r) = chan<Block>("input");
        let (result_s, result_r) = chan<Block>("result");
        spawn aes_gcm(command_r, input_r, result_s);
        (command_s, input_s, result_r, terminator)
    }

    next(state: ()) {
        let command = Command {
            encrypt: true,
            msg_blocks: u32:1,
            aad_blocks: u32:1,
            key: Key:[u8:0, ...],
            key_width: KeyWidth::KEY_128,
            iv: InitVector:0,
        };
        let tok = send(join(), command_out, command);

        let msg_block = Block:[
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let tok = send(tok, input_out, msg_block);

        let aad_block = Block:[
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let tok = send(tok, input_out, aad_block);

        let (tok, ctxt) = recv(tok, result_out);
        let expected_ctxt = Block:[
            [u8:0x03, u8:0x88, u8:0xda, u8:0xce],
            [u8:0x60, u8:0xb6, u8:0xa3, u8:0x92],
            [u8:0xf3, u8:0x28, u8:0xc2, u8:0xb9],
            [u8:0x71, u8:0xb2, u8:0xfe, u8:0x78],
        ];
        assert_eq(ctxt, expected_ctxt);

        let (tok, tag) = recv(tok, result_out);
        let expected_tag = Block:[
            [u8:0xd2, u8:0x4e, u8:0x50, u8:0x3a],
            [u8:0x1b, u8:0xb0, u8:0x37, u8:0x07],
            [u8:0x1c, u8:0x71, u8:0xb3, u8:0x5d],
            [u8:0x98, u8:0x7b, u8:0x86, u8:0x57],
        ];
        assert_eq(tag, expected_tag);

        let tok = send(tok, terminator, true);
    }
}

// Tests encryption with three blocks of plaintext and two blocks of AAD.
#[test_proc]
proc aes_gcm_multi_block_gcm {
    command_out: chan<Command> out;
    input_out: chan<Block> out;
    result_out: chan<Block> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (input_s, input_r) = chan<Block>("input");
        let (result_s, result_r) = chan<Block>("result");
        spawn aes_gcm(command_r, input_r, result_s);
        (command_s, input_s, result_r, terminator)
    }

    next(state: ()) {
        let key = Key:[
            u8:0xfe, u8:0xdc, u8:0xba, u8:0x98,
            u8:0x76, u8:0x54, u8:0x32, u8:0x10,
            u8:0x01, u8:0x23, u8:0x45, u8:0x67,
            u8:0x89, u8:0xab, u8:0xcd, u8:0xef,
            ...];
        let iv = InitVector:0xdead_beef_cafe_f00d_abba_baab;
        let command = Command {
            encrypt: true,
            aad_blocks: u32:2,
            msg_blocks: u32:3,
            key: key,
            key_width: KeyWidth::KEY_128,
            iv: iv,
        };
        let tok = send(join(), command_out, command);

        // AAD.
        let aad_block = Block:[
            [u8:0xff, u8:0xee, u8:0xdd, u8:0xcc],
            [u8:0xbb, u8:0xaa, u8:0x99, u8:0x88],
            [u8:0x77, u8:0x66, u8:0x55, u8:0x44],
            [u8:0x33, u8:0x22, u8:0x11, u8:0x00],
        ];
        let tok = send(tok, input_out, aad_block);
        let aad_block = Block:[
            [u8:0xf0, u8:0xe1, u8:0xd2, u8:0xc3],
            [u8:0xb4, u8:0xa5, u8:0x96, u8:0x87],
            [u8:0x78, u8:0x69, u8:0x5a, u8:0x4b],
            [u8:0x3c, u8:0x2d, u8:0x1e, u8:0x0f],
        ];
        let tok = send(tok, input_out, aad_block);

        // Message text.
        let msg_block = Block:[
            [u8:0x01, u8:0x02, u8:0x03, u8:0x04],
            [u8:0x50, u8:0x60, u8:0x70, u8:0x80],
            [u8:0x09, u8:0xa0, u8:0x0b, u8:0xc0],
            [u8:0xd0, u8:0x0e, u8:0xf0, u8:0x11],
        ];
        let tok = send(tok, input_out, msg_block);

        let msg_block = Block:[
            [u8:0x11, u8:0x22, u8:0x33, u8:0x44],
            [u8:0x55, u8:0x66, u8:0x77, u8:0x88],
            [u8:0x99, u8:0xaa, u8:0xbb, u8:0xcc],
            [u8:0xdd, u8:0xee, u8:0xff, u8:0xa5],
        ];
        let tok = send(tok, input_out, msg_block);

        let msg_block = Block:[
            [u8:0xaa, u8:0x55, u8:0xaa, u8:0x55],
            [u8:0x55, u8:0xaa, u8:0x55, u8:0xaa],
            [u8:0xa5, u8:0x5a, u8:0xa5, u8:0x5a],
            [u8:0x5a, u8:0xa5, u8:0x5a, u8:0xa5],
        ];
        let tok = send(tok, input_out, msg_block);

        // Verify the ciphertext.
        let (tok, ctxt) = recv(tok, result_out);
        let expected_ctxt = Block:[
            [u8:0x14, u8:0x68, u8:0x63, u8:0xde],
            [u8:0xb3, u8:0x68, u8:0xfb, u8:0x35],
            [u8:0xcb, u8:0xeb, u8:0xa2, u8:0x79],
            [u8:0xc9, u8:0xe3, u8:0xef, u8:0xef],
        ];
        assert_eq(ctxt, expected_ctxt);

        let (tok, ctxt) = recv(tok, result_out);
        let expected_ctxt = Block:[
            [u8:0x86, u8:0xe1, u8:0x68, u8:0x07],
            [u8:0xd5, u8:0x4f, u8:0x2d, u8:0x97],
            [u8:0xd6, u8:0xea, u8:0x78, u8:0xce],
            [u8:0x7d, u8:0xfc, u8:0x08, u8:0xb0],
        ];
        assert_eq(ctxt, expected_ctxt);

        let (tok, ctxt) = recv(tok, result_out);
        let expected_ctxt = Block:[
            [u8:0xcc, u8:0xe6, u8:0x42, u8:0x11],
            [u8:0x76, u8:0x65, u8:0x12, u8:0x1b],
            [u8:0xee, u8:0xe2, u8:0xab, u8:0xf9],
            [u8:0x52, u8:0x08, u8:0xd7, u8:0x7c],
        ];
        assert_eq(ctxt, expected_ctxt);

        // And finally verify the tag.
        let (tok, tag) = recv(tok, result_out);
        let expected_tag = Block:[
            [u8:0x6a, u8:0xa1, u8:0x83, u8:0x0f],
            [u8:0x84, u8:0x53, u8:0xe7, u8:0xcb],
            [u8:0x99, u8:0x93, u8:0x64, u8:0xaa],
            [u8:0x42, u8:0x8c, u8:0xeb, u8:0x65],
        ];
        assert_eq(tag, expected_tag);

        let tok = send(tok, terminator, true);
    }
}

// Verifies that the proc operates correctly in the case of non-standard
// commands: 0 blocks of AAD, 0 blocks of msg, or 0 blocks of either.
// We can't inspect proc internal state, so we finish up this sequence with a
// normal encryption and verify the results.
#[test_proc]
proc aes_128_gcm_zero_block_commands {
    command_out: chan<Command> out;
    input_out: chan<Block> out;
    result_in: chan<Block> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (input_s, input_r) = chan<Block>("input");
        let (result_s, result_r) = chan<Block>("result");
        spawn aes_gcm(command_r, input_r, result_s);
        (command_s, input_s, result_r, terminator)
    }

    next(state: ()) {
        let key = Key:[
            u8:0xfe, u8:0xdc, u8:0xba, u8:0x98,
            u8:0x76, u8:0x54, u8:0x32, u8:0x10,
            u8:0x01, u8:0x23, u8:0x45, u8:0x67,
            u8:0x89, u8:0xab, u8:0xcd, u8:0xef,
            ...];
        let iv = InitVector:0xdead_beef_cafe_f00d_abba_baab;

        // 1. Zero blocks of msg.
        let command = Command {
            encrypt: true,
            aad_blocks: u32:1,
            msg_blocks: u32:0,
            key: key,
            key_width: KeyWidth::KEY_128,
            iv: iv,
        };
        let tok = send(join(), command_out, command);

        let aad_block = Block:[
            [u8:0xff, u8:0xee, u8:0xdd, u8:0xcc],
            [u8:0xbb, u8:0xaa, u8:0x99, u8:0x88],
            [u8:0x77, u8:0x66, u8:0x55, u8:0x44],
            [u8:0x33, u8:0x22, u8:0x11, u8:0x00],
        ];
        let tok = send(tok, input_out, aad_block);
        // Receive one tag block.
        let (tok, _) = recv(tok, result_in);

        // 2. Zero blocks of AAD.
        let command = Command {
            aad_blocks: u32:0,
            msg_blocks: u32:1,
            ..command
        };
        let tok = send(tok, command_out, command);

        let msg_block = Block:[
            [u8:0xff, u8:0xee, u8:0xdd, u8:0xcc],
            [u8:0xbb, u8:0xaa, u8:0x99, u8:0x88],
            [u8:0x77, u8:0x66, u8:0x55, u8:0x44],
            [u8:0x33, u8:0x22, u8:0x11, u8:0x00],
        ];
        let tok = send(tok, input_out, msg_block);
        // Receive one block of ciphertext and one auth tag block.
        let (tok, _) = recv(tok, result_in);
        let (tok, _) = recv(tok, result_in);

        // 3. Zero blocks of either.
        let command = Command {
            aad_blocks: u32:0,
            msg_blocks: u32:0,
            ..command
        };
        let tok = send(tok, command_out, command);
        // Receive one auth tag block.
        let (tok, _) = recv(tok, result_in);

        // Now just make sure we can do a normal transaction.
        let command = Command {
            encrypt: true,
            msg_blocks: u32:1,
            aad_blocks: u32:1,
            key: Key:[u8:0, ...],
            key_width: KeyWidth::KEY_128,
            iv: InitVector:0,
        };
        let tok = send(tok, command_out, command);

        let aad_block = Block:[
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let tok = send(tok, input_out, aad_block);

        let msg_block = Block:[
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
            [u8:0x00, u8:0x00, u8:0x00, u8:0x00],
        ];
        let tok = send(tok, input_out, msg_block);

        let (tok, ctxt) = recv(tok, result_in);
        let expected_ctxt = Block:[
            [u8:0x03, u8:0x88, u8:0xda, u8:0xce],
            [u8:0x60, u8:0xb6, u8:0xa3, u8:0x92],
            [u8:0xf3, u8:0x28, u8:0xc2, u8:0xb9],
            [u8:0x71, u8:0xb2, u8:0xfe, u8:0x78],
        ];
        assert_eq(ctxt, expected_ctxt);

        let (tok, tag) = recv(tok, result_in);
        let expected_tag = Block:[
            [u8:0xd2, u8:0x4e, u8:0x50, u8:0x3a],
            [u8:0x1b, u8:0xb0, u8:0x37, u8:0x07],
            [u8:0x1c, u8:0x71, u8:0xb3, u8:0x5d],
            [u8:0x98, u8:0x7b, u8:0x86, u8:0x57],
        ];
        assert_eq(tag, expected_tag);

        send(tok, terminator, true);
    }
}

// Test of sample_generator.cc output.
#[test_proc]
proc sample_generator_test {
    command_out: chan<Command> out;
    input_out: chan<Block> out;
    result_in: chan<Block> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (input_s, input_r) = chan<Block>("input");
        let (result_s, result_r) = chan<Block>("result");
        spawn aes_gcm(command_r, input_r, result_s);
        (command_s, input_s, result_r, terminator)
    }

    next(state: ()) {
        let key = Key:[
            u8:0x66, u8:0x63, u8:0x23, u8:0x41,
            u8:0x15, u8:0xb9, u8:0x6c, u8:0x76,
            u8:0xe9, u8:0x52, u8:0xce, u8:0xf2,
            u8:0x57, u8:0x0e, u8:0xe4, u8:0xf6,
            u8:0x37, u8:0x4f, u8:0x99, u8:0xb0,
            u8:0x3a, u8:0x5a, u8:0xea, u8:0x62,
            u8:0xbb, u8:0x7c, u8:0x8e, u8:0x2e,
            u8:0x99, u8:0xd8, u8:0x3b, u8:0x0d,
        ];
        let iv = InitVector:0xb1b4_01ac_bd3c_eec8_e2b1_dd06;
        let aad = Block[2]:[
            Block:[
                u8[4]:[u8:0xe0, u8:0x25, u8:0x73, u8:0x0b],
                u8[4]:[u8:0x4e, u8:0xcc, u8:0x88, u8:0x96],
                u8[4]:[u8:0x19, u8:0x3a, u8:0x0f, u8:0x42],
                u8[4]:[u8:0x41, u8:0x24, u8:0xbe, u8:0xc6],
            ],
            Block:[
                u8[4]:[u8:0x1b, u8:0xeb, u8:0x89, u8:0x71],
                u8[4]:[u8:0x26, u8:0xfa, u8:0x2b, u8:0xbb],
                u8[4]:[u8:0x0b, u8:0x1e, u8:0xb5, u8:0x4e],
                u8[4]:[u8:0x49, u8:0xc0, u8:0x14, u8:0x36],
            ],
        ];
        let msg = Block:[
            u8[4]:[u8:0x2f, u8:0x78, u8:0x74, u8:0xd6],
            u8[4]:[u8:0xc5, u8:0x10, u8:0xc1, u8:0x67],
            u8[4]:[u8:0x42, u8:0xa6, u8:0xa2, u8:0x91],
            u8[4]:[u8:0x14, u8:0xc3, u8:0xcb, u8:0xdd],
        ];
        let expected_msg = Block:[
            u8[4]:[u8:0xd9, u8:0x55, u8:0xa2, u8:0xff],
            u8[4]:[u8:0x63, u8:0x06, u8:0xdb, u8:0x1a],
            u8[4]:[u8:0x05, u8:0xce, u8:0x45, u8:0x0e],
            u8[4]:[u8:0xb5, u8:0x74, u8:0x25, u8:0x40],
        ];
        let expected_auth_tag = Block:[
            u8[4]:[u8:0xea, u8:0x5b, u8:0x7c, u8:0xe9],
            u8[4]:[u8:0x68, u8:0x70, u8:0xdc, u8:0x82],
            u8[4]:[u8:0x85, u8:0x81, u8:0x7d, u8:0x50],
            u8[4]:[u8:0x8c, u8:0x91, u8:0xa5, u8:0xd3],
        ];

        let command = Command {
            encrypt: true,
            aad_blocks: u32:2,
            msg_blocks: u32:1,
            key: key,
            key_width: KeyWidth::KEY_256,
            iv: iv,
        };
        let tok = send(join(), command_out, command);

        let tok = send(tok, input_out, aad[0]);
        let tok = send(tok, input_out, aad[1]);
        let tok = send(tok, input_out, msg);
        let (tok, ctxt) = recv(tok, result_in);
        assert_eq(ctxt, expected_msg);

        let (tok, tag) = recv(tok, result_in);
        assert_eq(tag, expected_auth_tag);

        send(tok, terminator, true);
    }
}
