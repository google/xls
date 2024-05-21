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

// Implements the CTR mode of operation, using AES as the block cipher.
import std;
import xls.modules.aes.aes;
import xls.modules.aes.aes_common;

type Block = aes_common::Block;
type InitVector = aes_common::InitVector;
type Key = aes_common::Key;

// The command sent to the encrypting proc at the beginning of processing.
pub struct Command {
    // The number of bytes to expect in the incoming message.
    // At present, this number must be a multiple of 128.
    msg_bytes: u32,

    // The encryption key.
    key: Key,

    // The width of the encryption key.
    key_width: aes_common::KeyWidth,

    // The initialization vector for the operation.
    iv: InitVector,

    // The initial counter value. When used standalone, this should be 0, but
    // when used as part of GCM, we start encrypting the plaintext with a
    // counter value of 2.
    initial_ctr: u32,

    // The amount by which to increment ctr every cycle. Usually 1, but can be
    // non-unit when part of a parallel GCM implementation.
    ctr_stride: u32,
}

// The current FSM state of the encoding block.
pub enum Step : bool {
  IDLE = 0,
  PROCESSING = 1,
}

// The recurrent state of the proc.
pub struct State {
  step: Step,
  command: Command,
  ctr: uN[32],
  blocks_left: uN[32],
}

// Performs the actual work of encrypting (or decrypting!) a block in CTR mode.
fn aes_ctr_encrypt(key: Key, key_width: aes_common::KeyWidth, ctr: uN[128], block: Block) -> Block {
    // TODO(rspringer): Avoid the need for this two-step type conversion.
    let ctr_array = ctr as u32[4];
    let ctr_enc = aes::encrypt(
        key, key_width,
        Block:[
            ctr_array[0] as u8[4],
            ctr_array[1] as u8[4],
            ctr_array[2] as u8[4],
            ctr_array[3] as u8[4]
        ]);
    Block:[
        ((ctr_enc[0] as u32) ^ (block[0] as u32)) as u8[4],
        ((ctr_enc[1] as u32) ^ (block[1] as u32)) as u8[4],
        ((ctr_enc[2] as u32) ^ (block[2] as u32)) as u8[4],
        ((ctr_enc[3] as u32) ^ (block[3] as u32)) as u8[4],
    ]
}

// Note that encryption and decryption are the _EXACT_SAME_PROCESS_!
pub proc aes_ctr {
    command_in: chan<Command> in;
    ptxt_in: chan<Block> in;
    ctxt_out: chan<Block> out;

    init {
        State {
            step: Step::IDLE,
            command: Command {
                msg_bytes: u32:0,
                key: Key:[u8:0, ...],
                key_width: aes_common::KeyWidth::KEY_128,
                iv: InitVector:uN[96]:0,
                initial_ctr: u32:0,
                ctr_stride: u32:0,
            },
            ctr: uN[32]:0,
            blocks_left: uN[32]:0,
        }
    }

    config(command_in: chan<Command> in,
           ptxt_in: chan<Block> in, ctxt_out: chan<Block> out) {
        (command_in, ptxt_in, ctxt_out)
    }

    next(state: State) {
        let step = state.step;

        let (tok, cmd) = recv_if(
            join(), command_in, step == Step::IDLE, zero!<Command>());
        let cmd = if step == Step::IDLE { cmd } else { state.command };
        let ctr = if step == Step::IDLE { cmd.initial_ctr } else { state.ctr };
        let blocks_left = if step == Step::IDLE {
            std::ceil_div(cmd.msg_bytes, u32:16)
        } else {
            state.blocks_left
        };
        let full_ctr = cmd.iv ++ ctr;

        let (tok, block) = recv_if(
            tok, ptxt_in, blocks_left != u32:0, zero!<Block>());
        let ctxt = aes_ctr_encrypt(cmd.key, cmd.key_width, full_ctr, block);
        let tok = send(tok, ctxt_out, ctxt);

        let blocks_left = blocks_left - u32:1;
        let step = if blocks_left == u32:0 { Step::IDLE } else { Step::PROCESSING };

        // We don't have to worry about ctr overflowing (which would result in an
        // invalid encryption, since ctr starts at zero, and the maximum possible
        // number of blocks per command is 2^32 - 1.
        State { step: step, command: cmd, ctr: ctr + cmd.ctr_stride, blocks_left: blocks_left }
    }
}

#[test_proc]
proc aes_ctr_test_128 {
    terminator: chan<bool> out;

    command_out: chan<Command> out;
    ptxt_out: chan<Block>out;
    ctxt_in: chan<Block> in;

    init { () }

    config(terminator: chan<bool> out) {
        let (command_s, command_r) = chan<Command>("command");
        let (ptxt_s, ptxt_r) = chan<Block>("ptxt");
        let (ctxt_s, ctxt_r) = chan<Block>("ctxt");
        spawn aes_ctr(command_r, ptxt_r, ctxt_s);
        (terminator, command_s, ptxt_s, ctxt_r)
    }

    next(state: ()) {
        let key = Key:[
            u8:0x00, u8:0x01, u8:0x02, u8:0x03,
            u8:0x04, u8:0x05, u8:0x06, u8:0x07,
            u8:0x08, u8:0x09, u8:0x0a, u8:0x0b,
            u8:0x0c, u8:0x0d, u8:0x0e, u8:0x0f,
            ...
        ];
        let iv = u8[12]:[
            u8:0x10, u8:0x11, u8:0x12, u8:0x13,
            u8:0x14, u8:0x15, u8:0x16, u8:0x17,
            u8:0x18, u8:0x19, u8:0x1a, u8:0x1b,
        ] as InitVector;
        let cmd = Command {
            msg_bytes: u32:32,
            key: key,
            key_width: aes_common::KeyWidth::KEY_128,
            iv: iv,
            initial_ctr: u32:0,
            ctr_stride: u32:1,
        };
        let tok = send(join(), command_out, cmd);

        let plaintext_0 = Block:[
            u8[4]:[u8:0x20, u8:0x21, u8:0x22, u8:0x23],
            u8[4]:[u8:0x24, u8:0x25, u8:0x26, u8:0x27],
            u8[4]:[u8:0x28, u8:0x29, u8:0x2a, u8:0x2b],
            u8[4]:[u8:0x2c, u8:0x2d, u8:0x2e, u8:0x2f],
        ];
        let tok = send(tok, ptxt_out, plaintext_0);
        let (tok, ctxt) = recv(tok, ctxt_in);
        let expected = Block:[
            u8[4]:[u8:0x27, u8:0x6a, u8:0xec, u8:0x41],
            u8[4]:[u8:0xfd, u8:0xa9, u8:0x9f, u8:0x26],
            u8[4]:[u8:0x34, u8:0xc5, u8:0x43, u8:0x73],
            u8[4]:[u8:0xc7, u8:0x99, u8:0xd2, u8:0x19],
        ];
        assert_eq(ctxt, expected);

        let plaintext_1 = Block:[
            u8[4]:[u8:0x30, u8:0x31, u8:0x32, u8:0x33],
            u8[4]:[u8:0x34, u8:0x35, u8:0x36, u8:0x37],
            u8[4]:[u8:0x38, u8:0x39, u8:0x3a, u8:0x3b],
            u8[4]:[u8:0x3c, u8:0x3d, u8:0x3e, u8:0x3f],
        ];
        let tok = send(tok, ptxt_out, plaintext_1);
        let (tok, ctxt) = recv(tok, ctxt_in);
        let expected = Block:[
            u8[4]:[u8:0x3e, u8:0xe6, u8:0x17, u8:0xa9],
            u8[4]:[u8:0xe9, u8:0x25, u8:0x27, u8:0xd6],
            u8[4]:[u8:0x61, u8:0xe9, u8:0x34, u8:0x5a],
            u8[4]:[u8:0x8d, u8:0xaf, u8:0x6a, u8:0x2f],
        ];
        assert_eq(ctxt, expected);

        // Command #2.
        let cmd = Command {
            msg_bytes: u32:16,
            key: key,
            key_width: aes_common::KeyWidth::KEY_128,
            iv: iv,
            initial_ctr: u32:0,
            ctr_stride: u32:1,
        };

        let tok = send(tok, command_out, cmd);
        let plaintext_0 = Block:[
            u8[4]:[u8:0x20, u8:0x21, u8:0x22, u8:0x23],
            u8[4]:[u8:0x24, u8:0x25, u8:0x26, u8:0x27],
            u8[4]:[u8:0x28, u8:0x29, u8:0x2a, u8:0x2b],
            u8[4]:[u8:0x2c, u8:0x2d, u8:0x2e, u8:0x2f],
        ];
        let tok = send(tok, ptxt_out, plaintext_0);
        let (tok, ctxt) = recv(tok, ctxt_in);
        let expected = Block:[
            u8[4]:[u8:0x27, u8:0x6a, u8:0xec, u8:0x41],
            u8[4]:[u8:0xfd, u8:0xa9, u8:0x9f, u8:0x26],
            u8[4]:[u8:0x34, u8:0xc5, u8:0x43, u8:0x73],
            u8[4]:[u8:0xc7, u8:0x99, u8:0xd2, u8:0x19],
        ];
        assert_eq(ctxt, expected);

        // Now test decryption! Just do a single block.
        let cmd = Command {
            msg_bytes: u32:16,
            key: key,
            key_width: aes_common::KeyWidth::KEY_128,
            iv: iv,
            initial_ctr: u32:0,
            ctr_stride: u32:1,
        };
        let tok = send(tok, command_out, cmd);
        let ciphertext_0 = Block:[
            u8[4]:[u8:0x27, u8:0x6a, u8:0xec, u8:0x41],
            u8[4]:[u8:0xfd, u8:0xa9, u8:0x9f, u8:0x26],
            u8[4]:[u8:0x34, u8:0xc5, u8:0x43, u8:0x73],
            u8[4]:[u8:0xc7, u8:0x99, u8:0xd2, u8:0x19],
        ];
        let tok = send(tok, ptxt_out, ciphertext_0);
        let (tok, ptxt) = recv(tok, ctxt_in);
        let expected = Block:[
            u8[4]:[u8:0x20, u8:0x21, u8:0x22, u8:0x23],
            u8[4]:[u8:0x24, u8:0x25, u8:0x26, u8:0x27],
            u8[4]:[u8:0x28, u8:0x29, u8:0x2a, u8:0x2b],
            u8[4]:[u8:0x2c, u8:0x2d, u8:0x2e, u8:0x2f],
        ];
        assert_eq(ptxt, expected);

        let tok = send(tok, terminator, true);
    }
}
