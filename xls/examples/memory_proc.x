// Copyright 2024 The XLS Authors
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
// Simple example of how a memory interface could look like.
// For a more comprehensive example that also can work with SRAM codegen, have
// a look at ram.x

pub const MEM_SIZE = u32:256;

pub type MemWord = u32;

pub struct MemReq {
    is_write: bool,
    address: u32,
    wdata: MemWord,  // if is_write request, the data to write
}

type State = MemWord[MEM_SIZE];

const DEFAULT_VALUE = u32:0x12345678 as MemWord;

pub proc Memory {
    req_in: chan<MemReq> in;
    data_out: chan<u32> out;

    config(req_in: chan<MemReq> in, data_out: chan<u32> out) { (req_in, data_out) }

    init { State:[DEFAULT_VALUE, ...] }

    next(state: State) {
        let tok = join();
        let (tok, req) = recv(tok, req_in);
        let state = if req.is_write { update(state, req.address, req.wdata) } else { state };

        // A read request returns a response.
        let tok = send_if(tok, data_out, !req.is_write, state[req.address]);
        state
    }
}

const TEST_WRITE_VALUE: u32 = u32:0xcafe600d;

#[test_proc]
proc MemoryTest {
    terminator: chan<bool> out;
    cmd_out: chan<MemReq> out;
    read_data_in: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (cmd_s, cmd_r) = chan<MemReq>("cmd");
        let (data_s, data_r) = chan<u32>("data");
        spawn Memory(cmd_r, data_s);
        (terminator, cmd_s, data_r)
    }

    init { () }

    next(state: ()) {
        let tok = join();

        // Read will yield default value in memory.
        let read_req = MemReq { is_write: false, address: u32:12, wdata: u32:0 };
        let tok = send(tok, cmd_out, read_req);
        let (tok, value) = recv(tok, read_data_in);
        assert_eq(value, DEFAULT_VALUE);

        // Write value to a particular address...
        let write_req = MemReq { is_write: true, address: u32:12, wdata: TEST_WRITE_VALUE };
        let tok = send(tok, cmd_out, write_req);

        // ... and expect it to be retrieved
        let read_req = MemReq { is_write: false, address: u32:12, wdata: u32:0 };
        let tok = send(tok, cmd_out, read_req);
        let (tok, value) = recv(tok, read_data_in);
        assert_eq(value, TEST_WRITE_VALUE);

        let tok = send(tok, terminator, true);
    }
}
