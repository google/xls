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

// This file implements types and models for RAMs.
import std


// Single-port RAM request
pub struct RamReq<ADDR_WIDTH: u32, DATA_WIDTH:u32> {
  addr: bits[ADDR_WIDTH],
  data: bits[DATA_WIDTH],
  we: bool,
  re: bool,
}

// Single-port RAM response
pub struct RamResp<DATA_WIDTH: u32> {
  data: bits[DATA_WIDTH],
}

// Models a single-port RAM.
// Use the wrapper proc SinglePortRamModel to hide the proc state
// initialization.
proc SinglePortRamModel<DATA_WIDTH:u32, SIZE:u32, ADDR_WIDTH:u32=std::clog2(SIZE)> {
  req_chan: chan<RamReq<ADDR_WIDTH, DATA_WIDTH>> in;
  resp_chan : chan<RamResp<DATA_WIDTH>> out;

  init {
      bits[DATA_WIDTH][SIZE]: [bits[DATA_WIDTH]: 0, ...]
  }

  config(req: chan<RamReq<ADDR_WIDTH, DATA_WIDTH>> in,
         resp: chan<RamResp<DATA_WIDTH>> out) {
    (req, resp)
  }

  next(tok: token, state: bits[DATA_WIDTH][SIZE]) {
    let (tok, request) = recv(tok, req_chan);

    let (response, new_state) = if request.we {
      (
        RamResp { data: bits[DATA_WIDTH]:0 },
        update(state, request.addr, request.data),
      )
    } else {
      (
        RamResp { data: state[request.addr] },
        state,
      )
    };
    let tok = send_if(tok, resp_chan, request.re, response);
    new_state
  }
}

#[test_proc]
proc SinglePortRamModelTest {
  req_out: chan<RamReq<u32:10, u32:32>> out;
  resp_in: chan<RamResp<u32:32>> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (req_p, req_c) = chan<RamReq<u32:10, u32:32>>;
    let (resp_p, resp_c) = chan<RamResp<u32:32>>;
    spawn SinglePortRamModel<u32:32, u32:1024, >(req_c, resp_p);
    (req_p, resp_c, terminator)
  }

  next(tok: token, state: ()) {
    let MAX_OFFSET = u32:30;
    let _ = trace!(MAX_OFFSET);
    let tok = for (offset, tok): (u32, token) in range(u32:0, MAX_OFFSET) {
      // First, write the whole memory.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, RamReq {
            addr: addr as uN[10],
            data: (addr + offset) as uN[32],
            we: true,
            re: false,
        });
        tok
      } (tok);

      // Now check that what we wrote is still there.
      let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, RamReq {
          addr: addr as uN[10],
          data: uN[32]: 0,
          we: false,
          re: true,
        });
        let (tok, read_data) = recv(tok, resp_in);
        let _ = assert_eq(read_data.data, (addr + offset) as uN[32]);
        tok
      } (tok);
      tok
    } (tok);
    let tok = send(tok, terminator, true);
    ()
  }
}
