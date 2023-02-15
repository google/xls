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


// Abstract RAM requests and responses (arbitrary ports, options, etc.)
// Can be lowered to concrete RAM requests and responses, e.g. single-port RAM.
//
// Read: (address, mask) -> (data)
pub struct ReadReq<ADDR_WIDTH:u32, NUM_PARTITIONS:u32> {
  addr: bits[ADDR_WIDTH],
  mask: bits[NUM_PARTITIONS],
}
pub struct ReadResp<DATA_WIDTH:u32> {
  data: bits[DATA_WIDTH],
}

// Write: (address, data, mask) -> ()
pub struct WriteReq<ADDR_WIDTH:u32, DATA_WIDTH:u32, NUM_PARTITIONS:u32> {
  addr: bits[ADDR_WIDTH],
  data: bits[DATA_WIDTH],
  mask: bits[NUM_PARTITIONS],
}
pub struct WriteResp {}

pub fn WriteWordReq<NUM_PARTITIONS:u32, ADDR_WIDTH:u32, DATA_WIDTH:u32>(
  addr:uN[ADDR_WIDTH], data:uN[DATA_WIDTH]) ->
   WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
  WriteReq {
        addr: addr,
        data: data,
        mask: std::unsigned_max_value<NUM_PARTITIONS>(),
      }
}

pub fn ReadWordReq<NUM_PARTITIONS:u32, ADDR_WIDTH:u32>(addr:uN[ADDR_WIDTH]) ->
 ReadReq<ADDR_WIDTH, NUM_PARTITIONS> {
  ReadReq<ADDR_WIDTH, NUM_PARTITIONS> {
    addr: addr,
    mask: std::unsigned_max_value<NUM_PARTITIONS>(),
  }
}

// Behavior of reads and writes to the same address in the same "tick".
enum SimultaneousReadWriteBehavior : u2 {
  // The read shows the contents at the address before the write.
  READ_BEFORE_WRITE = 0,
  // The read shows the contents at the address after the write.
  WRITE_BEFORE_READ = 1,
  // Reading an address that is being written in the same tick causes an
  // assertion failure.
  ASSERT_NO_CONFLICT = 2,
}

// Flatten an array into a word.
fn flatten<N:u32, M:u32, TOTAL:u32={N*M}>(value: uN[N][M]) -> uN[TOTAL] {
  value as uN[TOTAL]
}

// Expands a mask of NUM_PARTITIONS bits to a mask of DATA_WIDTH bits, repeating
// each bit in the smaller mask. The RAM model has a notion of "partitions", a
// group of (potentially many) bits that are all activated by a single bit in
// the mask. When masking data bits, it is useful to expand the mask from 1 bit
// per partition to one bit bit per data bit.
fn expand_mask<DATA_WIDTH:u32, NUM_PARTITIONS:u32,
 EXPANSION_FACTOR:u32={std::ceil_div(DATA_WIDTH, NUM_PARTITIONS)}>(
  partition_mask:uN[NUM_PARTITIONS]) -> uN[DATA_WIDTH] {
  for (idx, data_mask): (u32, uN[DATA_WIDTH]) in range(u32:0, NUM_PARTITIONS) {
    let data_mask_segment =
      flatten(u1[EXPANSION_FACTOR]: [partition_mask[idx +: u1], ...]);
    ((data_mask_segment as uN[DATA_WIDTH]) << (idx * EXPANSION_FACTOR)) |
      data_mask
  } (uN[DATA_WIDTH]:0)
}

#[test]
fn expand_mask_test() {
  // Try expanding all 2-bit masks to 4-bit masks.
  let _ = assert_eq(u4:0b0000, expand_mask<u32:4>(u2:0b00));
  let _ = assert_eq(u4:0b1100, expand_mask<u32:4>(u2:0b10));
  let _ = assert_eq(u4:0b0011, expand_mask<u32:4>(u2:0b01));
  let _ = assert_eq(u4:0b1111, expand_mask<u32:4>(u2:0b11));

  // Try expanding all 2-bit masks to 3-bit masks, which are the same as the
  // 4-bit masks with the MSB dropped.
  let _ = assert_eq(u3:0b000, expand_mask<u32:3>(u2:0b00));
  let _ = assert_eq(u3:0b100, expand_mask<u32:3>(u2:0b10));
  let _ = assert_eq(u3:0b011, expand_mask<u32:3>(u2:0b01));
  let _ = assert_eq(u3:0b111, expand_mask<u32:3>(u2:0b11));

  // Try expanding 2-bit masks to 2-bit masks, which should be identity.
  let _ = assert_eq(u2:0b00, expand_mask<u32:2>(u2:0b00));
  let _ = assert_eq(u2:0b10, expand_mask<u32:2>(u2:0b10));
  let _ = assert_eq(u2:0b01, expand_mask<u32:2>(u2:0b01));
  let _ = assert_eq(u2:0b11, expand_mask<u32:2>(u2:0b11));
}

// Writes value `write_value` with write mask `mask` over previous value
// `mem_word`. The first element in the return tuple is the updated value and
// the second is the updated initialization.
fn write_word<DATA_WIDTH:u32, NUM_PARTITIONS:u32>(
  mem_word: uN[DATA_WIDTH],
  mem_initialized: bool[NUM_PARTITIONS],
  write_value: uN[DATA_WIDTH],
  mask: uN[NUM_PARTITIONS],
) -> (uN[DATA_WIDTH], bool[NUM_PARTITIONS]) {
  // TODO: compute mask when NUM_PARTITIONS != DATA_WIDTH
  let expanded_mask = expand_mask<DATA_WIDTH>(mask);
  let new_word = (mem_word & !expanded_mask) | (write_value & expanded_mask);
  let new_initialization =
    for (idx, partial_initialization): (u32, bool[NUM_PARTITIONS]) in
    range(u32:0, NUM_PARTITIONS) {
      if mask[idx+:bool] {
        update(partial_initialization, idx, true)
      } else { partial_initialization }
  } (mem_initialized);
  (new_word, new_initialization)
}

#[test]
fn write_word_test() {
  let _ = assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b11),
    (u2:1, [true, true])
  );
  let _ = assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b01),
    (u2:1, [true, false])
  );
  let _ = assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b10),
    (u2:0, [false, true])
  );
  let _ = assert_eq(
    write_word(u2:0, [false, true], u2:0, u2:0b01),
    (u2:0, [true, true])
  );
  let _ = assert_eq(
    write_word(u2:0, [false], u2:0, u1:0b0),
    (u2:0, [false])
  );
  let _ = assert_eq(
    write_word(u2:0, [false], u2:0, u1:0b1),
    (u2:0, [true])
  );
}

fn bits_to_bool_array<N:u32>(x:uN[N]) -> bool[N] {
  for (idx, partial): (u32, bool[N]) in range(u32:0, N) {
    update(partial, idx, x[idx+:bool])
  }(bool[N]:[false,...])
}

// Model of an abstract RAM. The RAM has channel pairs for supported operations,
// the usage model is to send on a `req` channel and receive on the
// corresponding `resp` channel. The `resp` channel may be an empty struct where
// the receive only serves to provide a token for completion of the op.
// Supported operations include read, write, masked write, and set.
// Parameters:
//  DATA_WIDTH: number of bits in each word in the RAM.
//  SIZE: number of words in the RAM.
//  WORD_PARTITION_SIZE: granularity (in bits) that words can be masked, e.g.
//   WORD_PARTITION_SIZE=u32:1 means that masked writes can set individual bits.
//   Default value is 1.
//  SIMULTANEOUS_READ_WRITE_BEHAVIOR: behavior when a read and write happen in
//   the same tick. Read before write is the default behavior.
//  INITIALIZED: if true, the RAM is marked as being initialized. When
//   ASSERT_VALID_READ is true, reading to an unwritten address will not throw
//   an assertion error because the memory is considered to be "written" at
//   initialization. False by default.
//   TODO(google/xls#818): Currently, DSLX default parameters don't support
//   complex expressions, and as a result this parameter can't have a default.
//   For now, we remove the parameter until DSLX supports more complex
//   parameters. If INITIALIZED is set, the initial value of the memory is 0 and
//   cannot be overridden.
//  ASSERT_VALID_READ: if true, add assertion that read operations are only
//   performed on values that a previous write has set. This is meant to model
//   asserting that a user doesn't read X from an unitialized SRAM.
proc RamModel<DATA_WIDTH:u32, SIZE:u32, WORD_PARTITION_SIZE:u32={u32:1},
  SIMULTANEOUS_READ_WRITE_BEHAVIOR:SimultaneousReadWriteBehavior=
   {SimultaneousReadWriteBehavior::READ_BEFORE_WRITE},
  INITIALIZED:bool={false},
  ASSERT_VALID_READ:bool={true}, ADDR_WIDTH:u32 = {std::clog2(SIZE)},
  NUM_PARTITIONS:u32=
  {(DATA_WIDTH + WORD_PARTITION_SIZE - u32:1)/WORD_PARTITION_SIZE}> {

  read_req: chan<ReadReq<ADDR_WIDTH, NUM_PARTITIONS>> in;
  read_resp: chan<ReadResp<DATA_WIDTH>> out;
  write_req: chan<WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in;
  write_resp: chan<WriteResp> out;

  init {
      (
        // mem contents initialized to whatever INITIAL_VALUE contains (zero by
        // default).
        // TODO(google/xls#818): use a parameter for the initial value.
        uN[DATA_WIDTH][SIZE]:[uN[DATA_WIDTH]:0,...],
        // mem_initialized initialized to whatever INITIALIZED is (false by
        // default, indicating no data has been written yet).
        bool[NUM_PARTITIONS][SIZE]:
         [bool[NUM_PARTITIONS]:[INITIALIZED, ...], ...],
      )
  }

  config(
    read_req: chan<ReadReq<ADDR_WIDTH, NUM_PARTITIONS>> in,
    read_resp: chan<ReadResp<DATA_WIDTH>> out,
    write_req: chan<WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in,
    write_resp: chan<WriteResp> out,
  ) {
    (read_req, read_resp, write_req, write_resp)
  }

  next(tok:token, state:(bits[DATA_WIDTH][SIZE], bool[NUM_PARTITIONS][SIZE])) {
    // state consists of an array storing the memory state, as well as an array
    // indicating if the each subword partition has been initialized.
    let (mem, mem_initialized) = state;

    // Perform nonblocking receives on each request channel.
    let (tok, read_req, read_req_valid) = recv_non_blocking(tok, read_req);
    let (tok, write_req, write_req_valid) = recv_non_blocking(tok, write_req);

    // Assert memory being read is initialized by checking that all partitions
    // have been initialized.
    let _ = if read_req_valid && ASSERT_VALID_READ {
      assert_eq(
        mem_initialized[read_req.addr],
        bits_to_bool_array(read_req.mask))
    } else { () };

    let (value_to_write, written_mem_initialized) = write_word(
      mem[write_req.addr], mem_initialized[write_req.addr],
      write_req.data, write_req.mask);

    let unmasked_read_value =
     if write_req_valid && read_req.addr == write_req.addr {
      // If we are simultaneously reading and writing the same address, check
      // SIMULTANEOUS_READ_WRITE_BEHAVIOR for the desired behavior.
      match SIMULTANEOUS_READ_WRITE_BEHAVIOR {
        SimultaneousReadWriteBehavior::READ_BEFORE_WRITE => mem[read_req.addr],
        SimultaneousReadWriteBehavior::WRITE_BEFORE_READ => value_to_write,
        SimultaneousReadWriteBehavior::ASSERT_NO_CONFLICT => {
          // Assertion failure, we have a conflicting read and write.
          let _ = assert_eq(true, false);
          mem[read_req.addr]  // Need to return something.
        },
      }
    } else { mem[read_req.addr] };
    let read_resp_value = ReadResp<DATA_WIDTH> {
      data: unmasked_read_value & read_req.mask,
    };
    let tok = send_if(tok, read_resp, read_req_valid, read_resp_value);

    // If we're doing a write, update the memory and mem_initialized. We
    // previously computed the updated values as they were potentially needed
    // for reads if writes were visible before reads.
    let mem = if write_req_valid {
      update(mem, write_req.addr, value_to_write)
    } else { mem };
    let mem_initialized = if write_req_valid {
      update(mem_initialized, write_req.addr, written_mem_initialized)
    } else { mem_initialized };
    let tok = send_if(tok, write_resp, write_req_valid, WriteResp{});

    (mem, mem_initialized)
  }
}

// Tests writing various patterns to a small memory, including some masked
// writes.
#[test_proc]
proc RamModelWriteReadMaskedWriteReadTest {
  read_req: chan<ReadReq<u32:8, u32:32>> out;
  read_resp: chan<ReadResp<u32:32>> in;
  write_req: chan<WriteReq<u32:8, u32:32, u32:32>> out;
  write_resp: chan<WriteResp> in;

  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (read_req_p, read_req_c) = chan<ReadReq<u32:8, u32:32>>;
    let (read_resp_p, read_resp_c) = chan<ReadResp<u32:32>>;
    let (write_req_p, write_req_c) = chan<WriteReq<u32:8, u32:32, u32:32>>;
    let (write_resp_p, write_resp_c) = chan<WriteResp>;
    spawn RamModel<
      u32:32,  // DATA_WIDTH
      u32:256, // SIZE
      u32:1    // WORD_PARTITION_SIZE
    >(
      read_req_c, read_resp_p, write_req_c, write_resp_p);
    (read_req_p, read_resp_c, write_req_p, write_resp_c, terminator,)
  }

  next(tok: token, state: ()) {
    let NUM_OFFSETS = u32:8;
    let tok = for (offset, tok): (u32, token) in range(u32:0, NUM_OFFSETS) {
      let _ = trace!(offset);

      // First, write the whole memory.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:256) {
        let tok = send(tok, write_req, WriteWordReq<u32:32>(
            addr as uN[8],
            (addr + offset) as uN[32]));
        let (tok, _) = recv(tok, write_resp);
        tok
      } (tok);

      // Now check that what we wrote is still there.
      let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:256) {
        let tok = send(tok, read_req, ReadWordReq<u32:32>(addr as uN[8]));
        let (tok, read_data) = recv(tok, read_resp);
        let _ = assert_eq(read_data.data, (addr + offset) as uN[32]);
        tok
      } (tok);
      tok
    } (tok);

    // Test that masked writes work.
    let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:256) {
      let bit_idx = addr & u32:0x1F;  // addr % 32
      let tok = send(tok, write_req,
        WriteReq<u32:8, u32:32, u32:32> {
          addr: addr as uN[8],
          data: u32:0,
          mask: (u32:1 << bit_idx),
        });
      let (tok, _) = recv(tok, write_resp);
      let tok = send(tok, read_req, ReadWordReq<u32:32>(addr as uN[8]));
      let (tok, read_data) = recv(tok, read_resp);
      let expected = (addr + NUM_OFFSETS - u32:1) & !(u32:1 << bit_idx);
      let _ = assert_eq(read_data.data, expected);
      tok
    } (tok);

    let tok = send(tok, terminator, true);
    ()
  }
}

// Tests that a RAM with initialization can be read.
#[test_proc]
proc RamModelInitializationTest {
  read_req: chan<ReadReq<u32:8, u32:32>> out;
  read_resp: chan<ReadResp<u32:32>> in;

  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (read_req_p, read_req_c) = chan<ReadReq<u32:8, u32:32>>;
    let (read_resp_p, read_resp_c) = chan<ReadResp<u32:32>>;
    let (_, write_req_c) = chan<WriteReq<u32:8, u32:32, u32:32>>;
    let (write_resp_p, _) = chan<WriteResp>;
    spawn RamModel<
      u32:32,   // DATA_WIDTH
      u32:256,  // SIZE
      u32:1,    // WORD_PARTITION_SIZE
      SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
      // SIMULTANEOUS_READ_WRITE_BEHAVIOR
      true     // INITIALIZED
    >(
      read_req_c, read_resp_p, write_req_c, write_resp_p);
    (read_req_p, read_resp_c, terminator,)
  }

  next(tok: token, state: ()) {
    // Now check that what we wrote is still there.
    let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:256) {
      let tok = send(tok, read_req, ReadWordReq<u32:32>(addr as uN[8]));
      let (tok, read_data) = recv(tok, read_resp);
      let _ = assert_eq(read_data.data, u32:0);
      tok
    } (tok);

    let tok = send(tok, terminator, true);
    ()
  }
}

// Single-port RAM request
pub struct SinglePortRamReq<ADDR_WIDTH:u32, DATA_WIDTH:u32> {
  addr: bits[ADDR_WIDTH],
  data: bits[DATA_WIDTH],
  we: bool,
  re: bool,
}

// Single-port RAM response
pub struct SinglePortRamResp<DATA_WIDTH:u32> {
  data: bits[DATA_WIDTH],
}

// Models a single-port RAM.
proc SinglePortRamModel<DATA_WIDTH:u32, SIZE:u32,
 ADDR_WIDTH:u32={std::clog2(SIZE)}> {
  req_chan: chan<SinglePortRamReq<ADDR_WIDTH, DATA_WIDTH>> in;
  resp_chan : chan<SinglePortRamResp<DATA_WIDTH>> out;

  init {
      bits[DATA_WIDTH][SIZE]: [bits[DATA_WIDTH]: 0, ...]
  }

  config(req: chan<SinglePortRamReq<ADDR_WIDTH, DATA_WIDTH>> in,
         resp: chan<SinglePortRamResp<DATA_WIDTH>> out) {
    (req, resp)
  }

  next(tok: token, state: bits[DATA_WIDTH][SIZE]) {
    let (tok, request) = recv(tok, req_chan);

    let (response, new_state) = if request.we {
      (
        SinglePortRamResp { data: bits[DATA_WIDTH]:0 },
        update(state, request.addr, request.data),
      )
    } else {
      (
        SinglePortRamResp { data: state[request.addr] },
        state,
      )
    };
    let tok = send_if(tok, resp_chan, request.re, response);
    new_state
  }
}

// Tests writing various patterns to a single port RAM by reading the contents
// afterwards and checking that you got what you wrote.
#[test_proc]
proc SinglePortRamModelTest {
  req_out: chan<SinglePortRamReq<u32:10, u32:32>> out;
  resp_in: chan<SinglePortRamResp<u32:32>> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (req_p, req_c) = chan<SinglePortRamReq<u32:10, u32:32>>;
    let (resp_p, resp_c) = chan<SinglePortRamResp<u32:32>>;
    spawn SinglePortRamModel<
      u32:32,  // DATA_WIDTH
      u32:1024 // SIZE
    >(req_c, resp_p);
    (req_p, resp_c, terminator)
  }

  next(tok: token, state: ()) {
    let NUM_OFFSETS = u32:30;
    let _ = trace!(NUM_OFFSETS);
    let tok = for (offset, tok): (u32, token) in range(u32:0, NUM_OFFSETS) {
      let _ = trace!(offset);
      // First, write the whole memory.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, SinglePortRamReq {
            addr: addr as uN[10],
            data: (addr + offset) as uN[32],
            we: true,
            re: false,
        });
        tok
      } (tok);

      // Now check that what we wrote is still there.
      let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, SinglePortRamReq {
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
