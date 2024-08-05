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
import std;


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
pub enum SimultaneousReadWriteBehavior : u2 {
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
  assert_eq(u4:0b0000, expand_mask<u32:4>(u2:0b00));
  assert_eq(u4:0b1100, expand_mask<u32:4>(u2:0b10));
  assert_eq(u4:0b0011, expand_mask<u32:4>(u2:0b01));
  assert_eq(u4:0b1111, expand_mask<u32:4>(u2:0b11));

  // Try expanding all 2-bit masks to 3-bit masks, which are the same as the
  // 4-bit masks with the MSB dropped.
  assert_eq(u3:0b000, expand_mask<u32:3>(u2:0b00));
  assert_eq(u3:0b100, expand_mask<u32:3>(u2:0b10));
  assert_eq(u3:0b011, expand_mask<u32:3>(u2:0b01));
  assert_eq(u3:0b111, expand_mask<u32:3>(u2:0b11));

  // Try expanding 2-bit masks to 2-bit masks, which should be identity.
  assert_eq(u2:0b00, expand_mask<u32:2>(u2:0b00));
  assert_eq(u2:0b10, expand_mask<u32:2>(u2:0b10));
  assert_eq(u2:0b01, expand_mask<u32:2>(u2:0b01));
  assert_eq(u2:0b11, expand_mask<u32:2>(u2:0b11));
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
  assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b11),
    (u2:1, [true, true])
  );
  assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b01),
    (u2:1, [true, false])
  );
  assert_eq(
    write_word(u2:0, [false, false], u2:1, u2:0b10),
    (u2:0, [false, true])
  );
  assert_eq(
    write_word(u2:0, [false, true], u2:0, u2:0b01),
    (u2:0, [true, true])
  );
  assert_eq(
    write_word(u2:0, [false], u2:0, u1:0b0),
    (u2:0, [false])
  );
  assert_eq(
    write_word(u2:0, [false], u2:0, u1:0b1),
    (u2:0, [true])
  );
}

// Function to compute num partitions (e.g. mask width) for a data_width-wide
// word divided into word_partition_size-chunks.
pub fn num_partitions(word_partition_size: u32, data_width: u32) -> u32 {
  match word_partition_size {
    u32:0 => u32:0,
    _ => (word_partition_size + data_width - u32:1) / word_partition_size,
  }
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
proc RamModel<DATA_WIDTH:u32, SIZE:u32, WORD_PARTITION_SIZE:u32={u32:0},
  SIMULTANEOUS_READ_WRITE_BEHAVIOR:SimultaneousReadWriteBehavior=
   {SimultaneousReadWriteBehavior::READ_BEFORE_WRITE},
  INITIALIZED:bool={false},
  ASSERT_VALID_READ:bool={true}, ADDR_WIDTH:u32 = {std::clog2(SIZE)},
  NUM_PARTITIONS:u32={num_partitions(WORD_PARTITION_SIZE, DATA_WIDTH)}> {

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

  next(state:(bits[DATA_WIDTH][SIZE], bool[NUM_PARTITIONS][SIZE])) {
    // state consists of an array storing the memory state, as well as an array
    // indicating if the each subword partition has been initialized.
    let (mem, mem_initialized) = state;

    // Perform nonblocking receives on each request channel.
    let zero_read_req = ReadReq<ADDR_WIDTH, NUM_PARTITIONS> {
      addr:bits[ADDR_WIDTH]:0,
      mask:bits[NUM_PARTITIONS]:0,
    };
    let (tok, read_req, read_req_valid) =
      recv_non_blocking(join(), read_req, zero_read_req);
    let zero_write_req = WriteReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS> {
      addr:bits[ADDR_WIDTH]:0,
      data:bits[DATA_WIDTH]:0,
      mask:bits[NUM_PARTITIONS]:0,
    };
    let (tok, write_req, write_req_valid) =
      recv_non_blocking(tok, write_req, zero_write_req);

    // Assert memory being read is initialized by checking that all partitions
    // have been initialized.
    if read_req_valid && ASSERT_VALID_READ {
      let mem_initialized_as_bits =
        std::convert_to_bits_msb0(array_rev(mem_initialized[read_req.addr]));
      assert!(read_req.mask & !mem_initialized_as_bits == uN[NUM_PARTITIONS]:0, "memory_not_initialized")
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
        SimultaneousReadWriteBehavior::ASSERT_NO_CONFLICT => fail!("conflicting_read_and_write", mem[read_req.addr]),
        _ => fail!("impossible_case", uN[DATA_WIDTH]:0),
      }
    } else { mem[read_req.addr] };
    let read_resp_value = ReadResp<DATA_WIDTH> {
      data: unmasked_read_value & expand_mask<DATA_WIDTH>(read_req.mask),
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
  read_req: chan<ReadReq<8, 32>> out;
  read_resp: chan<ReadResp<32>> in;
  write_req: chan<WriteReq<8, 32, 32>> out;
  write_resp: chan<WriteResp> in;

  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (read_req_s, read_req_r) = chan<ReadReq<8, 32>>("read_req");
    let (read_resp_s, read_resp_r) = chan<ReadResp<32>>("read_rest");
    let (write_req_s, write_req_r) = chan<WriteReq<8, 32, 32>>("write_req");
    let (write_resp_s, write_resp_r) = chan<WriteResp>("write_resp");
    spawn RamModel<
      u32:32,  // DATA_WIDTH
      u32:256, // SIZE
      u32:1    // WORD_PARTITION_SIZE
    >(
      read_req_r, read_resp_s, write_req_r, write_resp_s);
    (read_req_s, read_resp_r, write_req_s, write_resp_r, terminator)
  }

  next(state: ()) {
    let NUM_OFFSETS = u32:8;
    let tok = for (offset, tok): (u32, token) in range(u32:0, NUM_OFFSETS) {
      trace!(offset);

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
        assert_eq(read_data.data, (addr + offset) as uN[32]);
        tok
      } (tok);
      tok
    } (join());

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
      assert_eq(read_data.data, expected);
      tok
    } (tok);

    let tok = send(tok, terminator, true);
  }
}

// Tests that a RAM with initialization can be read.
#[test_proc]
proc RamModelInitializationTest {
  read_req: chan<ReadReq<8, 32>> out;
  read_resp: chan<ReadResp<32>> in;

  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (read_req_s, read_req_r) = chan<ReadReq<8, 32>>("read_req");
    let (read_resp_s, read_resp_r) = chan<ReadResp<32>>("read_resp");
    let (_, write_req_r) = chan<WriteReq<8, 32, 32>>("write_req");
    let (write_resp_s, _) = chan<WriteResp>("write_resp");
    spawn RamModel<
      u32:32,   // DATA_WIDTH
      u32:256,  // SIZE
      u32:1,    // WORD_PARTITION_SIZE
      SimultaneousReadWriteBehavior::READ_BEFORE_WRITE,
      // SIMULTANEOUS_READ_WRITE_BEHAVIOR
      true     // INITIALIZED
    >(
      read_req_r, read_resp_s, write_req_r, write_resp_s);
    (read_req_s, read_resp_r, terminator)
  }

  next(state: ()) {
    // Now check that what we wrote is still there.
    let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:256) {
      let tok = send(tok, read_req, ReadWordReq<u32:32>(addr as uN[8]));
      let (tok, read_data) = recv(tok, read_resp);
      assert_eq(read_data.data, u32:0);
      tok
    } (join());

    let tok = send(tok, terminator, true);
  }
}

// Tests that RAM works with partitions larger than 1 bit
#[test_proc]
proc RamModelFourBitMaskReadWriteTest {
  read_req: chan<ReadReq<8, 2>> out;
  read_resp: chan<ReadResp<8>> in;
  write_req: chan<WriteReq<8, 8, 2>> out;
  write_resp: chan<WriteResp> in;

  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (read_req_s, read_req_r) = chan<ReadReq<8, 2>>("read_req");
    let (read_resp_s, read_resp_r) = chan<ReadResp<8>>("read_resp");
    let (write_req_s, write_req_r) = chan<WriteReq<8, 8, 2>>("write_req");
    let (write_resp_s, write_resp_r) = chan<WriteResp>("write_resp");
    spawn RamModel<
      u32:8,   // DATA_WIDTH
      u32:256, // SIZE
      u32:4    // WORD_PARTITION_SIZE
    >(
      read_req_r, read_resp_s, write_req_r, write_resp_s);
    (read_req_s, read_resp_r, write_req_s, write_resp_r, terminator)
  }

  next(state: ()) {
    // Write full words
    let tok = send(join(), write_req, WriteWordReq<u32:2>(
      u8:0,
      u8:0xFF));
    let (tok, _) = recv(tok, write_resp);
    let tok = send(tok, write_req, WriteWordReq<u32:2>(
      u8:1,
      u8:0xBA));
    let (tok, _) = recv(tok, write_resp);

    // Check that full words are written as expected
    let tok = send(tok, read_req, ReadWordReq<u32:2>(u8:0));
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0xFF);
    let tok = send(tok, read_req, ReadWordReq<u32:2>(u8:1));
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0xBA);

    // Write half-words
    let tok = send(tok, write_req, WriteReq{
      addr: u8:0,
      data: u8:0xDE,
      mask: u2:0b10,
      });
    let (tok, _) = recv(tok, write_resp);
    let tok = send(tok, write_req, WriteReq{
      addr: u8:1,
      data: u8:0x78,
      mask: u2:0b01,
      });
    let (tok, _) = recv(tok, write_resp);

    // Check that half-words are written as expected
    let tok = send(tok, read_req, ReadWordReq<u32:2>(u8:0));
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0xDF);
    let tok = send(tok, read_req, ReadWordReq<u32:2>(u8:1));
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0xB8);

    // Read half-words and check the result
    let tok = send(tok, read_req, ReadReq{
      addr: u8:0,
      mask: u2:0b01,
      });
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0x0F);
    let tok = send(tok, read_req, ReadReq{
      addr: u8:1,
      mask: u2:0b10,
      });
    let (tok, read_data) = recv(tok, read_resp);
    assert_eq(read_data.data, u8:0xB0);

    let tok = send(tok, terminator, true);
  }
}

// Single-port RAM request
pub struct RWRamReq<ADDR_WIDTH:u32, DATA_WIDTH:u32, NUM_PARTITIONS:u32> {
  addr: bits[ADDR_WIDTH],
  data: bits[DATA_WIDTH],
  // TODO(google/xls#861): represent masks when we have type generics.
  write_mask: (),
  read_mask: (),
  we: bool,
  re: bool,
}

// Single-port RAM response
pub struct RWRamResp<DATA_WIDTH:u32> {
  data: bits[DATA_WIDTH],
}

// Models a single-port RAM.
proc SinglePortRamModel<DATA_WIDTH:u32, SIZE:u32,
 WORD_PARTITION_SIZE:u32={u32:0},
 ADDR_WIDTH:u32={std::clog2(SIZE)},
 NUM_PARTITIONS:u32={num_partitions(WORD_PARTITION_SIZE, DATA_WIDTH)}> {
  req_chan: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in;
  resp_chan : chan<RWRamResp<DATA_WIDTH>> out;
  wr_comp_chan: chan<()> out;

  init {
      bits[DATA_WIDTH][SIZE]: [bits[DATA_WIDTH]: 0, ...]
  }

  config(req: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in,
         resp: chan<RWRamResp<DATA_WIDTH>> out,
         wr_comp: chan<()> out) {
    (req, resp, wr_comp)
  }

  next(state: bits[DATA_WIDTH][SIZE]) {
    let (tok, request) = recv(join(), req_chan);

    let (response, new_state) = if request.we {
      (
        RWRamResp { data: bits[DATA_WIDTH]:0 },
        update(state, request.addr, request.data),
      )
    } else {
      (
        RWRamResp { data: state[request.addr] },
        state,
      )
    };
    let resp_tok = send_if(tok, resp_chan, request.re, response);
    let wr_comp_tok = send_if(tok, wr_comp_chan, request.we, ());
    new_state
  }
}

// Tests writing various patterns to a single port RAM by reading the contents
// afterwards and checking that you got what you wrote.
#[test_proc]
proc SinglePortRamModelTest {
  req_out: chan<RWRamReq<10, 32, 0>> out;
  resp_in: chan<RWRamResp<32>> in;
  wr_comp_in: chan<()> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (req_s, req_r) = chan<RWRamReq<10, 32, 0>>("req");
    let (resp_s, resp_r) = chan<RWRamResp<32>>("resp");
    let (wr_comp_s, wr_comp_r) = chan<()>("wr_comp");
    spawn SinglePortRamModel<
      u32:32,   // DATA_WIDTH
      u32:1024, // SIZE
      u32:0     // WORD_PARTITION_SIZE
    >(req_r, resp_s, wr_comp_s);
    (req_s, resp_r, wr_comp_r, terminator)
  }

  next(state: ()) {
    let NUM_OFFSETS = u32:30;
    trace!(NUM_OFFSETS);
    let tok = for (offset, tok): (u32, token) in range(u32:0, NUM_OFFSETS) {
      trace!(offset);
      // First, write the whole memory.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, RWRamReq {
            addr: addr as uN[10],
            data: (addr + offset) as uN[32],
            write_mask: (),
            read_mask: (),
            we: true,
            re: false,
        });
        let (tok, _) = recv(tok, wr_comp_in);
        tok
      } (tok);

      // Now check that what we wrote is still there.
      let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req_out, RWRamReq {
          addr: addr as uN[10],
          data: uN[32]: 0,
          write_mask: (),
          read_mask: (),
          we: false,
          re: true,
        });
        let (tok, read_data) = recv(tok, resp_in);
        assert_eq(read_data.data, (addr + offset) as uN[32]);
        tok
      } (tok);
      tok
    } (join());
    let tok = send(tok, terminator, true);
  }
}

// Models a true dual-port RAM.
proc RamModel2RW<DATA_WIDTH:u32, SIZE:u32, WORD_PARTITION_SIZE:u32={u32:0},
  SIMULTANEOUS_READ_WRITE_BEHAVIOR:SimultaneousReadWriteBehavior=
   {SimultaneousReadWriteBehavior::READ_BEFORE_WRITE},
 ADDR_WIDTH:u32={std::clog2(SIZE)},
 NUM_PARTITIONS:u32={num_partitions(WORD_PARTITION_SIZE, DATA_WIDTH)}> {
  req_chan0: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in;
  req_chan1: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in;
  resp_chan0: chan<RWRamResp<DATA_WIDTH>> out;
  resp_chan1: chan<RWRamResp<DATA_WIDTH>> out;
  wr_comp_chan0: chan<()> out;
  wr_comp_chan1: chan<()> out;

  init {
      bits[DATA_WIDTH][SIZE]: [bits[DATA_WIDTH]: 0, ...]
  }

  config(req0: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in,
         resp0: chan<RWRamResp<DATA_WIDTH>> out,
         wr_comp0: chan<()> out,
         req1: chan<RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>> in,
         resp1: chan<RWRamResp<DATA_WIDTH>> out,
         wr_comp1: chan<()> out) {
    (req0, req1, resp0, resp1, wr_comp0, wr_comp1)
  }

  next(state: bits[DATA_WIDTH][SIZE]) {
    let (tok0, request0, valid0) =
      recv_non_blocking(join(), req_chan0, zero!<RWRamReq>());
    let (tok1, request1, valid1) =
      recv_non_blocking(join(), req_chan1, zero!<RWRamReq>());
    let fatal_hazard = valid0 && valid1 && request0.addr == request1.addr &&
      match SIMULTANEOUS_READ_WRITE_BEHAVIOR {
        SimultaneousReadWriteBehavior::ASSERT_NO_CONFLICT =>
          request0.we || request0.we,
        // Unless we're asserting no conflict, it's only an error if we write to
        // the same address with both ports.
        _ => request0.we && request1.we,
      };
    assert!(!fatal_hazard, "dual_port_memory_hazard");

    // Save state in case we are READ_BEFORE_WRITE.
    let state_before_write = state;

    // Do writes.
    let state = if valid0 && request0.we {
      update(state, request0.addr, request0.data)
    } else {
      state
    };
    let state = if valid1 && request1.we {
      update(state, request1.addr, request1.data)
    } else {
      state
    };

    let response0 =
     match (valid0 && request0.re, SIMULTANEOUS_READ_WRITE_BEHAVIOR) {
      (bool:1, SimultaneousReadWriteBehavior::READ_BEFORE_WRITE) =>
        RWRamResp { data: state_before_write[request0.addr] },
      (bool:1, _) => RWRamResp { data: state[request0.addr] },
      _ => RWRamResp { data: bits[DATA_WIDTH]:0 },
    };
    let response1 =
     match (valid1 && request1.re, SIMULTANEOUS_READ_WRITE_BEHAVIOR) {
      (bool:1, SimultaneousReadWriteBehavior::READ_BEFORE_WRITE) =>
        RWRamResp { data: state_before_write[request1.addr] },
      (bool:1, _) => RWRamResp { data: state[request1.addr] },
      _ => RWRamResp { data: bits[DATA_WIDTH]:0 },
    };

    let resp0_tok = send_if(tok0, resp_chan0, valid0 && request0.re, response0);
    let resp1_tok = send_if(tok1, resp_chan1, valid1 && request1.re, response1);
    let wr_comp0_tok = send_if(tok0, wr_comp_chan0, valid0 && request0.we, ());
    let wr_comp1_tok = send_if(tok1, wr_comp_chan1, valid1 && request1.we, ());
    state
  }
}

// Tests writing various patterns to a single port RAM by reading the contents
// afterwards and checking that you got what you wrote.
#[test_proc]
proc RamModel2RWTest {
  req0_out: chan<RWRamReq<10, 32, 0>> out;
  resp0_in: chan<RWRamResp<32>> in;
  wr_comp0_in: chan<()> in;
  req1_out: chan<RWRamReq<10, 32, 0>> out;
  resp1_in: chan<RWRamResp<32>> in;
  wr_comp1_in: chan<()> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (req0_s, req0_r) = chan<RWRamReq<10, 32, 0>>("req0");
    let (resp0_s, resp0_r) = chan<RWRamResp<32>>("resp0");
    let (wr_comp0_s, wr_comp0_r) = chan<()>("wr_comp0");
    let (req1_s, req1_r) = chan<RWRamReq<10, 32, 0>>("req1");
    let (resp1_s, resp1_r) = chan<RWRamResp<32>>("resp1");
    let (wr_comp1_s, wr_comp1_r) = chan<()>("wr_comp");
    spawn RamModel2RW<
      u32:32,   // DATA_WIDTH
      u32:1024, // SIZE
      u32:0     // WORD_PARTITION_SIZE
    >(req0_r, resp0_s, wr_comp0_s, req1_r, resp1_s, wr_comp1_s);
    (req0_s, resp0_r, wr_comp0_r, req1_s, resp1_r, wr_comp1_r, terminator)
  }

  next(state: ()) {
    let NUM_OFFSETS = u32:30;
    trace!(NUM_OFFSETS);
    let tok = for (offset, tok): (u32, token) in range(u32:0, NUM_OFFSETS) {
      let tok = trace_fmt!("offset = {}", offset);
      // First, write the whole memory.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req0_out, RWRamReq {
            addr: addr as uN[10],
            data: (addr + offset) as uN[32],
            write_mask: (),
            read_mask: (),
            we: true,
            re: false,
        });
        let (tok, _) = recv(tok, wr_comp0_in);
        tok
      } (tok);

      // Now check that what we wrote is still there.
      let tok = for (addr, tok) : (u32, token) in range(u32:0, u32:1024) {
        let tok = send(tok, req1_out, RWRamReq {
          addr: addr as uN[10],
          data: uN[32]: 0,
          write_mask: (),
          read_mask: (),
          we: false,
          re: true,
        });
        let (tok, read_data) = recv(tok, resp1_in);
        assert_eq(read_data.data, (addr + offset) as uN[32]);
        tok
      } (tok);

      // Now, write addr and read addr-1 simultaneously.
      let tok = for (addr, tok): (u32, token) in range(u32:0, u32:1024) {
        // write addr
        let write_tok = send(tok, req0_out, RWRamReq {
            addr: addr as uN[10],
            data: (addr - offset) as uN[32],
            write_mask: (),
            read_mask: (),
            we: true,
            re: false,
        });
        let read_tok = send_if(tok, req1_out, addr > u32:0, RWRamReq {
          addr: (addr as u10) - u10:1,
          data: uN[32]:0,
          write_mask: (),
          read_mask: (),
          we: false,
          re: true,
        });
        let (write_tok, _) = recv(write_tok, wr_comp0_in);
        let (read_tok, read_data) = recv_if(read_tok, resp1_in, addr > u32:0,
                                            RWRamResp<32> { data:-u32:1-offset});
        assert_eq(read_data.data, addr - u32:1 - offset);
        join(write_tok, read_tok)
      } (tok);

      tok
    } (join());
    let tok = send(tok, terminator, true);
  }
}
