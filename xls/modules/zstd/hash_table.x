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

import std;

import xls.examples.ram;

pub struct HashTableReadReq<
    KEY_W: u32, SIZE: u32,
    SIZE_W: u32 = {std::clog2(SIZE + u32:1)}
> {
    num_entries_log2: uN[SIZE_W],  // number of HashTable entries used in the runtime
    key: uN[KEY_W],
}

pub struct HashTableReadResp<VALUE_W: u32> {
    is_match: bool,
    value: uN[VALUE_W]
}

pub struct HashTableWriteReq<
    KEY_W: u32, VALUE_W: u32, SIZE: u32,
    SIZE_W:u32 = { std::clog2(SIZE + u32:1)}
> {
    num_entries_log2: uN[SIZE_W],  // number of HashTable entries used in the runtime
    key: uN[KEY_W],
    value: uN[VALUE_W],
}

pub struct HashTableWriteResp {}

fn knuth_hash_slow<CONSTANT: u32, KEY_W: u32, HASH_W: u32>(key: uN[KEY_W]) -> uN[HASH_W] {
    (((key * CONSTANT) as u32) >> (u32:32 - HASH_W)) as uN[HASH_W]
}

fn knuth_hash<CONSTANT: u32, KEY_W: u32, HASH_W: u32>(key: uN[KEY_W]) -> uN[HASH_W] {
    let result = for (i, result): (u32, uN[KEY_W]) in range(u32:0, u32:32) {
        if (CONSTANT >> i) as u1 { result + (key << i) } else { result }
    }(uN[KEY_W]:0);

    (result >> (u32:32 - HASH_W)) as uN[HASH_W]
}

#[test]
fn knuth_hash_check() {
    const KNUTH_CONSTANT = u32:0x1e35a7bd;
    const HASH_W = u32:32;

    for (i, ()) in range(u32:0, u32:1 << u32:7) {
        let hash_slow = knuth_hash_slow<KNUTH_CONSTANT, u32:32, HASH_W>(i);
        let hash_fast = knuth_hash<KNUTH_CONSTANT, u32:32, HASH_W>(i);
        assert_eq(hash_slow, hash_fast);
    }(());
}

struct RamData<VALUE_W: u32> {
    value: uN[VALUE_W],
    valid: bool,
}

proc HashTableReadReqHandler<
    KEY_W: u32, VALUE_W: u32, SIZE: u32, KNUTH_CONSTANT: u32,
    HASH_W: u32 = {std::clog2(SIZE)},
    RAM_DATA_W: u32 = {VALUE_W + u32:1},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(u32:1, RAM_DATA_W)}
> {
    type ReadReq = HashTableReadReq<KEY_W, SIZE>;
    type RamReadReq = ram::ReadReq<HASH_W, RAM_NUM_PARTITIONS>;

    read_req_r: chan<ReadReq> in;
    ram_read_req_s: chan<RamReadReq> out;

    config(
        read_req_r: chan<ReadReq> in,
        ram_read_req_s: chan<RamReadReq> out
    ) {
        (read_req_r, ram_read_req_s)
    }

    init { }

    next(state: ()) {
        let tok = join();

        let (tok_read, read_req, read_req_valid) =
            recv_non_blocking(tok, read_req_r, zero!<ReadReq>());

        let ram_read_req = if read_req_valid {
            let hash_mask = (uN[HASH_W]:1 << read_req.num_entries_log2) - uN[HASH_W]:1;
            let hash = knuth_hash<KNUTH_CONSTANT, KEY_W, HASH_W>(read_req.key) & hash_mask;
            RamReadReq { addr: hash, mask: !uN[RAM_NUM_PARTITIONS]:0 }
        } else {
            zero!<RamReadReq>()
        };

        send_if(tok_read, ram_read_req_s, read_req_valid, ram_read_req);
    }
}

proc HashTableReadRespHandler<
    VALUE_W: u32,
    RAM_DATA_W: u32 = {VALUE_W + u32:1} // value width + data valid width,
> {
    type RamReadResp = ram::ReadResp<RAM_DATA_W>;
    type ReadResp = HashTableReadResp<VALUE_W>;

    ram_read_resp_r: chan<RamReadResp> in;
    read_resp_s: chan<ReadResp> out;

    config(
        ram_read_resp_r: chan<RamReadResp> in,
        read_resp_s: chan<ReadResp> out
    ) {
        (ram_read_resp_r, read_resp_s)
    }

    init {  }

    next(state: ()) {
        let tok = join();

        let (tok, ram_read_resp, ram_read_resp_valid) =
            recv_non_blocking(tok, ram_read_resp_r, zero!<RamReadResp>());

        let read_resp = if ram_read_resp_valid {
            let ram_data = RamData<VALUE_W> {
                value: (ram_read_resp.data >> u32:1) as uN[VALUE_W],
                valid: ram_read_resp.data as u1,
            };
            ReadResp {
                is_match: ram_data.valid,
                value: ram_data.value
            }
        } else {
            zero!<ReadResp>()
        };

        send_if(tok, read_resp_s, ram_read_resp_valid, read_resp);
    }
}

proc HashTableWriteReqHandler<
    KEY_W: u32, VALUE_W: u32, SIZE: u32, KNUTH_CONSTANT: u32,
    HASH_W: u32 = {std::clog2(SIZE)},
    RAM_DATA_W: u32 = {VALUE_W + u32:1},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(u32:1, RAM_DATA_W)}
> {
    type WriteReq = HashTableWriteReq<KEY_W, VALUE_W, SIZE>;
    type RamWriteReq = ram::WriteReq<HASH_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;

    write_req_r: chan<WriteReq> in;
    ram_write_req_s: chan<RamWriteReq> out;

    config(
        write_req_r: chan<WriteReq> in,
        ram_write_req_s: chan<RamWriteReq> out
    ) {
        (write_req_r, ram_write_req_s)
    }

    init {  }

    next(state: ()) {
        let tok = join();

        let (tok_write, write_req, write_req_valid) =
            recv_non_blocking(tok, write_req_r, zero!<WriteReq>());

        let ram_write_req = if write_req_valid {
            let hash_mask = (uN[HASH_W]:1 << write_req.num_entries_log2) - uN[HASH_W]:1;
            let hash = knuth_hash<KNUTH_CONSTANT, KEY_W, HASH_W>(write_req.key) & hash_mask;
            let data = write_req.value ++ true;
            RamWriteReq { addr: hash, data, mask: !uN[RAM_NUM_PARTITIONS]:0 }
        } else {
            zero!<RamWriteReq>()
        };

        send_if(tok_write, ram_write_req_s, write_req_valid, ram_write_req);
    }
}

proc HashTableWriteRespHandler {
    type RamWriteResp = ram::WriteResp;
    type WriteResp = HashTableWriteResp;

    ram_write_resp_r: chan<RamWriteResp> in;
    write_resp_s: chan<WriteResp> out;

    config(
        ram_write_resp_r: chan<RamWriteResp> in,
        write_resp_s: chan<WriteResp> out
    ) {
        (ram_write_resp_r, write_resp_s)
    }

    init { }

    next(state: ()) {
        let tok = join();

        let (tok, _, ram_write_resp_valid) =
            recv_non_blocking(tok, ram_write_resp_r, zero!<RamWriteResp>());

        send_if(tok, write_resp_s, ram_write_resp_valid, WriteResp {});
    }
}

pub proc HashTable<
    KEY_W: u32, VALUE_W: u32, SIZE: u32,
    HASH_W: u32 = {std::clog2(SIZE)},
    KNUTH_CONSTANT: u32 = {u32:0x1e35a7bd},
    RAM_DATA_W: u32 = {VALUE_W + u32:1},
    RAM_NUM_PARTITIONS: u32 = {ram::num_partitions(u32:1, RAM_DATA_W)}
> {
    type ReadReq = HashTableReadReq<KEY_W, SIZE>;
    type ReadResp = HashTableReadResp<VALUE_W>;
    type WriteReq = HashTableWriteReq<KEY_W, VALUE_W, SIZE>;
    type WriteResp = HashTableWriteResp;

    type RamReadReq = ram::ReadReq<HASH_W, RAM_NUM_PARTITIONS>;
    type RamReadResp = ram::ReadResp<RAM_DATA_W>;
    type RamWriteReq = ram::WriteReq<HASH_W, RAM_DATA_W, RAM_NUM_PARTITIONS>;
    type RamWriteResp = ram::WriteResp;

    config(
        read_req_r: chan<ReadReq> in,
        read_resp_s: chan<ReadResp> out,
        write_req_r: chan<WriteReq> in,
        write_resp_s: chan<WriteResp> out,
        ram_read_req_s: chan<RamReadReq> out,
        ram_read_resp_r: chan<RamReadResp> in,
        ram_write_req_s: chan<RamWriteReq> out,
        ram_write_resp_r: chan<RamWriteResp> in
    ) {
        spawn HashTableReadReqHandler<KEY_W, VALUE_W, SIZE, KNUTH_CONSTANT>(
            read_req_r, ram_read_req_s
        );

        spawn HashTableReadRespHandler<VALUE_W>(
            ram_read_resp_r, read_resp_s
        );
        spawn HashTableWriteReqHandler<KEY_W, VALUE_W, SIZE, KNUTH_CONSTANT>(
            write_req_r, ram_write_req_s
        );
        spawn HashTableWriteRespHandler(
            ram_write_resp_r, write_resp_s
        );
    }

    init { }

    next(state: ()) { }
}

const INST_KEY_W = u32:32;
const INST_VALUE_W = u32:32;
const INST_SIZE = u32:512;
const INST_HASH_W = std::clog2(INST_SIZE);
const INST_RAM_DATA_W = INST_VALUE_W + u32:1;
const INST_RAM_NUM_PARTITIONS = ram::num_partitions(u32:1, INST_RAM_DATA_W);

proc HashTableInst {
    type InstReadReq = HashTableReadReq<INST_KEY_W, INST_SIZE>;
    type InstReadResp = HashTableReadResp<INST_VALUE_W>;
    type InstWriteReq = HashTableWriteReq<INST_KEY_W, INST_VALUE_W, INST_SIZE>;
    type InstWriteResp = HashTableWriteResp;

    type InstRamReadReq = ram::ReadReq<INST_HASH_W, INST_RAM_NUM_PARTITIONS>;
    type InstRamReadResp = ram::ReadResp<INST_RAM_DATA_W>;
    type InstRamWriteReq = ram::WriteReq<INST_HASH_W, INST_RAM_DATA_W, INST_RAM_NUM_PARTITIONS>;
    type InstRamWriteResp = ram::WriteResp;

    config(
        read_req_r: chan<InstReadReq> in,
        read_resp_s: chan<InstReadResp> out,
        write_req_r: chan<InstWriteReq> in,
        write_resp_s: chan<InstWriteResp> out,
        ram_read_req_s: chan<InstRamReadReq> out,
        ram_read_resp_r: chan<InstRamReadResp> in,
        ram_write_req_s: chan<InstRamWriteReq> out,
        ram_write_resp_r: chan<InstRamWriteResp> in
    ) {
        spawn HashTable<INST_KEY_W, INST_VALUE_W, INST_SIZE>(
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r
        );
    }

    init {  }

    next(state: ()) { }
}

const TEST_KEY_W = u32:32;
const TEST_VALUE_W = u32:32;
const TEST_SIZE = u32:512;
const TEST_SIZE_W = std::clog2(TEST_SIZE + u32:1);
const TEST_HASH_W = std::clog2(TEST_SIZE);
const TEST_RAM_DATA_W = TEST_VALUE_W + u32:1;
const TEST_WORD_PARTITION_SIZE = u32:1;
const TEST_RAM_NUM_PARTITIONS = ram::num_partitions(TEST_WORD_PARTITION_SIZE, TEST_RAM_DATA_W);
const TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_INITIALIZED = true;

type TestReadReq = HashTableReadReq<TEST_KEY_W, TEST_SIZE>;
type TestReadResp = HashTableReadResp<TEST_VALUE_W>;
type TestWriteReq = HashTableWriteReq<TEST_KEY_W, TEST_VALUE_W, TEST_SIZE>;
type TestWriteResp = HashTableWriteResp;

type TestRamReadReq = ram::ReadReq<TEST_HASH_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamReadResp = ram::ReadResp<TEST_RAM_DATA_W>;
type TestRamWriteReq = ram::WriteReq<TEST_HASH_W, TEST_RAM_DATA_W, TEST_RAM_NUM_PARTITIONS>;
type TestRamWriteResp = ram::WriteResp;

struct TestData {
    num_entries_log2: uN[TEST_SIZE_W],
    key: uN[TEST_KEY_W],
    value: uN[TEST_VALUE_W]
}

const TEST_DATA = TestData[32]:[
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0x6109d84c, value: uN[TEST_VALUE_W]:0xdb370dd7},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:7, key: uN[TEST_KEY_W]:0xe773dc7f, value: uN[TEST_VALUE_W]:0xc8f9f817},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:8, key: uN[TEST_KEY_W]:0xd2254d4a, value: uN[TEST_VALUE_W]:0xa0b4c4bd},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0x4c794548, value: uN[TEST_VALUE_W]:0x8a3e6693},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:3, key: uN[TEST_KEY_W]:0xed1884be, value: uN[TEST_VALUE_W]:0x1787d635},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:5, key: uN[TEST_KEY_W]:0x6c40cc5d, value: uN[TEST_VALUE_W]:0x1e0916a3},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0xa7ad798c, value: uN[TEST_VALUE_W]:0x6efa1a96},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:3, key: uN[TEST_KEY_W]:0x8e3bb720, value: uN[TEST_VALUE_W]:0x6d0a7d57},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0xbf9f7bd4, value: uN[TEST_VALUE_W]:0x46ff026c},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:3, key: uN[TEST_KEY_W]:0xd8c1cd03, value: uN[TEST_VALUE_W]:0xdb5b0ded},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0xd1b33035, value: uN[TEST_VALUE_W]:0x7a21e0ed},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:5, key: uN[TEST_KEY_W]:0x8d512e0c, value: uN[TEST_VALUE_W]:0x708a536b},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0x1a950036, value: uN[TEST_VALUE_W]:0x9097f883},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:3, key: uN[TEST_KEY_W]:0x00707a86, value: uN[TEST_VALUE_W]:0xbcb29fa7},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0x2fcd78a1, value: uN[TEST_VALUE_W]:0x71bae380},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:8, key: uN[TEST_KEY_W]:0x34d8adc5, value: uN[TEST_VALUE_W]:0xdff20f62},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:5, key: uN[TEST_KEY_W]:0xd04ebdda, value: uN[TEST_VALUE_W]:0x9c785523},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:5, key: uN[TEST_KEY_W]:0x9b419a1a, value: uN[TEST_VALUE_W]:0xf1d27361},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0x9eb7784d, value: uN[TEST_VALUE_W]:0x58a9d8f2},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:7, key: uN[TEST_KEY_W]:0x6d7499ef, value: uN[TEST_VALUE_W]:0x40387b18},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:8, key: uN[TEST_KEY_W]:0xb255d705, value: uN[TEST_VALUE_W]:0x73ecbb7b},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:8, key: uN[TEST_KEY_W]:0x132c9499, value: uN[TEST_VALUE_W]:0x48b85084},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0xd3acf006, value: uN[TEST_VALUE_W]:0xbbd2f2b9},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:2, key: uN[TEST_KEY_W]:0x0dd951cd, value: uN[TEST_VALUE_W]:0x975ab3fe},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:6, key: uN[TEST_KEY_W]:0x3d6cd6b1, value: uN[TEST_VALUE_W]:0xe18f2e83},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:5, key: uN[TEST_KEY_W]:0xf511fadb, value: uN[TEST_VALUE_W]:0xb99e2ab4},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:2, key: uN[TEST_KEY_W]:0x90bea2bb, value: uN[TEST_VALUE_W]:0xc88b54c2},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0xf2513572, value: uN[TEST_VALUE_W]:0x42ef67d9},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0x2dd80b55, value: uN[TEST_VALUE_W]:0x3b399d05},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0x823af460, value: uN[TEST_VALUE_W]:0x89d154ba},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:9, key: uN[TEST_KEY_W]:0x8ab8897e, value: uN[TEST_VALUE_W]:0xb30eb8c5},
    TestData {num_entries_log2: uN[TEST_SIZE_W]:2, key: uN[TEST_KEY_W]:0xe9499524, value: uN[TEST_VALUE_W]:0xb4a30d68},
];


#[test_proc]
proc HashTable_test {
    terminator_s: chan<bool> out;
    read_req_s: chan<TestReadReq> out;
    read_resp_r: chan<TestReadResp> in;
    write_req_s: chan<TestWriteReq> out;
    write_resp_r: chan<TestWriteResp> in;

    config(terminator_s: chan<bool> out) {
        let (read_req_s, read_req_r) = chan<TestReadReq>("read_req");
        let (read_resp_s, read_resp_r) = chan<TestReadResp>("read_resp");
        let (write_req_s, write_req_r) = chan<TestWriteReq>("write_req");
        let (write_resp_s, write_resp_r) = chan<TestWriteResp>("write_resp");

        let (ram_read_req_s, ram_read_req_r) = chan<TestRamReadReq>("ram_read_req");
        let (ram_read_resp_s, ram_read_resp_r) = chan<TestRamReadResp>("ram_read_resp");
        let (ram_write_req_s, ram_write_req_r) = chan<TestRamWriteReq>("ram_write_req");
        let (ram_write_resp_s, ram_write_resp_r) = chan<TestRamWriteResp>("ram_write_resp");

        spawn ram::RamModel<
            TEST_RAM_DATA_W, TEST_SIZE, TEST_WORD_PARTITION_SIZE,
            TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR, TEST_INITIALIZED
        >(ram_read_req_r, ram_read_resp_s, ram_write_req_r, ram_write_resp_s);

        spawn HashTable<TEST_KEY_W, TEST_VALUE_W, TEST_SIZE>(
            read_req_r, read_resp_s,
            write_req_r, write_resp_s,
            ram_read_req_s, ram_read_resp_r,
            ram_write_req_s, ram_write_resp_r
        );

        (
            terminator_s,
            read_req_s, read_resp_r,
            write_req_s, write_resp_r
        )
    }

    init { }

    next(state: ()) {
        let tok = join();

        let tok = for ((i, test_data), tok): ((u32, TestData), token) in enumerate(TEST_DATA) {
            // try to read data that was not written
            let read_req = TestReadReq {
                num_entries_log2: test_data.num_entries_log2,
                key: test_data.key
            };
            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{}.1 read request {:#x}", i + u32:1, read_req);

            let (tok, read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{}.1 read response {:#x}", i + u32:1, read_resp);
            if read_resp.is_match {
                // there may be match in case of a conflict
                assert_eq(false, test_data.value == read_resp.value);
            } else { };

            // write data
            let write_req = TestWriteReq {
                num_entries_log2: test_data.num_entries_log2,
                key: test_data.key,
                value: test_data.value,
            };
            let tok = send(tok, write_req_s, write_req);
            trace_fmt!("Sent #{} write request {:#x}", i + u32:1, write_req);

            let (tok, write_resp) = recv(tok, write_resp_r);
            trace_fmt!("Received #{} write response {:#x}", i + u32:1, write_resp);

            // read data after it was written
            let read_req = TestReadReq {
                num_entries_log2: test_data.num_entries_log2,
                key: test_data.key
            };
            let tok = send(tok, read_req_s, read_req);
            trace_fmt!("Sent #{}.2 read request {:#x}", i + u32:1, read_req);

            let (tok, read_resp) = recv(tok, read_resp_r);
            trace_fmt!("Received #{}.2 read response {:#x}", i + u32:1, read_resp);
            assert_eq(TestReadResp { is_match: true, value: test_data.value }, read_resp);

            tok
        }(tok);

        send(tok, terminator_s, true);
    }
}
