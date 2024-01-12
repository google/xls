import std;
import xls.examples.ram;

enum RamPrinterStatus : u2 {
    IDLE = 0,
    BUSY = 1,
}

struct RamPrinterState<ADDR_WIDTH: u32> { status: RamPrinterStatus, addr: bits[ADDR_WIDTH] }

proc RamPrinter<DATA_WIDTH: u32, SIZE: u32, NUM_PARTITIONS: u32, ADDR_WIDTH: u32, NUM_MEMORIES: u32>
{
    print_r: chan<()> in;
    finish_s: chan<()> out;
    req_s: chan<ram::RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>>[NUM_MEMORIES] out;
    resp_r: chan<ram::RWRamResp<DATA_WIDTH>>[NUM_MEMORIES] in;

    config(
        print_r: chan<()> in,
        finish_s: chan<()> out,
        req_s: chan<ram::RWRamReq<ADDR_WIDTH, DATA_WIDTH, NUM_PARTITIONS>>[NUM_MEMORIES] out,
        resp_r: chan<ram::RWRamResp<DATA_WIDTH>>[NUM_MEMORIES] in
    ) { (print_r, finish_s, req_s, resp_r) }

    init {
        RamPrinterState {
            status: RamPrinterStatus::IDLE,
            addr: bits[ADDR_WIDTH]:0
        }
    }

    next(tok: token, state: RamPrinterState) {

        let is_idle = state.status == RamPrinterStatus::IDLE;
        let (tok, _) = recv_if(tok, print_r, is_idle, ());

        let (tok, row) = for (i, (tok, row)): (u32, (token, bits[DATA_WIDTH][NUM_MEMORIES])) in
            range(u32:0, NUM_MEMORIES) {
            let tok = send(
                tok, req_s[i],
                ram::RWRamReq {
                    addr: state.addr,
                    data: bits[DATA_WIDTH]:0,
                    write_mask: (),
                    read_mask: (),
                    we: false,
                    re: true
                });
            let (tok, resp) = recv(tok, resp_r[i]);
            let row = update(row, i, resp.data);
            (tok, row)
        }((tok, bits[DATA_WIDTH][NUM_MEMORIES]:[bits[DATA_WIDTH]:0, ...]));

        let is_start = state.addr == bits[ADDR_WIDTH]:0;
        let is_last = state.addr == (SIZE - u32:1) as bits[ADDR_WIDTH];

        if is_start { trace_fmt!(" ========= RAM content ========= ", ); } else {  };

        trace_fmt!(" {}:\t{:x} ", state.addr, row);
        let tok = send_if(tok, finish_s, is_last, ());

        if is_last {
            RamPrinterState {
                addr: bits[ADDR_WIDTH]:0,
                status: RamPrinterStatus::IDLE
            }
        } else {
            RamPrinterState {
                addr: state.addr + bits[ADDR_WIDTH]:1,
                status: RamPrinterStatus::BUSY
            }
        }
    }
}

const TEST_NUM_MEMORIES = u32:8;
const TEST_SIZE = u32:10;
const TEST_DATA_WIDTH = u32:8;
const TEST_WORD_PARTITION_SIZE = u32:0;
const TEST_NUM_PARTITIONS = ram::num_partitions(TEST_WORD_PARTITION_SIZE, TEST_DATA_WIDTH);
const TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR = ram::SimultaneousReadWriteBehavior::READ_BEFORE_WRITE;
const TEST_ADDR_WIDTH = std::clog2(TEST_SIZE);

#[test_proc]
proc RamPrinterTest {
    terminator: chan<bool> out;
    req0_s: chan<ram::RWRamReq<TEST_ADDR_WIDTH, TEST_DATA_WIDTH, TEST_NUM_PARTITIONS>>[TEST_NUM_MEMORIES] out;
    resp0_r: chan<ram::RWRamResp<TEST_DATA_WIDTH>>[TEST_NUM_MEMORIES] in;
    wr_comp0_r: chan<()>[TEST_NUM_MEMORIES] in;
    req1_s: chan<ram::RWRamReq<TEST_ADDR_WIDTH, TEST_DATA_WIDTH, TEST_NUM_PARTITIONS>>[TEST_NUM_MEMORIES] out;
    resp1_r: chan<ram::RWRamResp<TEST_DATA_WIDTH>>[TEST_NUM_MEMORIES] in;
    wr_comp1_r: chan<()>[TEST_NUM_MEMORIES] in;
    print_s: chan<()> out;
    finish_r: chan<()> in;

    config(terminator: chan<bool> out) {
        let (req0_s, req0_r) = chan<ram::RWRamReq<TEST_ADDR_WIDTH, TEST_DATA_WIDTH, TEST_NUM_PARTITIONS>>[TEST_NUM_MEMORIES];
        let (resp0_s, resp0_r) = chan<ram::RWRamResp<TEST_DATA_WIDTH>>[TEST_NUM_MEMORIES];
        let (wr_comp0_s, wr_comp0_r) = chan<()>[TEST_NUM_MEMORIES];
        let (req1_s, req1_r) = chan<ram::RWRamReq<TEST_ADDR_WIDTH, TEST_DATA_WIDTH, TEST_NUM_PARTITIONS>>[TEST_NUM_MEMORIES];
        let (resp1_s, resp1_r) = chan<ram::RWRamResp<TEST_DATA_WIDTH>>[TEST_NUM_MEMORIES];
        let (wr_comp1_s, wr_comp1_r) = chan<()>[TEST_NUM_MEMORIES];

        let (print_s, print_r) = chan<()>;
        let (finish_s, finish_r) = chan<()>;

        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[0], resp0_s[0], wr_comp0_s[0], req1_r[0], resp1_s[0], wr_comp1_s[0]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[1], resp0_s[1], wr_comp0_s[1], req1_r[1], resp1_s[1], wr_comp1_s[1]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[2], resp0_s[2], wr_comp0_s[2], req1_r[2], resp1_s[2], wr_comp1_s[2]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[3], resp0_s[3], wr_comp0_s[3], req1_r[3], resp1_s[3], wr_comp1_s[3]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[4], resp0_s[4], wr_comp0_s[4], req1_r[4], resp1_s[4], wr_comp1_s[4]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[5], resp0_s[5], wr_comp0_s[5], req1_r[5], resp1_s[5], wr_comp1_s[5]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[6], resp0_s[6], wr_comp0_s[6], req1_r[6], resp1_s[6], wr_comp1_s[6]);
        spawn ram::RamModel2RW<TEST_DATA_WIDTH, TEST_SIZE, TEST_WORD_PARTITION_SIZE, TEST_SIMULTANEOUS_READ_WRITE_BEHAVIOR>
            (req0_r[7], resp0_s[7], wr_comp0_s[7], req1_r[7], resp1_s[7], wr_comp1_s[7]);

        spawn RamPrinter<TEST_DATA_WIDTH, TEST_SIZE, TEST_NUM_PARTITIONS, TEST_ADDR_WIDTH, TEST_NUM_MEMORIES>
            (print_r, finish_s, req0_s, resp0_r);

        (terminator, req0_s, resp0_r, wr_comp0_r, req1_s, resp1_r, wr_comp1_r, print_s, finish_r)
    }

    init { }

    next(tok: token, state: ()) {
        let tok = send(tok, print_s, ());
        let (tok, _) = recv(tok, finish_r);
        let tok = send(tok, terminator, true);
    }
}
