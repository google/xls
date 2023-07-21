# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Event

from cocotb_bus.scoreboard import Scoreboard
from cocotb_xls import XLSChannelDriver, XLSChannelMonitor

from dbe import TokMK, TokCP, TokLT, Mark, Token, PlainData
from dbe_common_test import DslxPlainData8, DslxToken

from typing import Sequence


@cocotb.coroutine
async def reset(clk, rst, cycles=1):
    await RisingEdge(clk)
    rst.value = 1
    await ClockCycles(clk, cycles)
    rst.value = 0


async def do_comparison_test(
        dut,
        test_in: Sequence[Token], expected_out: Sequence[PlainData],
        post_watch_cycles: int = 100):
    """
    Tests the module by feeding it with `test_in`, reading the output,
    and comparing the output with `expected_out`.

    If the test receives a data item that does not match the one which
    is expected according to the `expected_out`, it terminates with a failure
    immediately.
    If the test receives all data items specified by `expected_out` after
    feeding all the `test_in` data, it continues to run for `post_watch_cycles`
    clock cycles, checking if it will receive any "garbage" data. If it does,
    that is also considered a failure. If it doesn't, then that'll be a pass.
    """
    # Convert i/o arrays to BinaryData
    test_in_bd = list(DslxToken.from_token(t).as_binary() for t in test_in)
    expected_out_bd = list(
        DslxPlainData8.from_plain_data(b).as_binary() for b in expected_out)

    all_data_received = Event("All expected data received")

    clock = Clock(dut.clk, 1, units="us")

    i_token = XLSChannelDriver(dut, "i_token", dut.clk)
    o_data = XLSChannelMonitor(dut, "o_data", dut.clk)

    sb = Scoreboard(dut, fail_immediately=True)
    sb.add_interface(o_data, expected_out_bd)

    def check_if_all_data_received(_):
        if o_data.stats.received_transactions == len(expected_out):
            all_data_received.set()

    o_data.add_callback(check_if_all_data_received)
    o_data.bus.rdy.setimmediatevalue(1)

    await cocotb.start(clock.start())
    await reset(dut.clk, dut.rst, 1)
    await i_token.write(test_in_bd)
    if expected_out:
        await all_data_received.wait()
    await ClockCycles(dut.clk, post_watch_cycles)


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_void(dut):
    """
    Check that in absence of input the module does not generate any output
    """
    await do_comparison_test(
        dut,
        test_in=[
        ],
        expected_out=[
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_lt1(dut):
    """
    Check that a single LT produces one symbol
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
        ],
        expected_out=[
            10,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_lt2(dut):
    """
    Check that a few LTs produce the same number of symbols
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokLT(20),
            TokLT(30),
        ],
        expected_out=[
            10,
            20,
            30,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_cp1(dut):
    """
    Check that a CP with length 1 works correctly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokCP(0, 1)
        ],
        expected_out=[
            10,
            10,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_cp2(dut):
    """
    Check that a CP with length >1 works correctly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokLT(20),
            TokLT(30),
            TokLT(40),
            TokCP(2, 3)
        ],
        expected_out=[
            10,
            20,
            30,
            40,
            20, 30, 40,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_cp3(dut):
    """
    Check that CP can properly repeat symbols produced by itself
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokLT(20),
            TokCP(1, 6)
        ],
        expected_out=[
            10,
            20,
            10, 20, 10, 20, 10, 20
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_cp4(dut):
    """
    Check that decoder handles several CP tokens correctly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokLT(20),
            TokCP(1, 2),
            TokCP(2, 4),
        ],
        expected_out=[
            10,
            20,
            10, 20,
            20, 10, 20, 20
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_cp4(dut):
    """
    Check that decoder handles mixed CP and LT tokens
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokCP(0, 1),
            TokLT(30),
            TokCP(0, 2),
            TokLT(40),
            TokCP(1, 2),
        ],
        expected_out=[
            10,
            10,
            30,
            30, 30,
            40,
            30, 40
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_end1(dut):
    """
    Check that END mark is propagated
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.END),
        ],
        expected_out=[
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_end2(dut):
    """
    Check that several END marks without any data in between are propagated
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokMK(Mark.END),
        ],
        expected_out=[
            Mark.END,
            Mark.END,
            Mark.END,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_block1(dut):
    """
    Check that a simple block with a single LT is handled properly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokMK(Mark.END),
        ],
        expected_out=[
            10,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_block2(dut):
    """
    Check that a simple block with two LT is handled properly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokLT(20),
            TokMK(Mark.END),
        ],
        expected_out=[
            10,
            20,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_block3(dut):
    """
    Check that a simple block with LT and CP is handled properly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokCP(0, 1),
            TokMK(Mark.END),
        ],
        expected_out=[
            10,
            10,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_blocks1(dut):
    """
    Check that two blocks with LT are handled properly
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokMK(Mark.END),
            TokLT(20),
            TokMK(Mark.END),
        ],
        expected_out=[
            10,
            Mark.END,
            20,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_blocks2(dut):
    """
    Check that two blocks with CP are handled correctly
    (this also validates that END token is not written to HB)
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokMK(Mark.END),
            TokCP(0, 1),
            TokMK(Mark.END),
        ],
        expected_out=[
            10,
            Mark.END,
            10,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_blocks3(dut):
    """
    Check that several blocks with CP are handled correctly
    (this also validates that END token is not written to HB)
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokLT(20),
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokCP(1, 2),
        ],
        expected_out=[
            10,
            Mark.END,
            Mark.END,
            20,
            Mark.END,
            Mark.END,
            10, 20
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_reset1(dut):
    """
    Check that RESET token is propagated
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.RESET),
        ],
        expected_out=[
            Mark.RESET,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_reset2(dut):
    """
    Check that RESET & END tokens are propagated
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.RESET),
            TokMK(Mark.END),
            TokMK(Mark.RESET),
            TokMK(Mark.RESET),
            TokMK(Mark.END),
            TokMK(Mark.END),
        ],
        expected_out=[
            Mark.RESET,
            Mark.END,
            Mark.RESET,
            Mark.RESET,
            Mark.END,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error1(dut):
    """
    Check that ERROR token is propagated
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.ERROR_BAD_MARK),
        ],
        expected_out=[
            Mark.ERROR_BAD_MARK,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error2(dut):
    """
    Check that ERROR token is propagated and no further tokens are
    processed afterwards
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokMK(Mark.ERROR_BAD_MARK),
            TokLT(1),
            TokCP(0, 1),
            TokMK(Mark.END),
            TokMK(Mark.ERROR_BAD_MARK),
        ],
        expected_out=[
            Mark.ERROR_BAD_MARK,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_reset1(dut):
    """
    Check that ERROR token is propagated, no further tokens are
    processed afterwards, and module can be RESET and start accepting tokens
    after that
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(1),
            TokMK(Mark.ERROR_BAD_MARK),
            TokLT(2),
            TokCP(0, 1),
            TokMK(Mark.END),
            TokMK(Mark.RESET),
            TokLT(3),
            TokMK(Mark.END),
        ],
        expected_out=[
            1,
            Mark.ERROR_BAD_MARK,
            Mark.RESET,
            3,
            Mark.END,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp1(dut):
    """
    Check that when given a CP that points to an unwritten HB location,
    ERROR_INVAL_CP is generated and no data is leaked from unwritten part
    of HB.
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokCP(0, 1),
        ],
        expected_out=[
            Mark.ERROR_INVAL_CP,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp2(dut):
    """
    Check that ERROR_INVAL_CP can be reset.
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokCP(0, 1),
            TokMK(Mark.RESET),
            TokLT(10),
            TokCP(0, 1),
        ],
        expected_out=[
            Mark.ERROR_INVAL_CP,
            Mark.RESET,
            10,
            10
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp3(dut):
    """
    Check that RESET resets HB pointers and then when given a CP that points
    to an unwritten HB location, ERROR_INVAL_CP is generated and no data is
    leaked from HB.
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(10),
            TokCP(0, 1),
            TokMK(Mark.RESET),
            TokCP(0, 1),
        ],
        expected_out=[
            10,
            10,
            Mark.RESET,
            Mark.ERROR_INVAL_CP,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp4(dut):
    """
    Check that after ERROR_INVAL_CP is produced, no other data is emitted
    besides RESET (if present)
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokCP(0, 1),
            TokLT(10),
            TokCP(0, 1),
            TokCP(10, 1),
            TokMK(Mark.END),
            TokMK(Mark.ERROR_BAD_MARK),
            TokMK(Mark.RESET),
        ],
        expected_out=[
            Mark.ERROR_INVAL_CP,
            Mark.RESET,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp5(dut):
    """
    Check that ERROR_INVAL_CP is produced for CP offset > 0
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(1),
            TokLT(2),
            TokLT(3),
            TokCP(0, 1),
            TokCP(2, 2),
            TokCP(10, 1),
        ],
        expected_out=[
            1,
            2,
            3,
            3,
            2, 3,
            Mark.ERROR_INVAL_CP,
        ])


@cocotb.test(timeout_time=1, timeout_unit='ms')
async def test_error_inval_cp6(dut):
    """
    Check that after ERROR_INVAL_CP is produced for maximum LZ4 CP offset
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokCP(0xFFFF, 1),
            TokMK(Mark.RESET),
            TokLT(10),
            TokCP(0xFFFF, 1),
        ],
        expected_out=[
            Mark.ERROR_INVAL_CP,
            Mark.RESET,
            10,
            Mark.ERROR_INVAL_CP,
        ])


@cocotb.test(timeout_time=1000, timeout_unit='ms')
async def test_error_inval_cp7(dut):
    """
    Check that ERROR_INVAL_CP is produced when only 1 symbol in buffer is
    unwritten
    """
    await do_comparison_test(
        dut,
        test_in=[TokLT(10)] * 65535 + [TokCP(65535, 1)],
        expected_out=[10] * 65535 + [Mark.ERROR_INVAL_CP],
    )


@cocotb.test(timeout_time=1000, timeout_unit='ms')
async def test_error_inval_cp8(dut):
    """
    Check that ERROR_INVAL_CP is not produced when buffer is full (test 1)
    """
    await do_comparison_test(
        dut,
        test_in=[TokLT(10)] * 65536 + [TokCP(65535, 1)],
        expected_out=[10] * 65536 + [10],
    )


@cocotb.test(timeout_time=1000, timeout_unit='ms')
async def test_error_inval_cp9(dut):
    """
    Check that ERROR_INVAL_CP is not produced when buffer is full (test 2)
    """
    await do_comparison_test(
        dut,
        test_in=[TokLT(10)] * 65536 + [TokCP(0, 1)],
        expected_out=[10] * 65536 + [10],
    )


@cocotb.test(timeout_time=1000, timeout_unit='ms')
async def test_cp_maxcount(dut):
    """
    Check that CP token with maximum permitted cp_cnt value works
    """
    await do_comparison_test(
        dut,
        test_in=[
            TokLT(7),
            TokCP(0, 65536),
            TokLT(15),
        ],
        expected_out=[7] * 65537 + [15],
    )
