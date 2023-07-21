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

from dbe import TokMK, TokCP, TokLT, Mark, Token,\
                PlainData, Params, Decoder
from dbe_common_test import DslxPlainData8, DslxToken
from dbe_data import DataFiles

from typing import Sequence


# Underlying LZ4 algorithm parameters the tests must be aware of
MINMATCH = 4
FINAL_LITERALS = 12


@cocotb.coroutine
async def reset(clk, rst, cycles=1):
    await RisingEdge(clk)
    rst.value = 1
    await ClockCycles(clk, cycles)
    rst.value = 0


async def do_comparison_test(
        dut,
        test_in: Sequence[PlainData], expected_out: Sequence[Token],
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
    test_in_bd = list(DslxPlainData8.from_plain_data(b).as_binary() for b in test_in)
    expected_out_bd = list(
        DslxToken.from_token(t).as_binary() for t in expected_out)

    all_data_received = Event("All expected data received")

    clock = Clock(dut.clk, 1, units="us")

    i_data = XLSChannelDriver(dut, "i_data", dut.clk)
    o_token = XLSChannelMonitor(dut, "o_token", dut.clk)

    sb = Scoreboard(dut, fail_immediately=True)
    sb.add_interface(o_token, expected_out_bd)

    def check_if_all_data_received(_):
        if o_token.stats.received_transactions == len(expected_out):
            all_data_received.set()

    o_token.add_callback(check_if_all_data_received)
    o_token.bus.rdy.setimmediatevalue(1)

    await cocotb.start(clock.start())
    await reset(dut.clk, dut.rst, 1)
    await i_data.write(test_in_bd)
    if expected_out:
        await all_data_received.wait()
    await ClockCycles(dut.clk, post_watch_cycles)
    if sb.errors:
        raise cocotb.TestFailure("Received data different from expected")


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_void(dut):
    """
    Check that in absence of input the module does not generate any output

    (this runs long as we want to observe module's behavior after it finishes
    clearing HT RAM)
    """
    await do_comparison_test(
        dut,
        test_in=[
        ],
        expected_out=[
        ],
        post_watch_cycles=50000)


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_end1(dut):
    """
    Check that module correctly passes END token
    """
    await do_comparison_test(
        dut,
        test_in=[
            Mark.END
        ],
        expected_out=[
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_end2(dut):
    """
    Check that module correctly passes several END tokens
    """
    await do_comparison_test(
        dut,
        test_in=[
            Mark.END,
            Mark.END,
            Mark.END,
        ],
        expected_out=[
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_short1(dut):
    """
    Check that module correctly passes a single symbol as a literal
    """
    await do_comparison_test(
        dut,
        test_in=[
            10,
            Mark.END,
        ],
        expected_out=[
            TokLT(10),
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_short2(dut):
    """
    Check that module correctly passes a single symbol as a literal
    when repeated in several blocks
    """
    await do_comparison_test(
        dut,
        test_in=[
            10,
            Mark.END,
            20,
            Mark.END,
            30,
            Mark.END,
        ],
        expected_out=[
            TokLT(10),
            TokMK(Mark.END),
            TokLT(20),
            TokMK(Mark.END),
            TokLT(30),
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_cp_simple(dut):
    """
    Check the simplest case where a single CP is emitted in RLE-like manner.
    
    This also tests producing CP for repetitive sequences and the generation
    of final literal sequence.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*20 + [15]*FINAL_LITERALS + [Mark.END],
        expected_out=
            [TokLT(7)]*1 + [TokCP(0, 19)] + [TokLT(15)]*FINAL_LITERALS
            + [TokMK(Mark.END)]
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_cp_long(dut):
    """
    Check that we can emit CP for a more complicated sequence than a
    repetition of a single symbol.
    
    This also partially tests the generation of a final literal sequence
    (after match fail & LT token).
    """
    await do_comparison_test(
        dut,
        test_in=
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            ] + [15]*FINAL_LITERALS + [Mark.END],
        expected_out=
            [
                TokLT(1), TokLT(2), TokLT(3), TokLT(4),
                TokLT(5), TokLT(6), TokLT(7), TokLT(8), TokLT(9),
            ]
            + [TokCP(8, 9)]
            + [TokLT(15)]*FINAL_LITERALS + [TokMK(Mark.END)]
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit1(dut):
    """
    Check that the final sequence is generated correctly when
    encoder is fed FINAL_LITERALS copies of the same symbol. All those
    should be emitted as literals (LT).
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*FINAL_LITERALS + [Mark.END],
        expected_out=
            [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit2(dut):
    """
    Check that the final sequence is generated correctly in a corner
    case with preceding match being 2 symbols too short of MINMATCH.

    Specifically, this tests the behavior of encoder when given a repeated
    sequence of some symbol of length MINMATCH+FINAL_LITERALS-1.
    - First symbol creates LT token (no other possibility)
    - The next symbol can already make a match (CP) of up to
      MINMATCH+FINAL_LITERALS-2 symbols long. But since at least
      FINAL_LITERALS last symbols should be emitted as literals, the maximum
      match length is limited by MINMATCH-2, which is less than MINMATCH, thus
      CP will not be generated, and all symbols are emitted as literals (LT).
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*(MINMATCH+FINAL_LITERALS-1) + [Mark.END],
        expected_out=
            [TokLT(7)]*(MINMATCH+FINAL_LITERALS-1) + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit3(dut):
    """
    Check that the final sequence is generated correctly in a corner
    case with preceding match being 1 symbol too short of MINMATCH.

    - First symbol creates LT token (no other possibility)
    - The next symbol can already make a match (CP) of up to
      MINMATCH+FINAL_LITERALS-1 symbols long. But since at least
      FINAL_LITERALS last symbols should be emitted as literals, the maximum
      match length is limited by MINMATCH-1, which is less than MINMATCH, thus
      CP will not be generated, and all symbols are emitted as literals (LT).
    """
    await do_comparison_test(
        dut,
        test_in=
            [7] * (MINMATCH+FINAL_LITERALS) + [Mark.END],
        expected_out=
            [TokLT(7)]*(MINMATCH+FINAL_LITERALS) + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit4(dut):
    """
    Check that the final sequence is generated correctly in a corner
    case with preceding match being exactly MINMATCH long.

    - First symbol creates LT token (no other possibility)
    - The next symbol can already make a match (CP) of up to
      MINMATCH+FINAL_LITERALS symbols long. Since at least FINAL_LITERALS
      last symbols should be emitted as literals, the maximum match length
      is limited by MINMATCH, which is enough for CP, and it should generate
      a CP token.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7] * (MINMATCH+FINAL_LITERALS+1) + [Mark.END],
        expected_out=
            [
                TokLT(7),
                TokCP(0, MINMATCH),
            ]
            + [TokLT(7)] * FINAL_LITERALS
            + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit5(dut):
    """
    Check that the final sequence is generated correctly after mismatch.

    Similar to test_finlit4, but with an extra CP in the beginning.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*(1+MINMATCH)
            + [8]*(1+MINMATCH+FINAL_LITERALS)
            + [Mark.END],
        expected_out=
            [
                TokLT(7),
                TokCP(0, MINMATCH),
                TokLT(8),
                TokCP(0, MINMATCH),
            ]
            + [TokLT(8)] * FINAL_LITERALS
            + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_finlit6(dut):
    """
    Check that the final sequence can be generated several times, if
    encoder is given several blocks one after another.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*(1+MINMATCH+FINAL_LITERALS)
            + [Mark.END]
            + [7]*(MINMATCH+FINAL_LITERALS)
            + [Mark.END]
            + [8]*(1+MINMATCH+FINAL_LITERALS)
            + [Mark.END],
        expected_out=
            [TokLT(7), TokCP(0, MINMATCH)]
            + [TokLT(7)] * FINAL_LITERALS
            + [TokMK(Mark.END)]
            + [TokCP(MINMATCH + FINAL_LITERALS - 1, MINMATCH)]
            + [TokLT(7)] * FINAL_LITERALS
            + [TokMK(Mark.END)]
            + [TokLT(8), TokCP(0, MINMATCH)]
            + [TokLT(8)] * FINAL_LITERALS
            + [TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=200, timeout_unit='ms')
async def test_reset1(dut):
    """
    Simple check for correctness of END and RESET token passing
    """
    await do_comparison_test(
        dut,
        test_in=[
            Mark.END,
            Mark.RESET,
            Mark.END,
            Mark.END,
            Mark.RESET,
            Mark.END,
        ],
        expected_out=[
            TokMK(Mark.END),
            TokMK(Mark.RESET),
            TokMK(Mark.END),
            TokMK(Mark.END),
            TokMK(Mark.RESET),
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset2(dut):
    """
    Check that RESET does not drop symbols following after it
    """
    await do_comparison_test(
        dut,
        test_in=[
            Mark.RESET,
            7,
            Mark.END,
        ],
        expected_out=[
            TokMK(Mark.RESET),
            TokLT(7),
            TokMK(Mark.END),
        ])


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset3(dut):
    """
    Check that RESET token aborts final sequence generation.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*(FINAL_LITERALS-1)
            + [Mark.RESET, Mark.END],
        expected_out=
            [TokMK(Mark.RESET), TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset4(dut):
    """
    Check that RESET token aborts final sequence generation (test #2)
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*FINAL_LITERALS
            + [Mark.RESET, Mark.END],
        expected_out=
            [TokMK(Mark.RESET), TokMK(Mark.END)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset5(dut):
    """
    Check that RESET token aborts final sequence generation (test #3)
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*(FINAL_LITERALS+1)
            + [Mark.RESET, Mark.END],
        expected_out=
            [
                TokLT(7),
                TokMK(Mark.RESET),
                TokMK(Mark.END),
            ],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset6(dut):
    """
    Check that RESET token does not abort final sequence generation when
    passed after the END token.
    """
    await do_comparison_test(
        dut,
        test_in=
            [7]*FINAL_LITERALS
            + [Mark.END, Mark.RESET],
        expected_out=
            [TokLT(7)]*FINAL_LITERALS
            + [TokMK(Mark.END), TokMK(Mark.RESET)],
        )


@cocotb.test(timeout_time=100, timeout_unit='ms')
async def test_reset7(dut):
    """
    Check that RESET mark prevents matching with symbols preceding the mark.

    Matching can happen across END mark (thus, emitted CP tokens can look
    past the END token), but it must not happen across RESET mark (thus,
    emitted CP tokens should not look past the RESET).
    """
    await do_comparison_test(
        dut,
        test_in=
            [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END]
            + [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END]
            + [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END]
            + [Mark.RESET]
            + [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END]
            + [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END]
            + [10, 20, 30, 40] + [7]*FINAL_LITERALS + [Mark.END],
        expected_out=
            [TokLT(10), TokLT(20), TokLT(30), TokLT(40)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
            + [TokCP(FINAL_LITERALS+3, 4)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
            + [TokCP(FINAL_LITERALS+3, 4)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
            + [TokMK(Mark.RESET)]
            + [TokLT(10), TokLT(20), TokLT(30), TokLT(40)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
            + [TokCP(FINAL_LITERALS+3, 4)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
            + [TokCP(FINAL_LITERALS+3, 4)]
            + [TokLT(7)]*FINAL_LITERALS + [TokMK(Mark.END)]
        )


@cocotb.test(timeout_time=1000, timeout_unit='ms')
async def test_compression(dut):
    """
    Compress a piece of text and check that it can be decompressed correctly.
    """
    num_bytes_to_compress = 10000

    clock = Clock(dut.clk, 1, units="us")
    i_data = XLSChannelDriver(dut, "i_data", dut.clk)
    o_token = XLSChannelMonitor(dut, "o_token", dut.clk)

    # Test completes when END token is received, detect that
    end_received = Event("END token received")

    toks = []
    def recv_cb(trans):
        tok = DslxToken.from_binary(trans).to_token()
        toks.append(tok)
        if isinstance(tok, TokMK) and tok.mark == Mark.END:
            end_received.set()

    o_token.add_callback(recv_cb)
    o_token.bus.rdy.setimmediatevalue(1)

    # Get plaintext for compression
    with open(DataFiles.DICKENS_TXT, 'rb') as f:
        text1 = list(f.read(num_bytes_to_compress))
    text1.append(Mark.END)

    # Run
    await cocotb.start(clock.start())
    await reset(dut.clk, dut.rst, 1)
    await i_data.write(
        tuple(DslxPlainData8.from_plain_data(b).as_binary() for b in text1))
    await end_received.wait()
    
    # Validate the output using reference decoder
    par = Params(symbits = 8, ptrbits = 16, cntbits = 16)
    dec = Decoder(par)
    text2 = dec.decode(toks)
    for i in range(max(len(text1), len(text2))):
        assert i < len(text1), (
            f"Extra chars in decoded text: "
            f"l1={len(text1)} l2={len(text2)}, byte: {text2[i]!r}"
        )
        assert i < len(text2), (
            f"Preliminary EOF in decoded text: "
            f"l1={len(text1)} l2={len(text2)}, byte: {text1[i]!r}"
        )
        s1 = text1[i]
        s2 = text2[i]
        if s1 != s2:
            raise AssertionError(f"Mismatch @ {i}: expected {s1!r} got {s2!r}")
