# Copyright 2025 The XLS Authors
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

import pathlib

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

from xls.modules.zstd.cocotb.utils import run_test

async def reset_dut(dut, rst_len=10):
  dut.rst.setimmediatevalue(0)
  await ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(1)
  await RisingEdge(dut.clk)
  dut.rst.setimmediatevalue(0)

@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame(dut):
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  await reset_dut(dut)

  dut.wr_en_0.value = 0x1
  dut.addr_0.value = 0x0
  dut.data_0.value = 0x2

  await RisingEdge(dut.clk)

  dut.wr_en_0.value = 0x0
  dut.rd_en_0.value = 0x1
  dut.addr_0.value = 0x0

  dut.wr_en_1.value = 0x1
  dut.addr_1.value = 0xc
  dut.data_1.value = 0x7

  await RisingEdge(dut.clk)

  dut.rd_en_0.value = 0x0

  dut.wr_en_1.value = 0x0
  dut.rd_en_1.value = 0x1
  dut.addr_1.value = 0xc

  await RisingEdge(dut.clk)

  dut.rd_en_1.value = 0x0

  await ClockCycles(dut.clk, 5)

  assert(dut.rd_data_0.value == 0x2)
  assert(dut.rd_data_1.value == 0x7)


if __name__ == "__main__":
  toplevel = "ram_2rw"
  verilog_sources = [
    "xls/modules/zstd/rtl/ram_2rw.v",
  ]
  test_module = [pathlib.Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)
