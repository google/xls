#!/usr/bin/env python
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

import random
import cocotb
import cocotbext.axi.axi_channels as axi

from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Event
from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.memory import AxiRamFromArray
from xls.modules.zstd.cocotb.utils import reset, run_test
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct, xls_dataclass
from xls.modules.zstd.cocotb.data_generator import DecompressFrame


DATA_W = 32
ADDR_W = 32
STATUS_W = 1
LAST_W = 1
STATUS_W = 1
ERROR_W = 1
ID_W = 4
DEST_W = 4
HT_SIZE_W = 10

MEM_SIZE = 0x10000
OBUF_ADDR = 0x1000
INPUT_RANGE = (0, 10)
INPUT_SIZE = 0x1000

signal_widths = {"bresp": 3}
axi.AxiBBus._signal_widths = signal_widths
axi.AxiBTransaction._signal_widths = signal_widths
axi.AxiBSource._signal_widths = signal_widths
axi.AxiBSink._signal_widths = signal_widths
axi.AxiBMonitor._signal_widths = signal_widths
signal_widths = {"rresp": 3, "rlast": 1}
axi.AxiRBus._signal_widths = signal_widths
axi.AxiRTransaction._signal_widths = signal_widths
axi.AxiRSource._signal_widths = signal_widths
axi.AxiRSink._signal_widths = signal_widths
axi.AxiRMonitor._signal_widths = signal_widths

@xls_dataclass
class Req(XLSStruct):
  input_offset: ADDR_W
  data_size: DATA_W
  output_offset: ADDR_W
  max_block_size: DATA_W

@xls_dataclass
class Resp(XLSStruct):
  status: STATUS_W

def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()
  monitor.add_callback(terminate_cb)

def connect_axi_read_bus(dut, name=""):
  AXI_AR = "axi_ar"
  AXI_R = "axi_r"

  if name != "":
      name += "_"

  bus_axi_ar = axi.AxiARBus.from_prefix(dut, name + AXI_AR)
  bus_axi_r = axi.AxiRBus.from_prefix(dut, name + AXI_R)

  return axi.AxiReadBus(bus_axi_ar, bus_axi_r)

def connect_axi_write_bus(dut, name=""):
  AXI_AW = "axi_aw"
  AXI_W = "axi_w"
  AXI_B = "axi_b"

  if name != "":
      name += "_"

  bus_axi_aw = axi.AxiAWBus.from_prefix(dut, name + AXI_AW)
  bus_axi_b = axi.AxiBBus.from_prefix(dut, name + AXI_B)
  bus_axi_w = axi.AxiWBus.from_prefix(dut, name + AXI_W)

  return axi.AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

def connect_axi_bus(dut, name=""):
  bus_axi_read = connect_axi_read_bus(dut, name)
  bus_axi_write = connect_axi_write_bus(dut, name)

  return axi.AxiBus(bus_axi_write, bus_axi_read)

def generate_random_bytes():
  # we do it this way so that there's a bigger probability of symbols repeating
  return bytearray(sum([[0,0,0,0,0,0,0, random.randint(*INPUT_RANGE)] for _ in range(INPUT_SIZE//8)], []))

@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def basic_test(dut):

  dut.rst.setimmediatevalue(0)
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  # setup input data
  input_data = generate_random_bytes()
  memory = AxiRamFromArray(connect_axi_bus(dut, "memory"), dut.clk, dut.rst, arr=input_data, size=MEM_SIZE)
  req = Req(
      input_offset=0x0,
      data_size=INPUT_SIZE,
      output_offset=OBUF_ADDR,
      max_block_size=128
  )

  # channels
  ch_resp = XLSChannel(dut, "resp_s", dut.clk, start_now=True)
  drv_req = XLSChannelDriver(dut, "req_r", dut.clk)
  mon_req = XLSChannelMonitor(dut, "req_r", dut.clk, Req)
  mon_resp = XLSChannelMonitor(dut, "resp_s", dut.clk, Resp)
  terminate = Event()
  set_termination_event(mon_resp, terminate, 1)

  # run benchmark
  await reset(dut.clk, dut.rst, cycles=10)
  await cocotb.start(drv_req.send(req))
  await terminate.wait()

  mem_contents = memory.read(OBUF_ADDR, 0x2000)
  dctx = DecompressFrame(mem_contents)
  assert(dctx == input_data)


if __name__ == "__main__":
  toplevel = "zstd_enc_wrapper"
  verilog_sources = [
    "xls/modules/zstd/zstd_enc_cocotb.v",
    "xls/modules/zstd/rtl/zstd_enc_wrapper.v",
    "xls/modules/zstd/rtl/xls_fifo_wrapper.v",
  ]

  test_module=[Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)
