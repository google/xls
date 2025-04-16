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

from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Event, ClockCycles
from cocotbext.axi.axi_channels import AxiAWBus, AxiWBus, AxiBBus, AxiWriteBus, AxiARBus, AxiRBus, AxiReadBus, AxiBus, AxiBTransaction, AxiBSource, AxiBSink, AxiBMonitor, AxiRTransaction, AxiRSource, AxiRSink, AxiRMonitor
from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.memory import AxiRamFromArray
from xls.modules.zstd.cocotb.utils import reset, run_test
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct, xls_dataclass

DATA_W = 64
ADDR_W = 32
STATUS_W = 1
LAST_W = 1
STATUS_W = 1
ERROR_W = 1
ID_W = 4
DEST_W = 4
HT_SIZE_W = 10

MEM_SIZE = 0x10000
OBUF_ADDR = 0x2000
LITERALS_OBUF_ADDR = OBUF_ADDR + 0x1000
SEQUENCES_OBUF_ADDR = OBUF_ADDR + 0x2000
INPUT_RANGE = (0, 10)
INPUT_SIZE = 0x1000

SEQ_CNT=None
LIT_CNT=None
STATUS=None

signal_widths = {"bresp": 3}
AxiBBus._signal_widths = signal_widths
AxiBTransaction._signal_widths = signal_widths
AxiBSource._signal_widths = signal_widths
AxiBSink._signal_widths = signal_widths
AxiBMonitor._signal_widths = signal_widths
signal_widths = {"rresp": 3, "rlast": 1}
AxiRBus._signal_widths = signal_widths
AxiRTransaction._signal_widths = signal_widths
AxiRSource._signal_widths = signal_widths
AxiRSink._signal_widths = signal_widths
AxiRMonitor._signal_widths = signal_widths

@xls_dataclass
class Req(XLSStruct):
  input_addr: ADDR_W
  input_size: ADDR_W
  output_lit_addr: ADDR_W
  output_seq_addr: ADDR_W
  zstd_params: HT_SIZE_W

@xls_dataclass
class Resp(XLSStruct):
  status: STATUS_W
  lit_cnt: 32
  seq_cnt: 32

def set_termination_event(monitor, event, transactions):
  def terminate_cb(resp):
    global SEQ_CNT, LIT_CNT, STATUS
    if monitor.stats.received_transactions == transactions:
      event.set()
    SEQ_CNT = resp.seq_cnt
    LIT_CNT = resp.lit_cnt
    STATUS = resp.status
  monitor.add_callback(terminate_cb)


def connect_axi_read_bus(dut, name=""):
  AXI_AR = "axi_ar"
  AXI_R = "axi_r"

  if name != "":
      name += "_"

  bus_axi_ar = AxiARBus.from_prefix(dut, name + AXI_AR)
  bus_axi_r = AxiRBus.from_prefix(dut, name + AXI_R)

  return AxiReadBus(bus_axi_ar, bus_axi_r)

def connect_axi_write_bus(dut, name=""):
  AXI_AW = "axi_aw"
  AXI_W = "axi_w"
  AXI_B = "axi_b"

  if name != "":
      name += "_"

  bus_axi_aw = AxiAWBus.from_prefix(dut, name + AXI_AW)
  bus_axi_b = AxiBBus.from_prefix(dut, name + AXI_B)
  bus_axi_w = AxiWBus.from_prefix(dut, name + AXI_W)

  return AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

def connect_axi_bus(dut, name=""):
  bus_axi_read = connect_axi_read_bus(dut, name)
  bus_axi_write = connect_axi_write_bus(dut, name)

  return AxiBus(bus_axi_write, bus_axi_read)


def simple_decode(literals, sequences):
  def word_from_bytes(bytes, ix):
    return int(bytes[ix]) + (int(bytes[ix+1]) << 8)

  literals_ix = 0
  decoded = []
  for i in range(0, len(sequences), 8):
    match_length = word_from_bytes(sequences, i)
    step_back = word_from_bytes(sequences, i + 2)
    copy_length = word_from_bytes(sequences, i + 4)
    for j in range(literals_ix * 8, (literals_ix + copy_length) * 8):
      decoded.append(literals[j])

    matched = []
    current_decoded_len = len(decoded)
    for j in range(
      current_decoded_len - step_back * 8,
      current_decoded_len - (step_back - match_length) * 8):
      matched.append(decoded[j])
    decoded += matched
    literals_ix += copy_length
  return decoded

def generate_input_data():
  random.seed(42) # for reproducibility
  return bytearray(sum(
    [[
        random.randint(*INPUT_RANGE),0,0,0,0,0,0,0]
        for _ in range(INPUT_SIZE//8)
      ],
    []
  ))


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def basic_test(dut):

  dut.rst.setimmediatevalue(0)
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  input_data = generate_input_data()
  memory = AxiRamFromArray(
    connect_axi_bus(dut, "memory"),
    dut.clk, dut.rst,
    arr=input_data, size=MEM_SIZE
  )
  req = Req(
      input_addr=0x0,
      input_size=INPUT_SIZE,
      output_lit_addr=LITERALS_OBUF_ADDR,
      output_seq_addr=SEQUENCES_OBUF_ADDR,
      zstd_params=9
  )

  # channels
  _ = XLSChannel(dut, "resp_s", dut.clk, start_now=True)
  drv_req = XLSChannelDriver(dut, "req_r", dut.clk)
  _ = XLSChannelMonitor(dut, "req_r", dut.clk, Req)
  mon_resp = XLSChannelMonitor(dut, "resp_s", dut.clk, Resp)
  terminate = Event()
  set_termination_event(mon_resp, terminate, 1)

  # run benchmark
  await reset(dut.clk, dut.rst, cycles=1000)
  await cocotb.start(drv_req.send(req))
  await terminate.wait()

  assert STATUS == 0
  await ClockCycles(dut.clk, 1000) # make sure all memory writes finish

  literals = memory.read(LITERALS_OBUF_ADDR, LIT_CNT * 8)
  sequences = memory.read(SEQUENCES_OBUF_ADDR, SEQ_CNT * 8)
  decoded = simple_decode(literals, sequences)

  all_equal = True
  for i in range(0, INPUT_SIZE):
    if input_data[i] != decoded[i]:
      all_equal = False
      print(
        f'''
        Bytes not equal at {hex(i)}: 
          expected {hex(input_data[i])}, 
          actual {hex(decoded[i])}
        '''
      )

  assert all_equal


if __name__ == "__main__":
  toplevel = "match_finder_wrapper"
  verilog_sources = [
    "xls/modules/zstd/match_finder_cocotb.v",
    "xls/modules/zstd/rtl/ram_1r1w.v",
    "xls/modules/zstd/rtl/xls_fifo_wrapper.v",
    "xls/modules/zstd/rtl/match_finder_wrapper.v",
  ]
  test_module=[Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)

