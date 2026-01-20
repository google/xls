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
from cocotb.triggers import Event, ClockCycles
from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.memory import AxiRamFromArray
from xls.modules.zstd.cocotb.utils import reset, run_test, connect_axi_bus
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

def deserialized_sequences(sequences):
  def word_from_bytes(bytes, ix):
    return int(bytes[ix]) + (int(bytes[ix+1]) << 8)
  for i in range(0, len(sequences), 6):
    match_length = word_from_bytes(sequences, i) + 3
    step_back = word_from_bytes(sequences, i + 2) - 3
    copy_length = word_from_bytes(sequences, i + 4)
    yield (match_length, step_back, copy_length)

def simple_decode(literals, sequences):
  literals_ix = 0
  decoded = []
  for seq in deserialized_sequences(sequences):
    (match_length, step_back, copy_length) = seq
    print(f"Processing sequence ({seq})...")
    for j in range(literals_ix, (literals_ix + copy_length)):
      print(f"{hex(len(decoded))} = {hex(literals[j])}")
      decoded.append(literals[j])
    matched = []
    current_decoded_len = len(decoded)
    for j in range(
      current_decoded_len - step_back,
      current_decoded_len - (step_back - match_length)):
      matched.append(decoded[j])
    print(f"{hex(len(decoded))} = {[hex(ma) for ma in matched]}")
    decoded += matched
    literals_ix += copy_length
  # Append the remaining literals
  for i in range(literals_ix, len(literals)):
    decoded.append(literals[i])

  return decoded

def generate_input_data():
  random.seed(42) # for reproducibility
  return bytearray(random.randint(*INPUT_RANGE) for _ in range(INPUT_SIZE))


@cocotb.test(timeout_time=10000, timeout_unit="ms")
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

  literals = memory.read(LITERALS_OBUF_ADDR, LIT_CNT)
  sequences = memory.read(SEQUENCES_OBUF_ADDR, SEQ_CNT * 6)
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
    "xls/modules/zstd/rtl/ram_1r1w.sv",
    "xls/modules/zstd/rtl/xls_fifo_wrapper.sv",
    "xls/modules/zstd/rtl/match_finder_wrapper.sv",
  ]
  test_module=[Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)

