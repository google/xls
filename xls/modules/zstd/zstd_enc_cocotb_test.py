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

from inspect import currentframe
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Event, RisingEdge
from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.memory import AxiRamFromArray
from xls.modules.zstd.cocotb.utils import reset, run_test, connect_axi_bus
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct, xls_dataclass
from xls.modules.zstd.cocotb.data_generator import DecompressFrame


DATA_W = 32
ADDR_W = 32
PARAMS_W = 1
STATUS_W = 1
LAST_W = 1
STATUS_W = 1
ERROR_W = 1
ID_W = 4
DEST_W = 4
HT_SIZE_W = 10

MEM_SIZE = 0x100000
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
  enable_rle: PARAMS_W
  enable_compressed: PARAMS_W

@xls_dataclass
class Resp(XLSStruct):
  status: STATUS_W
  written_bytes: ADDR_W

def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()
  monitor.add_callback(terminate_cb)

def generate_random_bytes():
  return bytearray([random.randint(*INPUT_RANGE) for _ in range(INPUT_SIZE)])

def generate_single_byte():
  return bytearray([0x42 for _ in range(INPUT_SIZE)])

async def testcase(dut, params, input_data, max_block_size = 128):
  dut.rst.setimmediatevalue(0)
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  # setup input data
  memory = AxiRamFromArray(connect_axi_bus(dut, "memory"), dut.clk, dut.rst, arr=input_data, size=MEM_SIZE)
  req = Req(
      input_offset=0x0,
      data_size=len(input_data),
      output_offset=OBUF_ADDR,
      max_block_size=max_block_size,
      **params
  )

  # channels
  ch_resp = XLSChannel(dut, "resp_s", dut.clk, start_now=True)
  drv_req = XLSChannelDriver(dut, "req_r", dut.clk)
  # mon_req = XLSChannelMonitor(dut, "req_r", dut.clk, Req)
  # mon_resp = XLSChannelMonitor(dut, "resp_s", dut.clk, Resp)
  terminate = Event()
  # set_termination_event(mon_resp, terminate, 1)

  # run benchmark
  await reset(dut.clk, dut.rst, cycles=10)
  await cocotb.start(drv_req.send(req))

  while True:
    await RisingEdge(dut.clk)
    if ch_resp.rdy.value and ch_resp.vld.value:
        resp = Resp.from_int(ch_resp.data.value.integer)
        break;

  with open("input.bin", "wb") as f:
    f.write(input_data)

  mem_contents = memory.read(OBUF_ADDR, resp.written_bytes)
  with open("result.zst", "wb") as f:
    f.write(mem_contents)

  dctx = DecompressFrame(mem_contents)

  for i in range(len(dctx)):
    print(f"comparing {hex(dctx[i])}=={hex(input_data[i])}")
    assert dctx[i] == input_data[i]

PREGENERATED_FILES_DIR = '../xls/modules/zstd/data/'

async def testcase_pregenerated(dut, filename, max_block_size, params):
  with open(PREGENERATED_FILES_DIR + filename, "rb") as f:
    input_data = f.read()
  input_data = bytearray(input_data)
  await testcase(
    dut,
    params,
    input_data,
    max_block_size
  )

@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def raw_block_test(dut):
  await testcase(
    dut,
    params={
      "enable_rle": False,
      "enable_compressed": False
    },
    input_data=generate_random_bytes()
  )

@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def rle_block_test(dut):
  await testcase(
    dut,
    params={
      "enable_rle": True,
      "enable_compressed": False
    },
    input_data=generate_single_byte()
  )

@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def comp_block_test(dut):
  random.seed(42) # for reproducibility
  await testcase(
    dut,
    params={
      "enable_rle": False,
      "enable_compressed": True
    },
    input_data=generate_random_bytes(),
    max_block_size=0x400
  )

def parse_decl(frame):
  fname = frame.f_code.co_name
  tokens = fname.split("_")
  filename = f"enc_{tokens[0]}_{tokens[1]}"
  max_block_size = int(tokens[3][:-1])
  mode = tokens[4]
  params = {
    "enable_rle": mode == "rle",
    "enable_compressed": mode == "compressed"
  }
  return filename, max_block_size, params

# pregenerated testcases
# convention:
# pregenerated_<input_size>B_block_<max_block_size>B_<mode>
# input_size should match one of the input sizes available in xls/modules/zstd/data
# mode should be one of:
# - compressed
# - rle
# - raw
# it is then parsed and a test with specified params is run

# RAW
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_2000B_block_100B_raw(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_2000B_block_500B_raw(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))

# COMPRESSED
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_200B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_200B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_200B_block_500B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_300B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_300B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_300B_block_500B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_500B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_500B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1024B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1024B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1024B_block_500B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1030B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1030B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=10000, timeout_unit="ms")
async def pregenerated_1030B_block_500B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=30000, timeout_unit="ms")
async def pregenerated_2000B_block_10B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def pregenerated_2000B_block_100B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))
@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def pregenerated_2000B_block_500B_compressed(dut): await testcase_pregenerated(dut, *parse_decl(currentframe()))


if __name__ == "__main__":
  toplevel = "zstd_enc_wrapper"
  verilog_sources = [
    "xls/modules/zstd/zstd_enc_cocotb.v",
    "xls/modules/zstd/rtl/zstd_enc_wrapper.sv",
    "xls/modules/zstd/rtl/xls_fifo_wrapper.sv",
    "xls/modules/zstd/rtl/ram_1r1w.sv"
  ]

  test_module=[Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)
