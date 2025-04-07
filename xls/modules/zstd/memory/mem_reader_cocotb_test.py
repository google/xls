# Copyright 2024 The XLS Authors
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
import random
import sys
import warnings

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Event
from cocotb_bus.scoreboard import Scoreboard
from cocotbext.axi import axi_channels
from cocotbext.axi.axi_ram import AxiRamRead
from cocotbext.axi.sparse_memory import SparseMemory

import xls.modules.zstd.cocotb.channel as xlschannel
from xls.modules.zstd.cocotb import utils
from xls.modules.zstd.cocotb import xlsstruct

# to disable warnings from hexdiff used by cocotb's Scoreboard
warnings.filterwarnings("ignore", category=DeprecationWarning)

DSLX_DATA_W = 64
DSLX_ADDR_W = 16

AXI_DATA_W = 128
AXI_ADDR_W = 16

LAST_W = 1
STATUS_W = 1
ERROR_W = 1
ID_W = 4
DEST_W = 4

# AXI
AXI_AR_PREFIX = "axi_ar"
AXI_R_PREFIX = "axi_r"

# MemReader
MEM_READER_REQ_CHANNEL = "req"
MEM_READER_RESP_CHANNEL = "resp"

# Override default widths of AXI response signals
signal_widths = {"rresp": 3, "rlast": 1}
axi_channels.AxiRBus._signal_widths = signal_widths
axi_channels.AxiRTransaction._signal_widths = signal_widths
axi_channels.AxiRSource._signal_widths = signal_widths
axi_channels.AxiRSink._signal_widths = signal_widths
axi_channels.AxiRMonitor._signal_widths = signal_widths

@xlsstruct.xls_dataclass
class MemReaderReq(xlsstruct.XLSStruct):
  addr: DSLX_ADDR_W
  length: DSLX_ADDR_W


@xlsstruct.xls_dataclass
class MemReaderResp(xlsstruct.XLSStruct):
  status: STATUS_W
  data: DSLX_DATA_W
  length: DSLX_ADDR_W
  last: LAST_W


@xlsstruct.xls_dataclass
class AxiReaderReq(xlsstruct.XLSStruct):
  addr: AXI_ADDR_W
  len: AXI_ADDR_W


@xlsstruct.xls_dataclass
class AxiStream(xlsstruct.XLSStruct):
  data: AXI_DATA_W
  str: AXI_DATA_W // 8
  keep: AXI_DATA_W // 8 = 0
  last: LAST_W = 0
  id: ID_W = 0
  dest: DEST_W = 0


@xlsstruct.xls_dataclass
class AxiReaderError(xlsstruct.XLSStruct):
  error: ERROR_W


@xlsstruct.xls_dataclass
class AxiAr(xlsstruct.XLSStruct):
  id: ID_W
  addr: AXI_ADDR_W
  region: 4
  len: 8
  size: 3
  burst: 2
  cache: 4
  prot: 3
  qos: 4


@xlsstruct.xls_dataclass
class AxiR(xlsstruct.XLSStruct):
  id: ID_W
  data: AXI_DATA_W
  resp: 3
  last: 1


def print_callback(name: str = "monitor"):
  def _print_callback(transaction):
    print(f" [{name}]: {transaction}")

  return _print_callback


def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      print("all transactions received")
      event.set()

  monitor.add_callback(terminate_cb)


def generate_test_data(test_cases, xfer_base=0x0, seed=1234):
  random.seed(seed)
  mem_size = 2**AXI_ADDR_W
  data_w_div8 = DSLX_DATA_W // 8

  assert xfer_base < mem_size, "Base address outside the memory span"

  req = []
  resp = []
  mem_writes = {}

  for xfer_offset, xfer_length in test_cases:
    xfer_addr = xfer_base + xfer_offset
    xfer_max_addr = xfer_addr + xfer_length

    if xfer_length == 0:
      req += [MemReaderReq(addr=xfer_addr, length=0)]
      resp += [MemReaderResp(status=0, data=0, length=0, last=1)]

    assert xfer_max_addr < mem_size, "Max address outside the memory span"
    req += [MemReaderReq(addr=xfer_addr, length=xfer_length)]

    rem = xfer_length % data_w_div8
    range_end = xfer_max_addr - (data_w_div8 - 1)
    for addr in range(xfer_addr, range_end, data_w_div8):
      last = ((addr + data_w_div8) >= xfer_max_addr) & (rem == 0)
      data = random.randint(0, 1 << (data_w_div8 * 8))
      mem_writes.update({addr: data})
      resp += [
        MemReaderResp(status=0, data=data, length=data_w_div8, last=last)
      ]

    if rem > 0:
      addr = xfer_max_addr - rem
      mask = (1 << (rem * 8)) - 1
      data = random.randint(0, 1 << (data_w_div8 * 8))
      mem_writes.update({addr: data})
      resp += [MemReaderResp(status=0, data=data & mask, length=rem, last=1)]

  return (req, resp, mem_writes)


async def test_mem_reader(dut, req_input, resp_output, mem_contents=None):
  if mem_contents is None:
    mem_contents = {}

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  # prefix unused objects with unused_
  # to suppress linter and keep the objects alive
  unused_mem_reader_resp_bus = xlschannel.XLSChannel(
    dut, MEM_READER_RESP_CHANNEL, dut.clk, start_now=True
  )
  mem_reader_req_driver = xlschannel.XLSChannelDriver(
    dut, MEM_READER_REQ_CHANNEL, dut.clk
  )
  mem_reader_resp_monitor = xlschannel.XLSChannelMonitor(
    dut,
    MEM_READER_RESP_CHANNEL,
    dut.clk,
    MemReaderResp,
    callback=print_callback()
  )

  terminate = Event()
  set_termination_event(mem_reader_resp_monitor, terminate, len(resp_output))

  scoreboard = Scoreboard(dut)
  scoreboard.add_interface(mem_reader_resp_monitor, resp_output)

  ar_bus = axi_channels.AxiARBus.from_prefix(dut, AXI_AR_PREFIX)
  r_bus = axi_channels.AxiRBus.from_prefix(dut, AXI_R_PREFIX)
  axi_read_bus = axi_channels.AxiReadBus(ar=ar_bus, r=r_bus)

  mem_size = 2**AXI_ADDR_W
  sparse_mem = SparseMemory(mem_size)
  for addr, data in mem_contents.items():
    sparse_mem.write(addr, (data).to_bytes(8, "little"))

  unused_memory = AxiRamRead(
    axi_read_bus, dut.clk, dut.rst, size=mem_size, mem=sparse_mem
  )

  await utils.reset(dut.clk, dut.rst, cycles=10)
  await mem_reader_req_driver.send(req_input)
  await terminate.wait()


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_zero_length_req(dut):
  req, resp, _ = generate_test_data(
    xfer_base=0xFFF, test_cases=[(0x101, 0)]
  )
  await test_mem_reader(dut, req, resp)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_aligned_transfer_shorter_than_bus(dut):
  req, resp, mem_contents = generate_test_data(
    xfer_base=0xFFF, test_cases=[(0x101, 1)]
  )
  await test_mem_reader(dut, req, resp, mem_contents)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_aligned_transfer_shorter_than_bus1(dut):
  req, resp, mem_contents = generate_test_data(
    xfer_base=0xFFF, test_cases=[(0x2, 1)]
  )
  await test_mem_reader(dut, req, resp, mem_contents)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_aligned_transfer_shorter_than_bus2(dut):
  req, resp, mem_contents = generate_test_data(
    xfer_base=0xFFF, test_cases=[(0x2, 17)]
  )
  await test_mem_reader(dut, req, resp, mem_contents)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_aligned_transfer_shorter_than_bus3(dut):
  req, resp, mem_contents = generate_test_data(
    xfer_base=0xFFF, test_cases=[(0x0, 0x1000)]
  )
  await test_mem_reader(dut, req, resp, mem_contents)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def mem_reader_aligned_transfer_shorter_than_bus4(dut):
  req, resp, mem_contents = generate_test_data(
    xfer_base=0x1, test_cases=[(0x0, 0xFFF), (0x1000, 0x1)]
  )
  await test_mem_reader(dut, req, resp, mem_contents)


if __name__ == "__main__":
  sys.path.append(str(pathlib.Path(__file__).parent))

  toplevel = "mem_reader_wrapper"
  verilog_sources = [
    "xls/modules/zstd/memory/mem_reader_adv.v",
    "xls/modules/zstd/memory/rtl/mem_reader_wrapper.v",
  ]
  test_module = [pathlib.Path(__file__).stem]
  utils.run_test(toplevel, test_module, verilog_sources)
