#!/usr/bin/env python
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


import random
import logging
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Event
from cocotb.binary import BinaryValue
from cocotb_bus.scoreboard import Scoreboard

from cocotbext.axi.axis import AxiStreamSource, AxiStreamBus, AxiStreamFrame
from cocotbext.axi.axi_channels import AxiAWBus, AxiWBus, AxiBBus, AxiWriteBus, AxiAWMonitor, AxiWMonitor, AxiBMonitor, AxiBTransaction, AxiBSource, AxiBSink
from cocotbext.axi.axi_ram import AxiRamWrite
from cocotbext.axi.sparse_memory import SparseMemory

from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.utils import reset, run_test
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct, xls_dataclass

ID_WIDTH = 4
ADDR_WIDTH = 16

# Override default widths of AXI response signals
signal_widths = {"bresp": 3}
AxiBBus._signal_widths = signal_widths
AxiBTransaction._signal_widths = signal_widths
AxiBSource._signal_widths = signal_widths
AxiBSink._signal_widths = signal_widths
AxiBMonitor._signal_widths = signal_widths

@xls_dataclass
class AxiWriterRespStruct(XLSStruct):
  status: 1

@xls_dataclass
class WriteRequestStruct(XLSStruct):
  address: ADDR_WIDTH
  length: ADDR_WIDTH

def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()
  monitor.add_callback(terminate_cb)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def ram_test(dut):
  GENERIC_ADDR_REQ_CHANNEL = "write_req"
  GENERIC_ADDR_RESP_CHANNEL = "write_resp"
  AXI_STREAM_CHANNEL = "axi_st_read"
  AXI_AW_CHANNEL = "axi_aw"
  AXI_W_CHANNEL = "axi_w"
  AXI_B_CHANNEL = "axi_b"

  terminate = Event()

  mem_size = 2**ADDR_WIDTH
  test_count = 200

  (addr_req_input, axi_st_input, addr_resp_expect, memory_verification, expected_memory) = generate_test_data_random(test_count, mem_size)

  dut.rst.setimmediatevalue(0)

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  resp_bus = XLSChannel(dut, GENERIC_ADDR_RESP_CHANNEL, dut.clk, start_now=True)

  driver_addr_req = XLSChannelDriver(dut, GENERIC_ADDR_REQ_CHANNEL, dut.clk)
  driver_axi_st = AxiStreamSource(AxiStreamBus.from_prefix(dut, AXI_STREAM_CHANNEL), dut.clk, dut.rst)

  bus_axi_aw = AxiAWBus.from_prefix(dut, AXI_AW_CHANNEL)
  bus_axi_w = AxiWBus.from_prefix(dut, AXI_W_CHANNEL)
  bus_axi_b = AxiBBus.from_prefix(dut, AXI_B_CHANNEL)
  bus_axi_write = AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

  monitor_addr_req = XLSChannelMonitor(dut, GENERIC_ADDR_REQ_CHANNEL, dut.clk, WriteRequestStruct)
  monitor_addr_resp = XLSChannelMonitor(dut, GENERIC_ADDR_RESP_CHANNEL, dut.clk, AxiWriterRespStruct)
  monitor_axi_aw = AxiAWMonitor(bus_axi_aw, dut.clk, dut.rst)
  monitor_axi_w = AxiWMonitor(bus_axi_w, dut.clk, dut.rst)
  monitor_axi_b = AxiBMonitor(bus_axi_b, dut.clk, dut.rst)

  set_termination_event(monitor_addr_resp, terminate, test_count)

  memory = AxiRamWrite(bus_axi_write, dut.clk, dut.rst, size=mem_size)

  log = logging.getLogger("cocotb.tb")
  log.setLevel(logging.WARNING)
  memory.log.setLevel(logging.WARNING)
  driver_axi_st.log.setLevel(logging.WARNING)

  scoreboard = Scoreboard(dut)
  scoreboard.add_interface(monitor_addr_resp, addr_resp_expect)

  await reset(dut.clk, dut.rst, cycles=10)
  await cocotb.start(driver_addr_req.send(addr_req_input))
  await cocotb.start(drive_axi_st(driver_axi_st, axi_st_input))
  await terminate.wait()

  for bundle in memory_verification:
    memory_contents = bytearray(memory.read(bundle["base_address"], bundle["length"]))
    expected_memory_contents = bytearray(expected_memory.read(bundle["base_address"], bundle["length"]))
    assert memory_contents == expected_memory_contents, "{} bytes of memory contents at base address {}:\n{}\nvs\n{}\nHEXDUMP:\n{}\nvs\n{}".format(hex(bundle["length"]), hex(bundle["base_address"]), memory_contents, expected_memory_contents, memory.hexdump(bundle["base_address"], bundle["length"]), expected_memory.hexdump(bundle["base_address"], bundle["length"]))

@cocotb.coroutine
async def drive_axi_st(driver, inputs):
  for axi_st_input in inputs:
    await driver.send(axi_st_input)

def generate_test_data_random(test_count, mem_size):
  AXI_AXSIZE_ENCODING_MAX_4B_TRANSFER = 2 # Must be in sync with AXI_AXSIZE_ENCODING enum in axi.x

  addr_req_input = []
  axi_st_input = []
  addr_resp_expect = []
  memory_verification = []
  memory = SparseMemory(mem_size)

  random.seed(1234)

  for i in range(test_count):
    xfer_addr = random.randrange(0, mem_size)
    # Don't allow unaligned writes
    xfer_addr_aligned = (xfer_addr // 4) * 4
    # Make sure we don't write beyond available memory
    memory_size_max_xfer_len = mem_size - xfer_addr_aligned
    arbitrary_max_xfer_len = 0x5000 # 20kB
    xfer_max_len = min(arbitrary_max_xfer_len, memory_size_max_xfer_len)
    xfer_len = random.randrange(1, xfer_max_len)
    transfer_req = WriteRequestStruct(
      address = xfer_addr_aligned,
      length = xfer_len,
    )
    addr_req_input.append(transfer_req)

    data_to_write = random.randbytes(xfer_len)
    axi_st_frame = AxiStreamFrame(tdata=data_to_write, tkeep=[15]*xfer_len, tid=(i % (1 << ID_WIDTH)), tdest=(i % (1 << ID_WIDTH)))
    axi_st_input.append(axi_st_frame)

    write_expected_memory(transfer_req, axi_st_frame.tdata, memory)

    memory_bundle = {
            "base_address": transfer_req.address,
            "length": transfer_req.length,
    }
    memory_verification.append(memory_bundle)

  addr_resp_expect = [AxiWriterRespStruct(status=False)] * test_count

  return (addr_req_input, axi_st_input, addr_resp_expect, memory_verification, memory)

def bytes_to_4k_boundary(addr):
    AXI_4K_BOUNDARY = 0x1000
    return AXI_4K_BOUNDARY - (addr % AXI_4K_BOUNDARY)

def write_expected_memory(transfer_req, data_to_write, memory):
    """
    Write test data to reference memory keeping the AXI 4kb boundary
    by spliting the write requests into smaller ones.
    """
    prev_id = 0
    address = transfer_req.address
    length = transfer_req.length

    BYTES_IN_TRANSFER = 4
    MAX_AXI_BURST_BYTES = 256 * BYTES_IN_TRANSFER

    while (length > 0):
      bytes_to_4k = bytes_to_4k_boundary(address)
      new_len = min(length, min(bytes_to_4k, MAX_AXI_BURST_BYTES))
      new_data = data_to_write[prev_id:prev_id+new_len]
      memory.write(address, new_data)
      address = address + new_len
      length = length - new_len
      prev_id = prev_id + new_len

def generate_test_data_arbitrary(mem_size):
  AXI_AXSIZE_ENCODING_MAX_4B_TRANSFER = 2 # Must be in sync with AXI_AXSIZE_ENCODING enum in axi.x

  addr_req_input = []
  axi_st_input = []
  addr_resp_expect = []
  memory_verification = []
  memory = SparseMemory(mem_size)

  xfer_addr_begin = [0, 8, 512, 1000, 0x1234, 256]
  xfer_len = [1, 2, 4, 8, 0x48d, 4]
  assert len(xfer_len) == len(xfer_addr_begin)
  testcase_num = len(xfer_addr_begin) # test cases to execute
  for i in range(testcase_num):
    transfer_req = WriteRequestStruct(
      address = xfer_addr_begin[i],
      length = xfer_len[i] * 4, # xfer_len[i] transfers per 4 bytes
    )
    addr_req_input.append(transfer_req)

    data_chunks = []
    data_bytes = [[(0xEF + j) & 0xFF, 0xBE, 0xAD, 0xDE] for j in range(xfer_len[i])]
    data_words = [int.from_bytes(data_bytes[j]) for j in range(xfer_len[i])]
    for j in range(xfer_len[i]):
      data_chunks += data_bytes[j]
    data_to_write = bytearray(data_chunks)
    axi_st_frame = AxiStreamFrame(tdata=data_to_write, tkeep=[15]*xfer_len[i], tid=i, tdest=i)
    axi_st_input.append(axi_st_frame)

    write_expected_memory(transfer_req, axi_st_frame.tdata, memory)

    memory_bundle = {
            "base_address": transfer_req.address,
            "length": transfer_req.length, # 4 byte words
    }
    memory_verification.append(memory_bundle)

  addr_resp_expect = [AxiWriterRespStruct(status=False)] * testcase_num

  return (addr_req_input, axi_st_input, addr_resp_expect, memory_verification, memory)

if __name__ == "__main__":
    toplevel = "axi_writer_wrapper"
    verilog_sources = [
      "xls/modules/zstd/memory/axi_writer.v",
      "xls/modules/zstd/memory/axi_writer_wrapper.v",
    ]
    test_module=[Path(__file__).stem]
    run_test(toplevel, test_module, verilog_sources)
