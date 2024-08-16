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
from enum import Enum
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

DATA_WIDTH = 32
ADDR_WIDTH = 16

# Override default widths of AXI response signals
signal_widths = {"bresp": 3}
AxiBBus._signal_widths = signal_widths
AxiBTransaction._signal_widths = signal_widths
AxiBSource._signal_widths = signal_widths
AxiBSink._signal_widths = signal_widths
AxiBMonitor._signal_widths = signal_widths

@xls_dataclass
class DataInStruct(XLSStruct):
  data: DATA_WIDTH
  length: ADDR_WIDTH
  last: 1

@xls_dataclass
class WriteReqStruct(XLSStruct):
  offset: ADDR_WIDTH
  length: ADDR_WIDTH

@xls_dataclass
class MemWriterRespStruct(XLSStruct):
  status: 1

class MemWriterRespStatus(Enum):
  OKAY = 0
  ERROR = 1

@xls_dataclass
class WriteRequestStruct(XLSStruct):
  address: ADDR_WIDTH
  length: ADDR_WIDTH

def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()
  monitor.add_callback(terminate_cb)

async def test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt):
  GENERIC_WRITE_REQ_CHANNEL = "req"
  GENERIC_WRITE_RESP_CHANNEL = "resp"
  GENERIC_DATA_IN_CHANNEL = "data_in"
  AXI_AW_CHANNEL = "axi_aw"
  AXI_W_CHANNEL = "axi_w"
  AXI_B_CHANNEL = "axi_b"

  terminate = Event()

  dut.rst.setimmediatevalue(0)

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  resp_bus = XLSChannel(dut, GENERIC_WRITE_RESP_CHANNEL, dut.clk, start_now=True)

  driver_write_req = XLSChannelDriver(dut, GENERIC_WRITE_REQ_CHANNEL, dut.clk)
  driver_data_in = XLSChannelDriver(dut, GENERIC_DATA_IN_CHANNEL, dut.clk)

  bus_axi_aw = AxiAWBus.from_prefix(dut, AXI_AW_CHANNEL)
  bus_axi_w = AxiWBus.from_prefix(dut, AXI_W_CHANNEL)
  bus_axi_b = AxiBBus.from_prefix(dut, AXI_B_CHANNEL)
  bus_axi_write = AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

  monitor_write_req = XLSChannelMonitor(dut, GENERIC_WRITE_REQ_CHANNEL, dut.clk, WriteRequestStruct)
  monitor_data_in = XLSChannelMonitor(dut, GENERIC_DATA_IN_CHANNEL, dut.clk, WriteRequestStruct)
  monitor_write_resp = XLSChannelMonitor(dut, GENERIC_WRITE_RESP_CHANNEL, dut.clk, MemWriterRespStruct)
  monitor_axi_aw = AxiAWMonitor(bus_axi_aw, dut.clk, dut.rst)
  monitor_axi_w = AxiWMonitor(bus_axi_w, dut.clk, dut.rst)
  monitor_axi_b = AxiBMonitor(bus_axi_b, dut.clk, dut.rst)

  set_termination_event(monitor_write_resp, terminate, resp_cnt)

  memory = AxiRamWrite(bus_axi_write, dut.clk, dut.rst, size=mem_size)

  log = logging.getLogger("cocotb.tb")
  log.setLevel(logging.WARNING)
  memory.log.setLevel(logging.WARNING)

  scoreboard = Scoreboard(dut)
  scoreboard.add_interface(monitor_write_resp, write_resp_expect)

  await reset(dut.clk, dut.rst, cycles=10)
  await cocotb.start(driver_write_req.send(write_req_input))
  await cocotb.start(driver_data_in.send(data_in_input))

  await terminate.wait()

  for bundle in memory_verification:
    memory_contents = bytearray(memory.read(bundle["base_address"], bundle["length"]))
    expected_memory_contents = bytearray(expected_memory.read(bundle["base_address"], bundle["length"]))
    assert memory_contents == expected_memory_contents, "{} bytes of memory contents at base address {}:\n{}\nvs\n{}\nHEXDUMP:\n{}\nvs\n{}".format(hex(bundle["length"]), hex(bundle["base_address"]), memory_contents, expected_memory_contents, memory.hexdump(bundle["base_address"], bundle["length"]), expected_memory.hexdump(bundle["base_address"], bundle["length"]))

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_single_burst_1_transfer(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_single_burst_1_transfer)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_single_burst_2_transfers(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_single_burst_2_transfers)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_single_burst_almost_max_burst_transfer(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_single_burst_almost_max_burst_transfer)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_single_burst_max_burst_transfer(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_single_burst_max_burst_transfer)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_multiburst_2_full_bursts(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_multiburst_2_full_bursts)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_multiburst_1_full_burst_and_single_transfer(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_multiburst_1_full_burst_and_single_transfer)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_multiburst_crossing_4kb_boundary(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_multiburst_crossing_4kb_boundary)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_multiburst_crossing_4kb_boundary_with_perfectly_aligned_full_bursts(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_multiburst_crossing_4kb_boundary_with_perfectly_aligned_full_bursts)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def ram_test_multiburst_crossing_4kb_boundary_with_2_full_bursts_and_1_transfer(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_arbitrary(mem_size, test_cases_multiburst_crossing_4kb_boundary_with_2_full_bursts_and_1_transfer)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=5000, timeout_unit="ms")
async def ram_test_not_full_packets(dut):
  mem_size = 2**ADDR_WIDTH

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_padded_test_data_arbitrary(mem_size, test_cases_not_full_packets)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

@cocotb.test(timeout_time=5000, timeout_unit="ms")
async def ram_test_random(dut):
  mem_size = 2**ADDR_WIDTH
  test_count = 200

  (write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt) = generate_test_data_random(test_count, mem_size)
  await test_writer(dut, mem_size, write_req_input, data_in_input, write_resp_expect, memory_verification, expected_memory, resp_cnt)

def generate_test_data_random(test_count, mem_size):
  AXI_AXSIZE_ENCODING_MAX_4B_TRANSFER = 2 # Must be in sync with AXI_AXSIZE_ENCODING enum in axi.x

  write_req_input = []
  data_in_input = []
  write_resp_expect = []
  memory_verification = []
  memory = SparseMemory(mem_size)

  random.seed(1234)

  xfer_baseaddr = 0

  for i in range(test_count):
    # Generate offset from the absolute address
    max_xfer_offset = mem_size - xfer_baseaddr
    xfer_offset = random.randrange(0, max_xfer_offset)
    xfer_addr = xfer_baseaddr + xfer_offset
    # Make sure we don't write beyond available memory
    memory_size_max_xfer_len = mem_size - xfer_addr
    arbitrary_max_xfer_len = 0x5000 # 20kB
    xfer_max_len = min(arbitrary_max_xfer_len, memory_size_max_xfer_len)
    xfer_len = random.randrange(1, xfer_max_len)

    write_req = WriteReqStruct(
      offset = xfer_offset,
      length = xfer_len,
    )
    write_req_input.append(write_req)

    data_to_write = random.randbytes(xfer_len)
    rem = xfer_len % 4
    for j in list(range(0, xfer_len-3, 4)):
      last = ((j + 4) >= xfer_len) & (rem == 0)
      data_in = DataInStruct(
          data = int.from_bytes(data_to_write[j:j+4], byteorder='little'),
          length = 4,
          last = last
      )
      data_in_input.append(data_in)
    if (rem > 0):
      data_in = DataInStruct(
          data = int.from_bytes(data_to_write[-rem:], byteorder='little'),
          length = rem,
          last = True
      )
      data_in_input.append(data_in)


    transfer_req = WriteRequestStruct(
      address = xfer_addr,
      length = xfer_len,
    )
    write_expected_memory(transfer_req, data_to_write, memory)

    memory_bundle = {
            "base_address": transfer_req.address,
            "length": transfer_req.length,
    }
    memory_verification.append(memory_bundle)

  write_resp_expect = [MemWriterRespStruct(status=MemWriterRespStatus.OKAY.value)] * test_count

  return (write_req_input, data_in_input, write_resp_expect, memory_verification, memory, test_count)

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

def generate_test_data_arbitrary(mem_size, test_cases):
  AXI_AXSIZE_ENCODING_MAX_4B_TRANSFER = 2 # Must be in sync with AXI_AXSIZE_ENCODING enum in axi.x
  test_count = len(test_cases)

  random.seed(1234)

  write_req_input = []
  data_in_input = []
  write_resp_expect = []
  memory_verification = []
  memory = SparseMemory(mem_size)

  xfer_baseaddr = 0x0
  assert xfer_baseaddr < mem_size

  max_xfer_offset = mem_size - xfer_baseaddr

  for xfer_offset, xfer_len in test_cases:
    assert xfer_offset <= max_xfer_offset
    xfer_addr = xfer_baseaddr + xfer_offset
    # Make sure we don't write beyond available memory
    memory_size_max_xfer_len = mem_size - xfer_addr
    arbitrary_max_xfer_len = 0x5000 # 20kB
    xfer_max_len = min(arbitrary_max_xfer_len, memory_size_max_xfer_len)
    assert xfer_len <= xfer_max_len

    write_req = WriteReqStruct(
      offset = xfer_offset,
      length = xfer_len,
    )
    write_req_input.append(write_req)

    data_to_write = random.randbytes(xfer_len)
    rem = xfer_len % 4
    for j in list(range(0, xfer_len-3, 4)):
      last = ((j + 4) >= xfer_len) & (rem == 0)
      data_in = DataInStruct(
          data = int.from_bytes(data_to_write[j:j+4], byteorder='little'),
          length = 4,
          last = last
      )
      data_in_input.append(data_in)
    if (rem > 0):
      data_in = DataInStruct(
          data = int.from_bytes(data_to_write[-rem:], byteorder='little'),
          length = rem,
          last = True
      )
      data_in_input.append(data_in)


    transfer_req = WriteRequestStruct(
      address = xfer_addr,
      length = xfer_len,
    )
    write_expected_memory(transfer_req, data_to_write, memory)

    memory_bundle = {
            "base_address": transfer_req.address,
            "length": transfer_req.length,
    }
    memory_verification.append(memory_bundle)

  write_resp_expect = [MemWriterRespStruct(status=MemWriterRespStatus.OKAY.value)] * test_count

  return (write_req_input, data_in_input, write_resp_expect, memory_verification, memory, test_count)

def generate_padded_test_data_arbitrary(mem_size, test_cases):
  AXI_AXSIZE_ENCODING_MAX_4B_TRANSFER = 2 # Must be in sync with AXI_AXSIZE_ENCODING enum in axi.x
  test_count = len(test_cases)

  random.seed(1234)

  write_req_input = []
  data_in_input = []
  write_resp_expect = []
  memory_verification = []
  memory = SparseMemory(mem_size)

  xfer_baseaddr = 0x0
  assert xfer_baseaddr < mem_size

  max_xfer_offset = mem_size - xfer_baseaddr

  for xfer_offset, xfer_len in test_cases:
    assert xfer_offset <= max_xfer_offset
    xfer_addr = xfer_baseaddr + xfer_offset
    # Make sure we don't write beyond available memory
    memory_size_max_xfer_len = mem_size - xfer_addr
    arbitrary_max_xfer_len = 0x5000 # 20kB
    xfer_max_len = min(arbitrary_max_xfer_len, memory_size_max_xfer_len)
    assert xfer_len <= xfer_max_len

    write_req = WriteReqStruct(
      offset = xfer_offset,
      length = xfer_len,
    )
    write_req_input.append(write_req)

    data_to_write = random.randbytes(xfer_len)
    bytes_to_packetize = xfer_len
    packetized_bytes = 0
    while(bytes_to_packetize):
      packet_len = random.randint(1, 4)

      if (bytes_to_packetize < packet_len):
        packet_len = bytes_to_packetize

      last = packet_len == bytes_to_packetize

      data_in = DataInStruct(
          data = int.from_bytes(data_to_write[packetized_bytes:packetized_bytes+packet_len], byteorder='little'),
          length = packet_len,
          last = last
      )
      data_in_input.append(data_in)

      bytes_to_packetize -= packet_len
      packetized_bytes += packet_len
    assert xfer_len == packetized_bytes


    transfer_req = WriteRequestStruct(
      address = xfer_addr,
      length = xfer_len,
    )
    write_expected_memory(transfer_req, data_to_write, memory)

    memory_bundle = {
            "base_address": transfer_req.address,
            "length": transfer_req.length,
    }
    memory_verification.append(memory_bundle)

  write_resp_expect = [MemWriterRespStruct(status=MemWriterRespStatus.OKAY.value)] * test_count

  return (write_req_input, data_in_input, write_resp_expect, memory_verification, memory, test_count)

if __name__ == "__main__":
    toplevel = "mem_writer_wrapper"
    verilog_sources = [
      "xls/modules/zstd/memory/mem_writer.v",
      "xls/modules/zstd/memory/mem_writer_wrapper.v",
    ]
    test_module=[Path(__file__).stem]
    run_test(toplevel, test_module, verilog_sources)

test_cases_single_burst_1_transfer = [
  # Aligned Address; Aligned Length
  (0x0, 0x4),
  # Aligned Address; Unaligned Length
  (0x10, 0x1),
  (0x24, 0x2),
  (0x38, 0x3),
  # Unaligned Address; Aligned Length
  (0x41, 0x4),
  (0x52, 0x4),
  (0x63, 0x4),
  # Unaligned Address; Unaligned Length
  (0x71, 0x1),
  (0x81, 0x2),
  (0x91, 0x3),
  (0xa2, 0x1),
  (0xb2, 0x2),
  (0xc2, 0x3),
  (0xd3, 0x1),
  (0xe3, 0x2),
  (0xf3, 0x3)
]

test_cases_single_burst_2_transfers = [
  # Aligned Address; Aligned Length
  (0x100, 0x8),
  # Aligned Address; Unaligned Length
  (0x110, 0x5),
  (0x120, 0x6),
  (0x130, 0x7),
  # Unaligned Address; Aligned Length
  (0x141, 0x8),
  (0x152, 0x8),
  (0x163, 0x8),
  # Unaligned Address; Unaligned Length
  (0x171, 0x5),
  (0x182, 0x5),
  (0x193, 0x5),
  (0x1A1, 0x6),
  (0x1B2, 0x6),
  (0x1C3, 0x6),
  (0x1D1, 0x7),
  (0x1E2, 0x7),
  (0x1F3, 0x7)
]

test_cases_single_burst_almost_max_burst_transfer = [
  # Aligned Address; Aligned Length
  (0x200, 0x3FC),
  # Aligned Address; Unaligned Length
  (0x600, 0x3F9),
  (0xA00, 0x3FA),
  (0x1000, 0x3FB),
  # Unaligned Address; Aligned Length
  (0x1401, 0x3FC),
  (0x1802, 0x3FC),
  (0x2003, 0x3FC),
  # Unaligned Address; Unaligned Length
  (0x2401, 0x3F9),
  (0x2802, 0x3F9),
  (0x2C03, 0x3F9),
  (0x3001, 0x3FA),
  (0x3402, 0x3FA),
  (0x3803, 0x3FA),
  (0x3C01, 0x3FB),
  (0x4002, 0x3FB),
  (0x4403, 0x3FB)
]

test_cases_single_burst_max_burst_transfer = [
  # Aligned Address; Aligned Length
  (0x4800, 0x400),
  # Aligned Address; Unaligned Length
  (0x4C00, 0x3FD),
  (0x5000, 0x3FE),
  (0x5400, 0x3FF),
  # Unaligned Address; Aligned Length
  (0x5801, 0x400),
  (0x6002, 0x400),
  (0x6803, 0x400),
  # Unaligned Address; Unaligned Length
  (0x7001, 0x3FD),
  (0x7802, 0x3FD),
  (0x8003, 0x3FD),
  (0x8801, 0x3FE),
  (0x9002, 0x3FE),
  (0x9803, 0x3FE),
  (0xA001, 0x3FF),
  (0xA802, 0x3FF),
  (0xB003, 0x3FF)
]

test_cases_multiburst_2_full_bursts = [
  # Aligned Address; Aligned Length
  (0x0400, 0x800),
  # Aligned Address; Unaligned Length
  (0x1000, 0x7FD),
  (0x1800, 0x7FE),
  (0x2000, 0x7FF),
  # Unaligned Address; Aligned Length
  (0x2801, 0x800),
  (0x3002, 0x800),
  (0x3803, 0x800),
  # Unaligned Address; Unaligned Length
  (0x4001, 0x7FD),
  (0x5002, 0x7FD),
  (0x6003, 0x7FD),
  (0x7001, 0x7FE),
  (0x8002, 0x7FE),
  (0x9003, 0x7FE),
  (0xA001, 0x7FF),
  (0xB002, 0x7FF),
  (0xF003, 0x7FF)
]

test_cases_multiburst_1_full_burst_and_single_transfer = [
  # Aligned Address; Aligned Length; Multi-Burst
  (0x0000, 0x404),
  # Aligned Address; Unaligned Length; Multi-Burst
  (0x0800, 0x401),
  (0x1000, 0x402),
  (0x1800, 0x403),
  # Unaligned Address; Aligned Length; Multi-Burst
  (0x2000, 0x404),
  (0x2800, 0x404),
  (0x3000, 0x404),
  # Unaligned Address; Unaligned Length; Multi-Burst
  (0x3801, 0x401),
  (0x5002, 0x401),
  (0x5803, 0x401),
  (0x6001, 0x402),
  (0x6802, 0x402),
  (0x7003, 0x402),
  (0x7801, 0x403),
  (0x8002, 0x403),
  (0x8803, 0x403)
]

test_cases_multiburst_crossing_4kb_boundary = [
  # Aligned Address; Aligned Length
  (0x0FFC, 0x8),
  # Aligned Address; Unaligned Length
  (0x1FFC, 0x5),
  (0x2FFC, 0x6),
  (0x3FFC, 0x7),
  # Unaligned Address; Aligned Length
  (0x4FFD, 0x8),
  (0x5FFE, 0x8),
  (0x6FFF, 0x8),
  # Unaligned Address; Unaligned Length
  (0x7FFD, 0x5),
  (0x8FFD, 0x6),
  (0x9FFD, 0x7),
  (0xAFFE, 0x5),
  (0xBFFE, 0x6),
  (0xCFFE, 0x7),
  (0xDFFF, 0x5),
  (0xEFFF, 0x6),
  # End of address space - wrap around
  (0x0FFF, 0x7),
]

test_cases_multiburst_crossing_4kb_boundary_with_perfectly_aligned_full_bursts = [
  # Aligned Address; Aligned Length; Multi-Burst; crossing 4kB boundary with perfectly aligned full bursts
  (0x0C00, 0x800),
  # Unaligned Address; Unaligned Length; Multi-Burst; crossing 4kB boundary with perfectly aligned full bursts
  (0x1C01, 0x7FF),
  (0x2C02, 0x7FE),
  (0x3C03, 0x7FD),
]

test_cases_multiburst_crossing_4kb_boundary_with_2_full_bursts_and_1_transfer = [
  # Aligned Address; Aligned Length
  (0x0C04, 0x800),
  # Aligned Address; Unaligned Length
  (0x1C04, 0x801),
  (0x2C04, 0x802),
  (0x3C04, 0x803),
  # Unaligned Address; Aligned Length
  (0x4C01, 0x800),
  (0x5C02, 0x800),
  (0x6C03, 0x800),
  # Unaligned Address; Unaligned Length
  (0x7C01, 0x801),
  (0x8C02, 0x802),
  (0x9C03, 0x803),
  (0xAC01, 0x802),
  (0xBC02, 0x802),
  (0xCC03, 0x802),
  (0xDC01, 0x803),
  (0xEC02, 0x803),
  # End of address space - wrap around
  (0x0C03, 0x803),
]

test_cases_not_full_packets = [
  # Aligned Address; Aligned Length
  (0x0000, 0x20),
  # Aligned Address; Unaligned Length
  (0x100, 0x21),
  (0x200, 0x22),
  (0x300, 0x23),
  # Unaligned Address; Aligned Length
  (0x401, 0x20),
  (0x502, 0x20),
  (0x603, 0x20),
  # Unaligned Address; Unaligned Length
  (0x701, 0x21),
  (0x802, 0x22),
  (0x903, 0x23),
  (0xA01, 0x22),
  (0xB02, 0x22),
  (0xC03, 0x22),
  (0xD01, 0x23),
  (0xE02, 0x23),
  (0xF03, 0x23),
]
