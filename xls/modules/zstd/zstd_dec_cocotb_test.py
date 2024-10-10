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


from enum import Enum
from pathlib import Path
import tempfile

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Event
from cocotb.binary import BinaryValue
from cocotb_bus.scoreboard import Scoreboard

from cocotbext.axi.axi_master import AxiMaster
from cocotbext.axi.axi_channels import AxiAWBus, AxiWBus, AxiBBus, AxiWriteBus, AxiARBus, AxiRBus, AxiReadBus, AxiBus, AxiBTransaction, AxiBSource, AxiBSink, AxiBMonitor, AxiRTransaction, AxiRSource, AxiRSink, AxiRMonitor
from cocotbext.axi.axi_ram import AxiRam
from cocotbext.axi.sparse_memory import SparseMemory

import zstandard

from xls.common import runfiles
from xls.modules.zstd.cocotb.channel import (
  XLSChannel,
  XLSChannelDriver,
  XLSChannelMonitor,
)
from xls.modules.zstd.cocotb.data_generator import GenerateFrame, BlockType
from xls.modules.zstd.cocotb.memory import init_axi_mem, AxiRamFromFile
from xls.modules.zstd.cocotb.utils import reset, run_test
from xls.modules.zstd.cocotb.xlsstruct import XLSStruct, xls_dataclass

MAX_ENCODED_FRAME_SIZE_B = 16384

# Override default widths of AXI response signals
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
class NotifyStruct(XLSStruct):
  pass

@xls_dataclass
class OutputStruct(XLSStruct):
  data: 64
  length: 32
  last: 1

class CSR(Enum):
  """
  Maps the offsets to the ZSTD Decoder registers
  """
  Status = 0x0
  Start = 0x4
  Reset = 0x8
  InputBuffer = 0xC
  OutputBuffer = 0x10
  WhoAmI = 0x14

class Status(Enum):
  """
  Codes for the Status register
  """
  IDLE = 0x0
  RUNNING = 0x1

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
  bus_axi_w = AxiWBus.from_prefix(dut, name + AXI_W)
  bus_axi_b = AxiBBus.from_prefix(dut, name + AXI_B)

  return AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

def connect_axi_bus(dut, name=""):
  bus_axi_read = connect_axi_read_bus(dut, name)
  bus_axi_write = connect_axi_write_bus(dut, name)

  return AxiBus(bus_axi_write, bus_axi_read)

def get_decoded_frame_bytes(ifh):
  dctx = zstandard.ZstdDecompressor()
  return dctx.decompress(ifh.read())

def get_decoded_frame_buffer(ifh, address, memory=SparseMemory(size=MAX_ENCODED_FRAME_SIZE_B)):
  dctx = zstandard.ZstdDecompressor()
  memory.write(address, dctx.decompress(ifh.read()))
  return memory

async def csr_write(cpu, csr, data):
  if type(data) is int:
    data = data.to_bytes(4, byteorder='little')
  assert len(data) <= 4
  await cpu.write(csr.value, data)

async def csr_read(cpu, csr):
  return await cpu.read(csr.value, 4)

async def test_csr(dut):

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  dut.rst.setimmediatevalue(0)
  await ClockCycles(dut.clk, 20)
  dut.rst.setimmediatevalue(1)
  await ClockCycles(dut.clk, 20)
  dut.rst.setimmediatevalue(0)

  csr_bus = connect_axi_bus(dut, "csr")

  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  await ClockCycles(dut.clk, 10)
  i = 0
  for reg in CSR:
    expected = bytearray.fromhex("EFBEADDE")
    expected[0] += i
    await csr_write(cpu, reg, expected)
    read = await csr_read(cpu, reg)
    assert read.data == expected
    i += 1
  await ClockCycles(dut.clk, 10)

async def configure_decoder(cpu, ibuf_addr, obuf_addr):
  status = await csr_read(cpu, CSR.Status)
  if int.from_bytes(status.data, byteorder='little') != Status.IDLE.value:
    await csr_write(cpu, CSR.Reset, 0x1)
  await csr_write(cpu, CSR.InputBuffer, ibuf_addr)
  await csr_write(cpu, CSR.OutputBuffer, obuf_addr)

async def start_decoder(cpu):
  await csr_write(cpu, CSR.Start, 0x1)

async def wait_for_idle(cpu):
  status = await csr_read(cpu, CSR.Status)
  while (int.from_bytes(status.data, byteorder='little') != Status.IDLE.value):
    status = await csr_read(cpu, CSR.Status)

def generate_expected_output(decoded_frame):
  packets = []
  frame_len = len(decoded_frame)
  last_len = frame_len % 8
  for i in range(frame_len // 8):
    start_id = i * 8
    end_id = start_id + 8
    packet_data = int.from_bytes(decoded_frame[start_id:end_id], byteorder='little')
    last_packet = (end_id==frame_len)
    packet = OutputStruct(data=packet_data, length=64, last=last_packet)
    packets.append(packet)
  if (last_len):
    packet_data = int.from_bytes(decoded_frame[-last_len:], byteorder='little')
    packet = OutputStruct(data=packet_data, length=last_len*8, last=True)
    packets.append(packet)
  return packets

async def reset_dut(dut, rst_len=10):
  dut.rst.setimmediatevalue(0)
  await ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(1)
  await ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(0)


async def test_decoder(dut, seed, block_type):
  NOTIFY_CHANNEL = "notify"
  OUTPUT_CHANNEL = "output"

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  memory_bus = connect_axi_bus(dut, "memory")
  csr_bus = connect_axi_bus(dut, "csr")
  notify_channel = XLSChannel(dut, NOTIFY_CHANNEL, dut.clk, start_now=True)
  monitor_notify = XLSChannelMonitor(dut, NOTIFY_CHANNEL, dut.clk, NotifyStruct)

  output_channel = XLSChannel(dut, OUTPUT_CHANNEL, dut.clk, start_now=True)
  monitor_output = XLSChannelMonitor(dut, OUTPUT_CHANNEL, dut.clk, OutputStruct)

  scoreboard = Scoreboard(dut)

  assert_notify = Event()
  set_termination_event(monitor_notify, assert_notify, 1)

  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  mem_size = MAX_ENCODED_FRAME_SIZE_B
  ibuf_addr = 0x0
  obuf_addr = mem_size // 2

  #FIXME: use delete_on_close=False after moving to python 3.12
  with tempfile.NamedTemporaryFile(delete=False) as encoded:
    await reset_dut(dut, 50)

    # Generate ZSTD frame to temporary file
    GenerateFrame(seed, block_type, encoded.name)

    expected_decoded_frame = get_decoded_frame_bytes(encoded)
    encoded.close()
    expected_output_packets = generate_expected_output(expected_decoded_frame)

    assert_expected_output = Event()
    set_termination_event(monitor_output, assert_expected_output, len(expected_output_packets))
    # Monitor stream output for packets with the decoded ZSTD frame
    scoreboard.add_interface(monitor_output, expected_output_packets)

    # Initialise testbench memory with generated ZSTD frame
    memory = AxiRamFromFile(bus=memory_bus, clock=dut.clk, reset=dut.rst, path=encoded.name, size=mem_size)

    await configure_decoder(cpu, ibuf_addr, obuf_addr)
    await start_decoder(cpu)
    await assert_expected_output.wait()
    await wait_for_idle(cpu)
    # TODO: Check decoded frame in memory under `obuf_addr` when ZstdDecoder
    #       will fully support memory output interface

  await assert_notify.wait()
  await ClockCycles(dut.clk, 20)

@cocotb.test(timeout_time=50, timeout_unit="ms")
async def zstd_csr_test(dut):
  await test_csr(dut)

#FIXME: Rework testbench to decode multiple ZSTD frames in one test
@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_raw_frames_test_1(dut):
  block_type = BlockType.RAW
  await test_decoder(dut, 1, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_raw_frames_test_2(dut):
  block_type = BlockType.RAW
  await test_decoder(dut, 2, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_raw_frames_test_3(dut):
  block_type = BlockType.RAW
  await test_decoder(dut, 3, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_raw_frames_test_4(dut):
  block_type = BlockType.RAW
  await test_decoder(dut, 4, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_raw_frames_test_5(dut):
  block_type = BlockType.RAW
  await test_decoder(dut, 5, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_rle_frames_test_1(dut):
  block_type = BlockType.RLE
  await test_decoder(dut, 1, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_rle_frames_test_2(dut):
  block_type = BlockType.RLE
  await test_decoder(dut, 2, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_rle_frames_test_3(dut):
  block_type = BlockType.RLE
  await test_decoder(dut, 3, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_rle_frames_test_4(dut):
  block_type = BlockType.RLE
  await test_decoder(dut, 4, block_type)

@cocotb.test(timeout_time=20000, timeout_unit="ms")
async def zstd_rle_frames_test_5(dut):
  block_type = BlockType.RLE
  await test_decoder(dut, 5, block_type)

#@cocotb.test(timeout_time=20000, timeout_unit="ms")
#async def zstd_compressed_frames_test(dut):
#  test_cases = 1
#  block_type = BlockType.COMPRESSED
#  await test_decoder(dut, test_cases, block_type)
#
#@cocotb.test(timeout_time=20000, timeout_unit="ms")
#async def zstd_random_frames_test(dut):
#  test_cases = 1
#  block_type = BlockType.RANDOM
#  await test_decoder(dut, test_cases, block_type)

if __name__ == "__main__":
  toplevel = "zstd_dec_wrapper"
  verilog_sources = [
    "xls/modules/zstd/zstd_dec.v",
    "xls/modules/zstd/xls_fifo_wrapper.v",
    "xls/modules/zstd/zstd_dec_wrapper.v",
    "xls/modules/zstd/axi_interconnect_wrapper.v",
    "xls/modules/zstd/axi_interconnect.v",
    "xls/modules/zstd/arbiter.v",
    "xls/modules/zstd/priority_encoder.v",
  ]
  test_module=[Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)
