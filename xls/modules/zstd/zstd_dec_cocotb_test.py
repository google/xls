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


import enum
import pathlib
import tempfile

import cocotb
from cocotb import triggers
from cocotb.clock import Clock

from cocotbext.axi import axi_channels
from cocotbext.axi.axi_master import AxiMaster
from cocotbext.axi.sparse_memory import SparseMemory

import xls.modules.zstd.cocotb.channel as xlschannel
from xls.modules.zstd.cocotb import data_generator
from xls.modules.zstd.cocotb.memory import AxiRamFromFile
from xls.modules.zstd.cocotb.utils import run_test
from xls.modules.zstd.cocotb import xlsstruct

AXI_DATA_W = 64
AXI_DATA_W_BYTES = AXI_DATA_W // 8
MAX_ENCODED_FRAME_SIZE_B = 16384
NOTIFY_CHANNEL = "notify"
RESET_CHANNEL = "reset"

# Override default widths of AXI response signals
signal_widths = {"bresp": 3}
axi_channels.AxiBBus._signal_widths = signal_widths
axi_channels.AxiBTransaction._signal_widths = signal_widths
axi_channels.AxiBSource._signal_widths = signal_widths
axi_channels.AxiBSink._signal_widths = signal_widths
axi_channels.AxiBMonitor._signal_widths = signal_widths
signal_widths = {"rresp": 3, "rlast": 1}
axi_channels.AxiRBus._signal_widths = signal_widths
axi_channels.AxiRTransaction._signal_widths = signal_widths
axi_channels.AxiRSource._signal_widths = signal_widths
axi_channels.AxiRSink._signal_widths = signal_widths
axi_channels.AxiRMonitor._signal_widths = signal_widths

@xlsstruct.xls_dataclass
class NotifyStruct(xlsstruct.XLSStruct):
  pass

class CSR(enum.Enum):
  """
  Maps the offsets to the ZSTD Decoder registers.
  """
  STATUS = 0
  START = 1
  INPUTBUFFER = 2
  OUTPUTBUFFER = 3

class Status(enum.Enum):
  """
  Codes for the Status register.
  """
  IDLE = 0x0
  RUNNING = 0x1

def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()
  monitor.add_callback(terminate_cb)

def connect_axi_read_bus(dut, name=""):
  axi_ar = "axi_ar"
  axi_r = "axi_r"

  if name:
      name += "_"

  bus_axi_ar = axi_channels.AxiARBus.from_prefix(dut, name + axi_ar)
  bus_axi_r = axi_channels.AxiRBus.from_prefix(dut, name + axi_r)

  return axi_channels.AxiReadBus(bus_axi_ar, bus_axi_r)

def connect_axi_write_bus(dut, name=""):
  axi_aw = "axi_aw"
  axi_w = "axi_w"
  axi_b = "axi_b"

  if name:
      name += "_"

  bus_axi_aw = axi_channels.AxiAWBus.from_prefix(dut, name + axi_aw)
  bus_axi_w = axi_channels.AxiWBus.from_prefix(dut, name + axi_w)
  bus_axi_b = axi_channels.AxiBBus.from_prefix(dut, name + axi_b)

  return axi_channels.AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

def connect_axi_bus(dut, name=""):
  bus_axi_read = connect_axi_read_bus(dut, name)
  bus_axi_write = connect_axi_write_bus(dut, name)

  return axi_channels.AxiBus(bus_axi_write, bus_axi_read)

async def csr_write(cpu, csr, data):
  if isinstance(data, int):
    data = data.to_bytes(AXI_DATA_W_BYTES, byteorder='little')
  assert len(data) <= AXI_DATA_W_BYTES
  await cpu.write(csr.value * AXI_DATA_W_BYTES, data)

async def csr_read(cpu, csr):
    return await cpu.read(csr.value * AXI_DATA_W_BYTES, AXI_DATA_W_BYTES)

async def test_csr(dut):

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  await reset_dut(dut, 50)

  csr_bus = connect_axi_bus(dut, "csr")

  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  await triggers.ClockCycles(dut.clk, 10)
  i = 0
  for reg in CSR:
    expected_src = bytearray.fromhex("0DF0AD8BEFBEADDE")
    assert len(expected_src) >= AXI_DATA_W_BYTES
    expected = expected_src[-AXI_DATA_W_BYTES:]
    expected[0] += i
    await csr_write(cpu, reg, expected)
    read = await csr_read(cpu, reg)
    assert read.data == expected, (
      "Expected data doesn't match contents of the {}".format(reg)
    )
    i += 1
  await triggers.ClockCycles(dut.clk, 10)

async def test_reset(dut):
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  await reset_dut(dut, 50)

  csr_bus = connect_axi_bus(dut, "csr")
  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  await triggers.ClockCycles(dut.clk, 10)
  await start_decoder(cpu)
  timeout = 10
  status = await csr_read(cpu, CSR.STATUS)
  while ((int.from_bytes(status.data, byteorder='little') == Status.IDLE.value)
      & (timeout != 0)):
    status = await csr_read(cpu, CSR.STATUS)
    timeout -= 1
  assert (timeout != 0)

  await reset_dut(dut, 50)
  await wait_for_idle(cpu, 10)

  await triggers.ClockCycles(dut.clk, 10)

async def configure_decoder(dut, cpu, ibuf_addr, obuf_addr):
  status = await csr_read(cpu, CSR.STATUS)
  if int.from_bytes(status.data, byteorder='little') != Status.IDLE.value:
    await reset_dut(dut, 50)
  await csr_write(cpu, CSR.INPUTBUFFER, ibuf_addr)
  await csr_write(cpu, CSR.OUTPUTBUFFER, obuf_addr)

async def start_decoder(cpu):
  await csr_write(cpu, CSR.START, 0x1)

async def wait_for_idle(cpu, timeout=100):
  status = await csr_read(cpu, CSR.STATUS)
  while ((int.from_bytes(status.data, byteorder='little') != Status.IDLE.value)
      & (timeout != 0)):
    status = await csr_read(cpu, CSR.STATUS)
    timeout -= 1
  assert (timeout != 0)

async def reset_dut(dut, rst_len=10):
  dut.rst.setimmediatevalue(0)
  await triggers.ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(1)
  await triggers.ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(0)

def connect_xls_channel(dut, channel_name, xls_struct):
  channel = xlschannel.XLSChannel(dut, channel_name, dut.clk, start_now=True)
  monitor = xlschannel.XLSChannelMonitor(dut, channel_name, dut.clk, xls_struct)

  return (channel, monitor)

def prepare_test_environment(dut):
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  memory_bus = connect_axi_bus(dut, "memory")
  csr_bus = connect_axi_bus(dut, "csr")
  axi_buses = {
      "memory": memory_bus,
      "csr": csr_bus
  }

  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  return (axi_buses, cpu)

async def test_decoder(dut, seed, block_type, axi_buses, cpu):
  memory_bus = axi_buses["memory"]

  (unused_notify_channel, notify_monitor) = connect_xls_channel(
    dut, NOTIFY_CHANNEL, NotifyStruct
  )
  assert_notify = triggers.Event()
  set_termination_event(notify_monitor, assert_notify, 1)

  mem_size = MAX_ENCODED_FRAME_SIZE_B
  ibuf_addr = 0x0
  obuf_addr = mem_size // 2

  #FIXME: use delete_on_close=False after moving to python 3.12
  with tempfile.NamedTemporaryFile(delete=False) as encoded:
    await reset_dut(dut, 50)

    # Generate ZSTD frame to temporary file
    data_generator.GenerateFrame(seed, block_type, encoded.name)

    expected_decoded_frame = data_generator.DecompressFrame(encoded.read())
    encoded.close()
    reference_memory = SparseMemory(mem_size)
    reference_memory.write(obuf_addr, expected_decoded_frame)

    # Initialise testbench memory with generated ZSTD frame
    memory = AxiRamFromFile(
      bus=memory_bus,
      clock=dut.clk,
      reset=dut.rst,
      path=encoded.name,
      size=mem_size
    )

    await configure_decoder(dut, cpu, ibuf_addr, obuf_addr)
    await start_decoder(cpu)
    await assert_notify.wait()
    await wait_for_idle(cpu)
    # Read decoded frame in chunks of AXI_DATA_W length
    # Compare against frame decompressed with the reference library
    chunks_nb = (
      len(expected_decoded_frame) + (AXI_DATA_W_BYTES - 1)
    ) // AXI_DATA_W_BYTES
    for read_op in range(0, chunks_nb):
      addr = obuf_addr + (read_op * AXI_DATA_W_BYTES)
      mem_contents = memory.read(addr, AXI_DATA_W_BYTES)
      exp_mem_contents = reference_memory.read(addr, AXI_DATA_W_BYTES)
      assert mem_contents == exp_mem_contents, (
        (
          "{} bytes of memory contents at address {} "
          "don't match the expected contents:\n"
          "{}\nvs\n{}"
        ).format(
          AXI_DATA_W_BYTES,
          hex(addr),
          hex(int.from_bytes(mem_contents, byteorder='little')),
          hex(int.from_bytes(exp_mem_contents, byteorder='little'))
        )
      )

  await triggers.ClockCycles(dut.clk, 20)

async def testing_routine(
  dut,
  test_cases=1,
  block_type=data_generator.BlockType.RANDOM
):
  (axi_buses, cpu) = prepare_test_environment(dut)
  for test_case in range(test_cases):
    await test_decoder(dut, test_case, block_type, axi_buses, cpu)
  print("Decoding {} ZSTD frames done".format(block_type.name))

@cocotb.test(timeout_time=50, timeout_unit="ms")
async def zstd_csr_test(dut):
  await test_csr(dut)

@cocotb.test(timeout_time=50, timeout_unit="ms")
async def zstd_reset_test(dut):
  await test_reset(dut)

@cocotb.test(timeout_time=500, timeout_unit="ms")
async def zstd_raw_frames_test(dut):
  test_cases = 5
  block_type = data_generator.BlockType.RAW
  await testing_routine(dut, test_cases, block_type)

@cocotb.test(timeout_time=500, timeout_unit="ms")
async def zstd_rle_frames_test(dut):
  test_cases = 5
  block_type = data_generator.BlockType.RLE
  await testing_routine(dut, test_cases, block_type)

#@cocotb.test(timeout_time=1000, timeout_unit="ms")
#async def zstd_compressed_frames_test(dut):
#  test_cases = 1
#  block_type = BlockType.COMPRESSED
#  await testing_routine(dut, test_cases, block_type)

#@cocotb.test(timeout_time=1000, timeout_unit="ms")
#async def zstd_random_frames_test(dut):
#  test_cases = 1
#  block_type = BlockType.RANDOM
#  await testing_routine(dut, test_cases, block_type)

if __name__ == "__main__":
  toplevel = "zstd_dec_wrapper"
  verilog_sources = [
    "xls/modules/zstd/zstd_dec.v",
    "xls/modules/zstd/rtl/xls_fifo_wrapper.v",
    "xls/modules/zstd/rtl/zstd_dec_wrapper.v",
    "third_party/verilog_axi/axi_crossbar_wrapper.v",
    "third_party/verilog_axi/axi_crossbar.v",
    "third_party/verilog_axi/axi_crossbar_rd.v",
    "third_party/verilog_axi/axi_crossbar_wr.v",
    "third_party/verilog_axi/axi_crossbar_addr.v",
    "third_party/verilog_axi/axi_register_rd.v",
    "third_party/verilog_axi/axi_register_wr.v",
    "third_party/verilog_axi/arbiter.v",
    "third_party/verilog_axi/priority_encoder.v",
  ]
  test_module=[pathlib.Path(__file__).stem]
  run_test(toplevel, test_module, verilog_sources)
