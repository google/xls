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
from cocotb.triggers import RisingEdge, ClockCycles, Event
from cocotb.binary import BinaryValue
from cocotb.utils import get_sim_time
from cocotb_bus.scoreboard import Scoreboard

from cocotbext.axi import axi_channels
from cocotbext.axi.axi_master import AxiMaster
from cocotbext.axi.axi_channels import (
  AxiAWBus,
  AxiWBus,
  AxiWMonitor,
  AxiBBus,
  AxiWriteBus,
  AxiARBus,
  AxiRBus,
  AxiReadBus,
  AxiBus,
  AxiBTransaction,
  AxiBSource,
  AxiBSink,
  AxiBMonitor,
  AxiRTransaction,
  AxiRSource,
  AxiRSink,
  AxiRMonitor,
)
from cocotbext.axi.axi_ram import AxiRam
from cocotbext.axi.sparse_memory import SparseMemory

import xls.modules.zstd.cocotb.channel as xlschannel
from xls.modules.zstd.cocotb import data_generator
from xls.modules.zstd.cocotb.memory import AxiRamFromFile
from xls.modules.zstd.cocotb.utils import run_test
from xls.modules.zstd.cocotb import xlsstruct

AXI_DATA_W = 64
AXI_DATA_W_BYTES = AXI_DATA_W // 8
MAX_ENCODED_FRAME_SIZE_B = 2**32
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


SYMBOL_W = 8
NUM_OF_BITS_W = 8
BASE_W = 16


@xlsstruct.xls_dataclass
class FseTableRecord(xlsstruct.XLSStruct):
  base: BASE_W
  num_of_bits: NUM_OF_BITS_W
  symbol: SYMBOL_W


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


def check_ram_contents(mem, expected, name=""):
  for i, value in enumerate(expected):
    assert mem[i].value == value


def print_fse_ram_contents(mem, name="", size=None):
  for i in range(size):
    data = FseTableRecord.from_int(mem[i].value)
    print(f"{name} [{i}]: {data}")


def print_ram_contents(mem, name="", size=None):
  for i in range(size):
    print(f"{name} [{i}]\t: {hex(mem[i].value)}")


def set_termination_event(monitor, event, transactions):
  def terminate_cb(_):
    if monitor.stats.received_transactions == transactions:
      event.set()

  monitor.add_callback(terminate_cb)


@cocotb.coroutine
async def set_handshake_event(clk, channel, event):
  while True:
    await RisingEdge(clk)
    if channel.rdy.value and channel.vld.value:
      event.set()


@cocotb.coroutine
async def get_handshake_event(dut, event, func):
  while True:
    await event.wait()
    func()
    event.clear()


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
    data = data.to_bytes(AXI_DATA_W_BYTES, byteorder="little")
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
    assert (
      read.data == expected
    ), "Expected data doesn't match contents of the {}".format(reg)
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
  while (int.from_bytes(status.data, byteorder="little") == Status.IDLE.value) & (
    timeout != 0
  ):
    status = await csr_read(cpu, CSR.STATUS)
    timeout -= 1
  assert timeout != 0

  await reset_dut(dut, 50)
  await wait_for_idle(cpu, 10)

  await triggers.ClockCycles(dut.clk, 10)


async def configure_decoder(dut, cpu, ibuf_addr, obuf_addr):
  status = await csr_read(cpu, CSR.STATUS)
  if int.from_bytes(status.data, byteorder="little") != Status.IDLE.value:
    await reset_dut(dut, 50)
  await csr_write(cpu, CSR.INPUTBUFFER, ibuf_addr)
  await csr_write(cpu, CSR.OUTPUTBUFFER, obuf_addr)


async def start_decoder(cpu):
  await csr_write(cpu, CSR.START, 0x1)


async def wait_for_idle(cpu, timeout=100):
  status = await csr_read(cpu, CSR.STATUS)
  while (int.from_bytes(status.data, byteorder="little") != Status.IDLE.value) & (
    timeout != 0
  ):
    status = await csr_read(cpu, CSR.STATUS)
    timeout -= 1
  assert timeout != 0


async def reset_dut(dut, rst_len=10):
  dut.rst.setimmediatevalue(0)
  await triggers.ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(1)
  await triggers.ClockCycles(dut.clk, rst_len)
  dut.rst.setimmediatevalue(0)


def get_clock_time(clock: Clock):
  return get_sim_time(units="step") / clock.period


def connect_xls_channel(dut, channel_name, xls_struct):
  channel = xlschannel.XLSChannel(dut, channel_name, dut.clk, start_now=True)
  monitor = xlschannel.XLSChannelMonitor(dut, channel_name, dut.clk, xls_struct)

  return (channel, monitor)


def prepare_test_environment(dut):
  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  memory_bus = connect_axi_bus(dut, "memory")
  csr_bus = connect_axi_bus(dut, "csr")
  axi_buses = {"memory": memory_bus, "csr": csr_bus}

  cpu = AxiMaster(csr_bus, dut.clk, dut.rst)

  return (axi_buses, cpu, clock)


async def test_fse_lookup_decoder(dut, clock, expected_fse_lookups):
  lookup_dec_resp_channel = xlschannel.XLSChannel(
    dut.ZstdDecoder.xls_modules_zstd_sequence_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__SequenceDecoderCtrl_0__FseLookupCtrl_0_next_inst148,
    "zstd_dec__flc_resp",
    dut.clk,
  )
  fse_lookup_resp_handshake = triggers.Event()

  block_cnt = 0

  def func():
    nonlocal block_cnt
    assert block_cnt <= len(expected_fse_lookups)
    print_fse_ram_contents(
      dut.ll_fse_ram.mem, "LL", size=len(expected_fse_lookups[block_cnt]["ll"])
    )
    print_fse_ram_contents(
      dut.ml_fse_ram.mem, "ML", size=len(expected_fse_lookups[block_cnt]["ml"])
    )
    print_fse_ram_contents(
      dut.of_fse_ram.mem, "OF", size=len(expected_fse_lookups[block_cnt]["of"])
    )
    check_ram_contents(
      dut.ll_fse_ram.mem, [x.value for x in expected_fse_lookups[block_cnt]["ll"]]
    )
    check_ram_contents(
      dut.ml_fse_ram.mem, [x.value for x in expected_fse_lookups[block_cnt]["ml"]]
    )
    check_ram_contents(
      dut.of_fse_ram.mem, [x.value for x in expected_fse_lookups[block_cnt]["of"]]
    )
    block_cnt += 1

  cocotb.start_soon(
    set_handshake_event(dut.clk, lookup_dec_resp_channel, fse_lookup_resp_handshake)
  )
  cocotb.start_soon(get_handshake_event(dut, fse_lookup_resp_handshake, func))


async def test_fse_lookup_decoder_for_huffman(dut, clock, expected_fse_lookups):
  lookup_dec_resp_channel = xlschannel.XLSChannel(
    dut.ZstdDecoder.xls_modules_zstd_comp_lookup_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanWeightsDecoder_0__HuffmanFseWeightsDecoder_0__CompLookupDecoder_0__64_8_16_1_15_8_32_1_7_9_8_1_8_16_1_next_inst5,
    "zstd_dec__fse_table_finish__1",
    dut.clk,
  )
  fse_lookup_resp_handshake = triggers.Event()

  block_cnt = 0

  def func():
    nonlocal block_cnt
    assert block_cnt <= len(expected_fse_lookups)
    print_fse_ram_contents(
      dut.huffman_literals_weights_fse_ram_ram.mem,
      f"HUFMMAN ({block_cnt})",
      size=len(expected_fse_lookups[block_cnt]),
    )
    check_ram_contents(
      dut.huffman_literals_weights_fse_ram_ram.mem,
      [x.value for x in expected_fse_lookups[block_cnt]],
    )
    block_cnt += 1

  cocotb.start_soon(
    set_handshake_event(dut.clk, lookup_dec_resp_channel, fse_lookup_resp_handshake)
  )
  cocotb.start_soon(get_handshake_event(dut, fse_lookup_resp_handshake, func))


async def test_huffman_weights(dut, clock, expected_huffman_weights):
  lookup_dec_resp_channel = xlschannel.XLSChannel(
    dut.ZstdDecoder.xls_modules_zstd_huffman_ctrl__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanControlAndSequence_0__32_64_next_inst20,
    "zstd_dec__weights_dec_resp",
    dut.clk,
  )
  huffman_weights_resp_handshake = triggers.Event()

  block_cnt = 0

  def func():
    nonlocal block_cnt
    assert block_cnt <= len(expected_huffman_weights)
    print_ram_contents(
      dut.huffman_literals_weights_mem_ram_ram.mem,
      f"WEIGHTS ({block_cnt})",
      size=64,
    )
    check_ram_contents(
      dut.huffman_literals_weights_mem_ram_ram.mem,
      expected_huffman_weights[block_cnt],
    )
    block_cnt += 1

  cocotb.start_soon(
    set_handshake_event(
      dut.clk, lookup_dec_resp_channel, huffman_weights_resp_handshake
    )
  )
  cocotb.start_soon(get_handshake_event(dut, huffman_weights_resp_handshake, func))


async def test_decoder(
  dut, seed, block_type, literal_type, axi_buses, cpu, clock, pregenerated
):
  """Test decoder with zstd-compressed data

  if a file name is provided in `pregenerated`, use it as input
  otherwise generate a random file using (seed, block_type, literal_type)

  The output of the decoder is compared with the output of decodercorpus
  using the same input file.
  """
  memory_bus = axi_buses["memory"]

  (unused_notify_channel, notify_monitor) = connect_xls_channel(
    dut, NOTIFY_CHANNEL, NotifyStruct
  )
  assert_notify = triggers.Event()
  set_termination_event(notify_monitor, assert_notify, 1)

  mem_size = MAX_ENCODED_FRAME_SIZE_B
  ibuf_addr = 0x0
  obuf_addr = mem_size // 2

  # TODO if pregenerated is used,
  # block_type, literal_type and seed aren't used. Handle this better.
  if pregenerated:
    encoded = open(pregenerated, "rb")
  else:
    # FIXME: use delete_on_close=False after moving to python 3.12
    encoded = tempfile.NamedTemporaryFile(delete=False)
    # Generate ZSTD frame to temporary file
    data_generator.GenerateFrame(seed, block_type, encoded.name, literal_type)

  print(
    "\nusing"
    + (" pregenerated" if pregenerated else f" randomly generated (seed={seed})")
    + f" input file for decoder: {encoded.name}\n"
  )

  await reset_dut(dut, 50)

  expected_decoded_frame = data_generator.DecompressFrame(encoded.read())
  encoded.close()
  reference_memory = SparseMemory(mem_size)
  reference_memory.write(obuf_addr, expected_decoded_frame)

  # Initialise testbench memory with generated ZSTD frame
  memory = AxiRamFromFile(
    bus=memory_bus, clock=dut.clk, reset=dut.rst, path=encoded.name, size=mem_size
  )

  await configure_decoder(dut, cpu, ibuf_addr, obuf_addr)
  output_monitor = AxiWMonitor(memory_bus.write.w, dut.clk, dut.rst)
  await start_decoder(cpu)
  decode_start = get_clock_time(clock)
  await output_monitor.wait()
  decode_first_packet = get_clock_time(clock)
  await assert_notify.wait()
  decode_end = get_clock_time(clock)
  await wait_for_idle(cpu)
  # Read decoded frame in chunks of AXI_DATA_W length
  # Compare against frame decompressed with the reference library
  expected_packet_count = (
    len(expected_decoded_frame) + (AXI_DATA_W_BYTES - 1)
  ) // AXI_DATA_W_BYTES
  for read_op in range(0, expected_packet_count):
    addr = obuf_addr + (read_op * AXI_DATA_W_BYTES)
    mem_contents = memory.read(addr, AXI_DATA_W_BYTES)
    exp_mem_contents = reference_memory.read(addr, AXI_DATA_W_BYTES)
    assert mem_contents == exp_mem_contents, (
      "{} bytes of memory contents at address {} "
      "don't match the expected contents:\n"
      "{}\nvs\n{}"
    ).format(
      AXI_DATA_W_BYTES,
      hex(addr),
      hex(int.from_bytes(mem_contents, byteorder="little")),
      hex(int.from_bytes(exp_mem_contents, byteorder="little")),
    )
  await ClockCycles(dut.clk, 20)


async def testing_routine(
  dut,
  test_cases=1,
  block_type=data_generator.BlockType.RANDOM,
  literal_type=data_generator.LiteralType.RANDOM,
  pregenerated=None,
  expected_fse_lookups=None,
  expected_fse_huffman_lookups=None,
  expected_huffman_weights=None,
):
  (axi_buses, cpu, clock) = prepare_test_environment(dut)
  frame_id = 0
  for test_case in range(test_cases):
    if expected_fse_lookups is not None:
      await test_fse_lookup_decoder(dut, clock, expected_fse_lookups)
    if expected_fse_huffman_lookups is not None:
      await test_fse_lookup_decoder_for_huffman(
        dut, clock, expected_fse_huffman_lookups
      )
    if expected_huffman_weights is not None:
      await test_huffman_weights(dut, clock, expected_huffman_weights)
    await test_decoder(
      dut, 2, block_type, literal_type, axi_buses, cpu, clock, pregenerated
    )
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


# Tests with pregenerated inputs
#
# block type and literal type in arguments and file names reflect what was used
# for generating them. The file names also contain the value of seed but the
# bazel environment is not hermetic in terms of reproducible results with the
# same seed value
# The tests are disabled by default as none of them passes currently.
# Use them to verify progress in specific parts of the decoder.

# TODO the workdir / data relation is weird. How to pass this better?
PREGENERATED_FILES_DIR = "../../xls/modules/zstd/data/"


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_raw_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_raw_1.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RAW
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_raw_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_raw_2.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RAW
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_rle_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_rle_1.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RLE
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def pregenerated_compressed_rle_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_rle_2.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RLE
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def pregenerated_compressed_random_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_random_1.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x01000100,
      0x06000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000010,
      0x00100030,
      0x00700000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000001,
      0x00010003,
      0x00080000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x10001000,
      0x50000000,
    ],
    [
      0x02000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000010,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00003000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00010000,
    ],
  ]

  expected_fse_huffman_lookups = [
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0016),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0018),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0001),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0002),
      FseTableRecord(symbol=0x08, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0003),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0004),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0005),
      FseTableRecord(symbol=0x01, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0006),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0007),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0008),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0009),
      FseTableRecord(symbol=0x07, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000A),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000B),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000C),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000D),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000F),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0010),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0011),
      FseTableRecord(symbol=0x06, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0012),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0013),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0014),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0015),
    ],
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0001),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0002),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0003),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0004),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0005),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0006),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0007),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0008),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0009),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000A),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000B),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000C),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000D),
      FseTableRecord(symbol=0x02, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000F),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0010),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0011),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0012),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0013),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0014),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0015),
      FseTableRecord(symbol=0x01, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0016),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0017),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0018),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0019),
    ],
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x01, num_of_bits=0x04, base=0x0000),
      FseTableRecord(symbol=0x02, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x05, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x06, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x08, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x0A, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x0D, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x10, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x13, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x16, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x19, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1C, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1F, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x21, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x23, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x25, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x27, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x29, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2B, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2D, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x01, num_of_bits=0x04, base=0x0010),
      FseTableRecord(symbol=0x02, num_of_bits=0x04, base=0x0000),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x04, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x06, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x07, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x09, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x0C, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x0F, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x12, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x15, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x18, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1B, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1E, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x20, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x22, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x24, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x26, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x28, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2A, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2C, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x01, num_of_bits=0x04, base=0x0020),
      FseTableRecord(symbol=0x01, num_of_bits=0x04, base=0x0030),
      FseTableRecord(symbol=0x02, num_of_bits=0x04, base=0x0010),
      FseTableRecord(symbol=0x04, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x05, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x07, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x08, num_of_bits=0x05, base=0x0020),
      FseTableRecord(symbol=0x0B, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x0E, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x11, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x14, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x17, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1A, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x1D, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x34, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x33, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x32, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x31, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x30, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2F, num_of_bits=0x06, base=0x0000),
      FseTableRecord(symbol=0x2E, num_of_bits=0x06, base=0x0000),
    ],
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_random_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_random_2.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


# Tests with predefined FSE tables and Huffman-encoded literals


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_107958(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_107958.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x10000000,
      0x00004000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00010000,
      0x00000005,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000020,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x03000000,
    ]
  ]

  expected_fse_huffman_lookups = [
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0018),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
      FseTableRecord(symbol=0x01, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0001),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0002),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0003),
      FseTableRecord(symbol=0x05, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0004),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0005),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0006),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0007),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0008),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0009),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000A),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000B),
      FseTableRecord(symbol=0x04, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000C),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000D),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000F),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0010),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0011),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0012),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0013),
      FseTableRecord(symbol=0x02, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0014),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0015),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0016),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0017),
    ]
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_204626(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_204626.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_210872(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_210872.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_299289(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_299289.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_319146(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_319146.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_331938(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_331938.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_333824(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_333824.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


# Test cases crated manually to allow working with small sizes of inputs.


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def pregenerated_compressed_minimal(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_minimal.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def pregenerated_uncompressed(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_uncompressed.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


# Test cases with predefined FSE tables and RAW/RLE literals


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_406229(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_406229.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_411034(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_411034.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_413015(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_413015.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_436165(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_436165.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_464057(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_464057.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_466803(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_466803.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_422473(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_422473.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_436965(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_436965.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_462302(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_462302.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_raw_literals_predefined_sequences_seed_408158(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_raw_literals_predefined_sequences_seed_408158.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_raw_literals_predefined_sequences_seed_499212(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_raw_literals_predefined_sequences_seed_499212.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


# Tests with inputs that correspond to the values in arrays defined in
# data/*.x files


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def comp_frame_fse_comp(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_fse_comp.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_fse_lookups = [
    {
      "ll": [
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0014),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0014),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0016),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0018),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0020),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0022),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0024),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0026),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0028),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0016),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0018),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x001A),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x001C),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x001E),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0020),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0022),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0024),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0026),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0028),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x002A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x002A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x002C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x002E),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0030),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0032),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0034),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0036),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0038),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x003A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x003C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x003E),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x002C),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x002E),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0030),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0032),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0034),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0036),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x0038),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x003A),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x003C),
        FseTableRecord(symbol=0x06, num_of_bits=0x01, base=0x003E),
      ],
      "of": [
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0014),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0014),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x01, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x02, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x05, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0014),
        FseTableRecord(symbol=0x01, num_of_bits=0x01, base=0x0016),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x02, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x05, num_of_bits=0x01, base=0x0012),
      ],
      "ml": [
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0014),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0014),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0018),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x001C),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0020),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0024),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0028),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x002C),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x00, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x15, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0030),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0034),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x0038),
        FseTableRecord(symbol=0x1C, num_of_bits=0x02, base=0x003C),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0000),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0002),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0004),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0014),
        FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0016),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x15, num_of_bits=0x01, base=0x0012),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0006),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0008),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x000A),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x000C),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x000E),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0010),
        FseTableRecord(symbol=0x1C, num_of_bits=0x01, base=0x0012),
      ],
    }
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_fse_lookups=expected_fse_lookups,
  )


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_fse_repeated(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_fse_repeated.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_huffman(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_huffman.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_huffman_fse(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_huffman_fse.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def raw_literals_compressed_sequences_seed_903062(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_compressed_sequences_seed_903062.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_rle_sequences_seed_700216(dut):
  input_name = PREGENERATED_FILES_DIR + "raw_literals_rle_sequences_seed_700216.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def rle_literals_compressed_sequences_seed_701326(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_compressed_sequences_seed_701326.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_rle_sequences_seed_2(dut):
  input_name = PREGENERATED_FILES_DIR + "rle_literals_rle_sequences_seed_2.zst"
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM
  await testing_routine(dut, test_cases, block_type, literal_type, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_compressed_sequences_seed_400077(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_compressed_sequences_seed_400077.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x10000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000200,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00300000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000004,
      0x00000000,
      0x00010000,
    ]
  ]

  expected_fse_huffman_lookups = [
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0018),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
      FseTableRecord(symbol=0x01, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0001),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0002),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0003),
      FseTableRecord(symbol=0x04, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0004),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0005),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0006),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0007),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0008),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0009),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000A),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000B),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000C),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000D),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000F),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0010),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0011),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0012),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0013),
      FseTableRecord(symbol=0x02, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0014),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0015),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0016),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0017),
    ]
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_predefined_rle_compressed_sequences_seed_400025(
  dut,
):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_predefined_rle_compressed_sequences_seed_400025.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x11111111,
      0x11111111,
    ]
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_predefined_rle_compressed_sequences_seed_400061(
  dut,
):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_predefined_rle_compressed_sequences_seed_400061.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x11111111,
      0x11111111,
    ]
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_rle_sequences_seed_403927(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_rle_sequences_seed_403927.zst"
  )
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RANDOM

  expected_huffman_weights = [
    [
      0x00003001,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x02001000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000020,
      0x01000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x80010010,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000700,
      0x10000000,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00600100,
      0x00000000,
      0x00000000,
      0x00000000,
      0x00000005,
      0x00100000,
    ]
  ]

  expected_fse_huffman_lookups = [
    [
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0012),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0014),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0016),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x0018),
      FseTableRecord(symbol=0x05, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001A),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001C),
      FseTableRecord(symbol=0x00, num_of_bits=0x01, base=0x001E),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0000),
      FseTableRecord(symbol=0x08, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0001),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0002),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0003),
      FseTableRecord(symbol=0x03, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0004),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0005),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0006),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0007),
      FseTableRecord(symbol=0x07, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0008),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0009),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000A),
      FseTableRecord(symbol=0x02, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000B),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000C),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000D),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000E),
      FseTableRecord(symbol=0x06, num_of_bits=0x05, base=0x0000),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x000F),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0010),
      FseTableRecord(symbol=0x00, num_of_bits=0x00, base=0x0011),
      FseTableRecord(symbol=0x01, num_of_bits=0x05, base=0x0000),
    ]
  ]

  await testing_routine(
    dut,
    test_cases,
    block_type,
    literal_type,
    input_name,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


# Tests with inputs generated randomly on test execution


@cocotb.test(timeout_time=5000, timeout_unit="ms")
async def zstd_compressed_frames_test(dut):
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RAW
  await testing_routine(dut, test_cases, block_type, literal_type)


# @cocotb.test(timeout_time=1000, timeout_unit="ms")
# async def zstd_random_frames_test(dut):
# test_cases = 1
# block_type = BlockType.RANDOM
# await testing_routine(dut, test_cases, block_type)

if __name__ == "__main__":
  with tempfile.NamedTemporaryFile(mode="w") as modified_zstd_verilog:
    toplevel = "zstd_dec_wrapper"
    test_module = [pathlib.Path(__file__).stem]
    verilog_sources = [
      modified_zstd_verilog.name,
      "xls/modules/zstd/rtl/xls_fifo_wrapper.sv",
      "xls/modules/zstd/rtl/zstd_dec_wrapper.sv",
      "xls/modules/zstd/axi_crossbar_wrapper.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_crossbar.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_crossbar_rd.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_crossbar_wr.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_crossbar_addr.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_register_rd.v",
      "external/com_github_alexforencich_verilog_axi/rtl/axi_register_wr.v",
      "external/com_github_alexforencich_verilog_axi/rtl/arbiter.v",
      "external/com_github_alexforencich_verilog_axi/rtl/priority_encoder.v",
      "xls/modules/zstd/rtl/ram_1r1w.v",
    ]

    with open("xls/modules/zstd/zstd_dec.v") as zstd_verilog:
      modified_content = zstd_verilog.read().replace(
        "__xls_modules_zstd", "xls_modules_zstd"
      )
      modified_zstd_verilog.write(modified_content)

    modified_zstd_verilog.flush() #
    run_test(toplevel, test_module, verilog_sources)
