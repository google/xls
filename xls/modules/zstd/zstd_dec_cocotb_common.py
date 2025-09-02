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


import math
import enum
import tempfile
import sys

import cocotb
from cocotb import triggers
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Event, Edge
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
import xls.modules.zstd.cocotb.utils as cocotb_utils
from xls.modules.zstd.cocotb import data_generator
from xls.modules.zstd.cocotb.memory import AxiRamFromFile
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


PARALLEL_ACCESS_WIDTH = 8
MAX_WEIGHT = 11
WEIGHT_LOG = math.ceil(math.log2(MAX_WEIGHT + 1))
VALID_W = 1


@xlsstruct.xls_dataclass
class CodeBuilderOutput(xlsstruct.XLSStruct):
  symbol_valid_7: VALID_W
  symbol_valid_6: VALID_W
  symbol_valid_5: VALID_W
  symbol_valid_4: VALID_W
  symbol_valid_3: VALID_W
  symbol_valid_2: VALID_W
  symbol_valid_1: VALID_W
  symbol_valid_0: VALID_W
  code_length_7: WEIGHT_LOG
  code_length_6: WEIGHT_LOG
  code_length_5: WEIGHT_LOG
  code_length_4: WEIGHT_LOG
  code_length_3: WEIGHT_LOG
  code_length_2: WEIGHT_LOG
  code_length_1: WEIGHT_LOG
  code_length_0: WEIGHT_LOG
  code_7: MAX_WEIGHT
  code_6: MAX_WEIGHT
  code_5: MAX_WEIGHT
  code_4: MAX_WEIGHT
  code_3: MAX_WEIGHT
  code_2: MAX_WEIGHT
  code_1: MAX_WEIGHT
  code_0: MAX_WEIGHT


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


def check_decoder_compliance(file_path):
  with open(file_path, mode='rb') as compressed_file:
    # Unused magic number
    compressed_file.seek(4)

    frame_header_descriptor = int.from_bytes(compressed_file.read(1))

    # 6th bit of a frame header indicates if a window descriptor is present
    SINGLE_SEGMENT_FLAG_MASK = 0b100000
    if frame_header_descriptor & SINGLE_SEGMENT_FLAG_MASK == 0:
      # Calculate a required window_size for the encoded file
      # and compare it with the ZSTD decoder history buffer size.
      # Based on window_size calculation from: RFC 8878
      # https://datatracker.ietf.org/doc/html/rfc8878#name-window-descriptor
      window_descriptor = int.from_bytes(compressed_file.read(1))

      MANTISSA_BITS = 0b111
      mantissa = window_descriptor & MANTISSA_BITS

      EXPONENT_BITS = 0b11111000
      exponent = (window_descriptor & EXPONENT_BITS) >> 3

      window_log = 10 + exponent
      window_base = 1 << window_log
      window_add = (window_base / 8) * mantissa
      window_size = int(window_base + window_add)

      # This value represent the size of the history buffer for the tested
      # instance of the ZSTD decoder, and should be in line with the value
      # of `INST_HB_SIZE_KB` declared in the zstd_dec.x file.
      HISTORY_BUFFER_SIZE = 64 * 1024 # 64 KB

      if window_size > HISTORY_BUFFER_SIZE:
        print(f"error: required window size greater than the actual history buffer: {window_size} > {HISTORY_BUFFER_SIZE}")
        return False

  return True


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


def fields_as_array(data, prefix, count):
  return [getattr(data, f"{prefix}_{i}") for i in range(count)]


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
    dut.ZstdDecoder.xls_modules_zstd_sequence_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__SequenceDecoderCtrl_0__FseLookupCtrl_0_next_inst149,
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


@cocotb.coroutine
async def await_state_cycle(clk, wire, func, startvals, endvals):
  """Monitors a state signal and reports elapsed cycles between start and end states.

  This will continously observe a given state signal and after any of startvals,
  and then endvals is matched with the value, report the elapsed cycles.

  For procs that have an FSM, use proper start state (the one used after IDLE) in startvals,
  and a state that corresponds to sending the result back, or switching back to IDLE in endvals.

  Args:
    wire: state signal to observe
    func (Callable[[int], None]): Callback function to report the elapsed cycles
    startvals (Iterable): values that define the start states
    endvals (Iterable): values that define the end states
  """
  while True:
    while wire.value not in startvals:
      await Edge(wire)
    start = get_clock_time(clk)
    while wire.value not in endvals:
      await Edge(wire)
    end = get_clock_time(clk)
    func(end - start)

async def report_fse_decoder_work(dut, clk):
  fse_state = dut.ZstdDecoder.xls_modules_zstd_fse_dec__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__FseDecoder_0__64_15_32_1_64_7_next_inst10.p2_____state_0__1
  start_states = [1]
  end_states = [16]

  def report(value):
    print(f'FSE Decoder finished after {value} cycles')

  cocotb.start_soon(await_state_cycle(clk, fse_state, report, start_states, end_states))

async def report_sequence_executor_work(dut, clk):
  state = dut.ZstdDecoder.xls_modules_zstd_sequence_executor__ZstdDecoderInst__ZstdDecoder_0__SequenceExecutor_0__32_64_64_0_0_0_13_8192_65536_next_inst150.p2_____state_0__1
  start_states = [1, 2]
  end_states = [0]

  def report(value):
    print(f'Sequence executor finished after {value} cycles')

  cocotb.start_soon(await_state_cycle(clk, state, report, start_states, end_states))

async def report_fse_table_creator_work(dut, clk):
  state = dut.ZstdDecoder.xls_modules_zstd_fse_table_creator__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__SequenceDecoder_0__FseLookupDecoder_0__CompLookupDecoder_0__FseTableCreator_0__8_16_1_15_32_1_9_8_1_8_16_1_next_inst16.p3_____state_0__1
  start_states = [1]
  end_states = [11]

  def report(value):
    print(f'FSE table creator finished after {value} cycles')

  cocotb.start_soon(await_state_cycle(clk, state, report, start_states, end_states))

def reverse_expected_huffman_codes(exp_codes):
  def reverse_bits(value, max_bits):
    bv = BinaryValue(value=value, n_bits=max_bits, bigEndian=False)
    return int(BinaryValue(value=bv.binstr[::-1], n_bits=max_bits, bigEndian=False))

  max_bits = max(d["length"] for d in exp_codes)

  codes = []
  for record in exp_codes:
    codes += [
      {
        "code": reverse_bits(record["code"], max_bits),
        "length": record["length"],
        "symbol": record["symbol"],
      }
    ]
  return codes


async def test_huffman_codes(dut, clock, expected_codes):
  WEIGHT_CODE_BUILDER_INST = (
    dut.ZstdDecoder.xls_modules_zstd_huffman_code_builder__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__WeightCodeBuilder_0_next_inst20
  )
  CODES_CHANNEL_NAME = "zstd_dec__code_builder_codes"

  codes_channel = xlschannel.XLSChannel(
    WEIGHT_CODE_BUILDER_INST, CODES_CHANNEL_NAME, dut.clk
  )
  huffman_code_handshake = triggers.Event()

  codes = []
  block_cnt = 0
  packet_cnt = 0
  symbol_cnt = 0

  def func():
    nonlocal codes
    nonlocal symbol_cnt
    nonlocal packet_cnt
    nonlocal block_cnt

    assert block_cnt <= 32
    codes_data = getattr(WEIGHT_CODE_BUILDER_INST, CODES_CHANNEL_NAME)
    data = CodeBuilderOutput.from_int(codes_data.value)

    symbol_valid_array = fields_as_array(data, "symbol_valid", 8)
    code_length_array = fields_as_array(data, "code_length", 8)
    code_array = fields_as_array(data, "code", 8)

    for symbol_valid, code_length, code in zip(
      symbol_valid_array, code_length_array, code_array
    ):
      if symbol_valid == 1:
        codes += [{"symbol": symbol_cnt, "code": code, "length": code_length}]
      symbol_cnt += 1
    packet_cnt += 1

    if packet_cnt == 32:
      assert codes == reverse_expected_huffman_codes(expected_codes[block_cnt])
      packet_cnt = 0
      symbol_cnt = 0
      block_cnt += 1
      codes = []

  cocotb.start_soon(
    set_handshake_event(dut.clk, codes_channel, huffman_code_handshake)
  )
  cocotb.start_soon(get_handshake_event(dut, huffman_code_handshake, func))


async def test_huffman_weights(dut, clock, expected_huffman_weights):
  lookup_dec_resp_channel = xlschannel.XLSChannel(
    dut.ZstdDecoder.xls_modules_zstd_huffman_ctrl__ZstdDecoderInst__ZstdDecoder_0__CompressBlockDecoder_0__LiteralsDecoder_0__HuffmanLiteralsDecoder_0__HuffmanControlAndSequence_0__32_64_next_inst21,
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

async def check_output(expected_packet_count, memory, reference_memory, output_monitor, obuf_addr, clock):
  # Read decoded frame in chunks of AXI_DATA_W length
  # Compare against frame decompressed with the reference library
  current_addr = obuf_addr
  decode_start = get_clock_time(clock)
  await output_monitor.wait()
  decode_first_packet = get_clock_time(clock)

  for read_op in range(0, expected_packet_count):
    current_addr = obuf_addr + (read_op * AXI_DATA_W_BYTES)
    exp_mem_contents = int.from_bytes(reference_memory.read(current_addr, AXI_DATA_W_BYTES), byteorder="little")
    mem_contents = (await output_monitor.recv()).wdata
    assert mem_contents == exp_mem_contents, (
      "{} bytes of memory contents at address {} "
      "don't match the expected contents:\n"
      "{}\nvs\n{}"
    ).format(
      AXI_DATA_W_BYTES,
      hex(current_addr),
      hex(mem_contents),
      hex(exp_mem_contents),
    )
    print(f'[cocotb] Got correct packet (addr: {hex(current_addr)}, data: {hex(mem_contents)}, clk: {get_clock_time(clock)})', file=sys.stderr)

  decode_last_packet = get_clock_time(clock)
  return (decode_start, decode_first_packet, decode_last_packet)

async def test_decoder(dut, axi_buses, cpu, clock, encoded_file):
  """Test decoder with zstd-compressed data provided in `encoded_file`

  The output of the decoder is compared with the output of decodercorpus
  using the same input file.
  """
  assert check_decoder_compliance(encoded_file.name), (f"error: '{encoded_file.name}' is not suitable for the decoder parameters")

  memory_bus = axi_buses["memory"]

  (unused_notify_channel, notify_monitor) = connect_xls_channel(
    dut, NOTIFY_CHANNEL, NotifyStruct
  )
  assert_notify = triggers.Event()
  set_termination_event(notify_monitor, assert_notify, 1)

  mem_size = MAX_ENCODED_FRAME_SIZE_B
  ibuf_addr = 0x0
  obuf_addr = mem_size // 2

  await reset_dut(dut, 50)

  expected_decoded_frame = data_generator.DecompressFrame(encoded_file.read())
  reference_memory = SparseMemory(mem_size)
  reference_memory.write(obuf_addr, expected_decoded_frame)
  expected_packet_count = (
    len(expected_decoded_frame) + (AXI_DATA_W_BYTES - 1)
  ) // AXI_DATA_W_BYTES

  # Initialise testbench memory with generated ZSTD frame
  memory = AxiRamFromFile(
    bus=memory_bus, clock=dut.clk, reset=dut.rst, path=encoded_file.name, size=mem_size
  )

  await configure_decoder(dut, cpu, ibuf_addr, obuf_addr)
  output_monitor = AxiWMonitor(memory_bus.write.w, dut.clk, dut.rst)
  await start_decoder(cpu)

  decode_times = await check_output(expected_packet_count, memory, reference_memory, output_monitor, obuf_addr, clock)
  (decode_start, decode_first_packet, decode_last_packet) = decode_times
  await assert_notify.wait()
  await wait_for_idle(cpu)
  decode_end = get_clock_time(clock)

  latency = decode_first_packet - decode_start
  throughput_repacketizer = expected_packet_count / (decode_end - decode_first_packet)
  throughput_bytes = throughput_repacketizer * AXI_DATA_W_BYTES
  print(f"Decoding latency: {latency} cycles")
  print(
    f"Decoding throughput: {throughput_bytes}B/cycle ({throughput_repacketizer} packets/cycle)"
  )

  await ClockCycles(dut.clk, 20)

  return (latency, throughput_bytes, throughput_repacketizer)


async def randomized_testing_routine(
  dut,
  test_cases=1,
  block_type=data_generator.BlockType.RANDOM,
  literal_type=data_generator.LiteralType.RANDOM,
  expected_fse_lookups=None,
  expected_fse_huffman_lookups=None,
  expected_huffman_weights=None,
  expected_huffman_codes=None,
):
  (axi_buses, cpu, clock) = prepare_test_environment(dut)
  measurements = []
  frame_id = 0
  seed = 2 # FIXME: Dehardcode
  for test_case in range(test_cases):
    if expected_fse_lookups is not None:
      await test_fse_lookup_decoder(dut, clock, expected_fse_lookups)
    if expected_fse_huffman_lookups is not None:
      await test_fse_lookup_decoder_for_huffman(
        dut, clock, expected_fse_huffman_lookups
      )
    if expected_huffman_codes is not None:
      await test_huffman_codes(dut, clock, expected_huffman_codes)
    if expected_huffman_weights is not None:
      await test_huffman_weights(dut, clock, expected_huffman_weights)

    # FIXME: use delete_on_close=False after moving to python 3.12
    with tempfile.NamedTemporaryFile(delete=False) as input_file:
      # Generate ZSTD frame to temporary file
      data_generator.GenerateFrame(seed, block_type, input_file.name, literal_type)
      print(
        f"\nusing randomly generated (seed={seed}) input file for decoder: {input_file.name}\n"
      )
      measurements.append(await test_decoder(dut, axi_buses, cpu, clock, input_file))

  print("Decoding {} ZSTD frames done".format(block_type.name))
  for measurement in measurements:
    print(
      f"Frame #{frame_id}: latency: {measurement[0]} cycles; throughput: {measurement[1]} B/cycle ({measurement[2]} packets/cycle)"
    )
    frame_id += 1


async def pregenerated_testing_routine(
  dut,
  pregenerated_path,
  expected_fse_lookups=None,
  expected_fse_huffman_lookups=None,
  expected_huffman_weights=None,
  expected_huffman_codes=None,
):
  (axi_buses, cpu, clock) = prepare_test_environment(dut)
  print(
    f"\nusing pregenerated input file for decoder: {pregenerated_path}\n"
  )
  await report_fse_decoder_work(dut, clock)
  await report_sequence_executor_work(dut, clock)
  await report_fse_table_creator_work(dut, clock)

  if expected_fse_lookups is not None:
    await test_fse_lookup_decoder(dut, clock, expected_fse_lookups)
  if expected_fse_huffman_lookups is not None:
    await test_fse_lookup_decoder_for_huffman(
      dut, clock, expected_fse_huffman_lookups
    )
  if expected_huffman_codes is not None:
    await test_huffman_codes(dut, clock, expected_huffman_codes)
  if expected_huffman_weights is not None:
    await test_huffman_weights(dut, clock, expected_huffman_weights)

  with open(pregenerated_path, 'rb') as input_file:
    measurement = await test_decoder(dut, axi_buses, cpu, clock, input_file)

  print(
    f"Frame #0: latency: {measurement[0]} cycles; throughput: {measurement[1]} B/cycle ({measurement[2]} packets/cycle)"
  )


def run_test(test_module, build_args=[], sim="icarus"):
  toplevel = "zstd_dec_wrapper"
  verilog_sources = [
    "xls/modules/zstd/zstd_dec.v",
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

  cocotb_utils.run_test(toplevel, test_module, verilog_sources, build_args=build_args, sim=sim)
