# Copyright 2023 The XLS Authors
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

import os
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import cocotb
from cocotb.binary import BinaryValue
from cocotb.clock import Clock
from cocotb.handle import ModifiableObject, SimHandleBase
from cocotb.runner import check_results_file, get_runner
from cocotb.triggers import ClockCycles, Edge, Event, First, Join, RisingEdge
from cocotb_bus.scoreboard import Scoreboard
from cocotb_xls import XLSChannel, XLSChannelDriver, XLSChannelMonitor

from xls.common import runfiles, test_base
from xls.simulation.cocotb.cocotb_struct import Struct
from xls.tools.dashboard.utils import get_dashboard_json_dump_str

RECV_CHANNEL = "rle_enc__input_r"
SEND_CHANNEL = "rle_enc__output_s"

SYMBOL_WIDTH = 32
COUNT_WIDTH = 2

DashboardJSONs = Sequence[Dict]


class EncInData(Struct):
  _STRUCT_FIELDS_ = [
    ("symbol", SYMBOL_WIDTH, int),
    ("last", 1, bool),
  ]


class EncOutData(Struct):
  _STRUCT_FIELDS_ = [
    ("symbol", SYMBOL_WIDTH, int),
    ("count", COUNT_WIDTH, int),
    ("last", 1, bool),
  ]


def generate_input_data(input_symbols: Sequence[int]) -> Sequence[BinaryValue]:
  """Converts the input symbols to a list of BinaryValues that can be used
  as input data for the simulation
  """
  values = []
  for i, symbol in enumerate(input_symbols):
    last = 1 if i == len(input_symbols) - 1 else 0
    values += [EncInData.from_fields(symbol=symbol, last=last).as_binary()]
  return values


def generate_expected_output_data(
  input_symbols: Sequence[int],
) -> Sequence[BinaryValue]:
  """Given the input symbols, produces an expected output from the RLE encoder"""

  MAX_COUNT = 2**COUNT_WIDTH - 1

  prev_symbol = input_symbols[0]
  prev_count = 1
  prev_last = 0 if len(input_symbols) > 1 else 1

  output = []
  for i, symbol in enumerate(input_symbols[1:]):
    if symbol != prev_symbol:
      output.append(
        EncOutData.from_fields(
          symbol=prev_symbol, count=prev_count, last=prev_last
        ).as_binary()
      )
      prev_symbol = symbol
      prev_count = 1
    else:
      if prev_count == MAX_COUNT:
        output.append(
          EncOutData.from_fields(
            symbol=prev_symbol, count=prev_count, last=prev_last
          ).as_binary()
        )
        prev_count = 1
        prev_symbol = prev_symbol
      else:
        prev_count += 1
        prev_symbol = prev_symbol

    if i == len(input_symbols) - 2:
      prev_last = 1

  output.append(
    EncOutData.from_fields(symbol=prev_symbol, count=prev_count, last=1).as_binary()
  )

  return output


def create_dashboard_json_with_measurements(delay, performance) -> DashboardJSONs:
  """Creates Dashboard JSON from delay and performance measurements"""
  return [
    {
      "name": "Measurements",
      "group": "performance",
      "type": "table",
      "value": [
        ["Property", "Value"],
        ["Delay [cycles]", delay],
        ["Performance [symbols/cycle]", performance],
      ],
    }
  ]


@dataclass
class SimulationData:
  """Auxiliary structure used to store design-related data for the simulation"""

  driver: XLSChannelDriver
  monitor: XLSChannelMonitor
  scoreboard: Scoreboard
  input_r: XLSChannel
  output_s: XLSChannel
  clk: ModifiableObject
  rst: ModifiableObject
  terminate: Event
  send_event: Event
  recv_event: Event
  recv_stop_event: Event


def init_sim(dut: SimHandleBase, data_to_recv: Sequence[BinaryValue]) -> SimulationData:
  """Extracts all design-related data required for simulation"""

  send_event = Event("Data send")
  recv_event = Event("Data received")
  recv_stop_event = Event("Data received")

  driver = XLSChannelDriver(dut, RECV_CHANNEL, dut.clk)
  monitor = XLSChannelMonitor(dut, SEND_CHANNEL, dut.clk, event=recv_event)
  input_r = XLSChannel(dut, RECV_CHANNEL)
  output_s = XLSChannel(dut, SEND_CHANNEL)

  scoreboard = Scoreboard(dut, fail_immediately=True)
  scoreboard.add_interface(monitor, deepcopy(data_to_recv))

  expected_packet_count = len(data_to_recv)
  terminate = Event("Received the last packet of data")

  def terminate_cb(_):
    if monitor.stats.received_transactions == expected_packet_count:
      terminate.set()

  monitor.add_callback(terminate_cb)

  return SimulationData(
    driver=driver,
    monitor=monitor,
    scoreboard=scoreboard,
    input_r=input_r,
    output_s=output_s,
    terminate=terminate,
    clk=dut.clk,
    rst=dut.rst,
    send_event=send_event,
    recv_event=recv_event,
    recv_stop_event=recv_stop_event,
  )


@cocotb.coroutine
async def recv(clk, send_channel, recv_stop_event):
  """Cocotb coroutine that acts as a proc receiving data from a channel"""
  while True:
    send_channel.rdy.value = send_channel.vld.value
    await First(Edge(send_channel.vld), RisingEdge(clk))
    if recv_stop_event.is_set():
      break


@cocotb.coroutine
async def recv_stop(recv_task, recv_stop_event):
  """Cocotb coroutine used to stop receiving data in the recv task"""
  recv_stop_event.set()
  await Join(recv_task)


@cocotb.coroutine
async def reset(clk, rst, cycles=1):
  """Cocotb coroutine that performs a reset"""
  rst.setimmediatevalue(1)
  await ClockCycles(clk, cycles)
  rst.value = 0


@cocotb.coroutine
async def verify_output(dut, input_symbols):
  """Auxiliary coroutine that can check whether the encoder produced
  expected data out of provided input symbols
  """
  enc_in_data = generate_input_data(input_symbols)
  enc_out_data = generate_expected_output_data(input_symbols)

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())
  await reset(dut.clk, dut.rst)

  sim = init_sim(dut, enc_out_data)

  recv_coroutine = recv(dut.clk, sim.output_s, sim.recv_stop_event)
  recv_task = cocotb.start_soon(recv_coroutine)

  await sim.driver.send(enc_in_data)
  await sim.terminate.wait()
  await recv_stop(recv_task, sim.recv_stop_event)
  sim.monitor.kill()


@cocotb.coroutine
async def measure_time(clk, start, stop):
  """Cocotb coroutine that measures time between the start and stop events"""
  cnt = 0
  while True:
    await start.wait()
    if stop.is_set():
      break
    await RisingEdge(clk)
    cnt += 1

  return cnt


@cocotb.coroutine
async def measure_delay(dut):
  """Cocotb coroutine for measuring delay of the RLE encoder"""

  input_symbols = [0xA]
  enc_in_data = generate_input_data(input_symbols)
  enc_out_data = generate_expected_output_data(input_symbols)

  await reset(dut.clk, dut.rst)
  sim = init_sim(dut, enc_out_data)

  measure_coroutine = measure_time(sim.clk, sim.send_event, sim.recv_event)
  measure_task = cocotb.start_soon(measure_coroutine)

  recv_coroutine = recv(dut.clk, sim.output_s, sim.recv_stop_event)
  recv_task = cocotb.start_soon(recv_coroutine)

  await sim.driver._send(enc_in_data, None, sim.send_event)
  await sim.terminate.wait()
  await recv_stop(recv_task, sim.recv_stop_event)
  delay = await Join(measure_task)
  sim.monitor.kill()

  return delay


@cocotb.coroutine
async def measure_performance(dut, delay=0):
  """Cocotb coroutine for measuring performance of the RLE encoder"""

  number_of_symbols = 10000
  input_symbols = [random.randint(0x0, 0xF) for _ in range(number_of_symbols)]
  enc_in_data = generate_input_data(input_symbols)
  enc_out_data = generate_expected_output_data(input_symbols)

  await reset(dut.clk, dut.rst)

  sim = init_sim(dut, enc_out_data)
  measure_coroutine = measure_time(sim.clk, sim.send_event, sim.terminate)
  measure_task = cocotb.start_soon(measure_coroutine)

  recv_coroutine = recv(sim.clk, sim.output_s, sim.recv_stop_event)
  recv_task = cocotb.start_soon(recv_coroutine)

  await sim.driver._send(enc_in_data[0], None, sim.send_event)
  await sim.driver.send(enc_in_data[1:])

  await sim.terminate.wait()
  await recv_stop(recv_task, sim.recv_stop_event)
  cycles = await Join(measure_task)
  sim.monitor.kill()

  return number_of_symbols / (cycles - delay)


@cocotb.test(timeout_time=1, timeout_unit="ms")
async def overflow_test(dut):
  """Cocotb test for checking overflow handling in the RLE encoder"""

  input_symbols = [0xA, 0xA, 0xA, 0xA, 0xB]
  await verify_output(dut, input_symbols)


@cocotb.test(timeout_time=150, timeout_unit="ms")
async def rle_measurements_test(dut):
  """Cocotb test for measuring delay and performance of the RLE encoder"""

  clock = Clock(dut.clk, 10, units="us")
  cocotb.start_soon(clock.start())

  delay = await measure_delay(dut)
  performance = await measure_performance(dut, delay)
  data = create_dashboard_json_with_measurements(delay, performance)
  dump_str = get_dashboard_json_dump_str(data)
  dut._log.info(dump_str)


class RLETest(test_base.TestCase):
  def test_rle(self):
    runfiles._BASE_PATH = "com_icarus_iverilog"
    iverilog_path = Path(runfiles.get_path("iverilog"))
    vvp_path = Path(runfiles.get_path("vvp"))
    os.environ["PATH"] += os.pathsep + str(iverilog_path.parent)
    os.environ["PATH"] += os.pathsep + str(vvp_path.parent)

    hdl_toplevel = "rle_enc"
    output_dir = os.path.join(
      os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", ""), "sim_build"
    )
    test_module = [Path(__file__).stem]

    runner = get_runner("icarus")
    runner.build(
      verilog_sources=["xls/modules/rle/rle_enc_compression_block.v"],
      hdl_toplevel=hdl_toplevel,
      timescale=("1ns", "1ps"),
      waves=True,
      build_dir=output_dir,
    )
    results_xml = runner.test(
      hdl_toplevel=hdl_toplevel,
      test_module=test_module,
      waves=True,
    )
    check_results_file(results_xml)


if __name__ == "__main__":
  test_base.main()
