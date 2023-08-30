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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import cocotb
from cocotb.binary import BinaryValue
from cocotb.clock import Clock
from cocotb.handle import ModifiableObject, SimHandleBase
from cocotb.runner import check_results_file, get_runner
from cocotb.triggers import ClockCycles, Event, RisingEdge
from cocotb_bus.scoreboard import Scoreboard
from cocotb_xls import XLSChannel, XLSChannelDriver, XLSChannelMonitor

from xls.common import runfiles


@dataclass
class SimulationData:
  """Auxiliary structure used to store design-related data for the simulation"""

  clock: Clock
  driver: XLSChannelDriver
  monitor: XLSChannelMonitor
  scoreboard: Scoreboard
  data_r: XLSChannel
  data_s: XLSChannel
  terminate: Event
  clk: ModifiableObject
  rst: ModifiableObject


def init_sim(dut: SimHandleBase, data_to_recv: Sequence[BinaryValue]) -> SimulationData:
  """Extracts all design-related data required for simulation"""

  RECV_CHANNEL = "passthrough__data_r"
  SEND_CHANNEL = "passthrough__data_s"

  clock = Clock(dut.clk, 10, units="us")

  driver = XLSChannelDriver(dut, RECV_CHANNEL, dut.clk)
  monitor = XLSChannelMonitor(dut, SEND_CHANNEL, dut.clk)
  data_r = XLSChannel(dut, RECV_CHANNEL)
  data_s = XLSChannel(dut, SEND_CHANNEL)

  scoreboard = Scoreboard(dut, fail_immediately=True)
  scoreboard.add_interface(monitor, deepcopy(data_to_recv))

  expected_packet_count = len(data_to_recv)
  terminate = Event("Received the last packet of data")

  def terminate_cb(_):
    if monitor.stats.received_transactions == expected_packet_count:
      terminate.set()

  monitor.add_callback(terminate_cb)

  return SimulationData(
    clock=clock,
    driver=driver,
    monitor=monitor,
    scoreboard=scoreboard,
    data_r=data_r,
    data_s=data_s,
    terminate=terminate,
    clk=dut.clk,
    rst=dut.rst,
  )


@cocotb.coroutine
async def recv(clk, send_channel):
  """Cocotb coroutine that acts as a proc receiving data from a channel"""
  send_channel.rdy.setimmediatevalue(0)
  while True:
    send_channel.rdy.value = send_channel.vld.value
    await RisingEdge(clk)


@cocotb.coroutine
async def reset(clk, rst, cycles=1):
  """Cocotb coroutine that performs the reset"""
  rst.setimmediatevalue(1)
  await ClockCycles(clk, cycles)
  rst.value = 0


@cocotb.test(timeout_time=10, timeout_unit="ms")
async def passthrough_test(dut):
  """Cocotb test of the Passthrough proc"""
  test_data = [BinaryValue(x, n_bits=32, bigEndian=False) for x in range(10)]
  sim = init_sim(dut, test_data)

  cocotb.start_soon(sim.clock.start())
  await reset(sim.clk, sim.rst)

  cocotb.start_soon(recv(sim.clk, sim.data_s))
  await sim.driver.send(test_data)
  await sim.terminate.wait()


if __name__ == "__main__":
  runfiles._BASE_PATH = "com_icarus_iverilog"
  iverilog_path = Path(runfiles.get_path("iverilog"))
  vvp_path = Path(runfiles.get_path("vvp"))
  os.environ["PATH"] += os.pathsep + str(iverilog_path.parent)
  os.environ["PATH"] += os.pathsep + str(vvp_path.parent)

  runner = get_runner("icarus")
  runner.build(
    verilog_sources=["xls/examples/passthrough.v"],
    hdl_toplevel="passthrough",
    timescale=("1ns", "1ps"),
    waves=True,
  )
  results_xml = runner.test(
    hdl_toplevel="passthrough",
    test_module=[Path(__file__).stem],
    waves=True,
  )
  check_results_file(results_xml)
