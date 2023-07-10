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

import cocotb

from cocotb.clock import Clock
from cocotb.triggers import Event, ClockCycles
from cocotb.binary import BinaryValue

from cocotb_bus.scoreboard import Scoreboard
from cocotb_xls import XLSChannelDriver, XLSChannelMonitor

from typing import List, Any


def list_to_binary_value(data: List, n_bits: int = 32, bigEndian=False) -> List[BinaryValue]:
    return [BinaryValue(x, n_bits, bigEndian) for x in data]


def init_sim(dut, data_to_send: List[BinaryValue], data_to_recv: List[BinaryValue]):
    clock = Clock(dut.clk, 10, units="us")

    driver = XLSChannelDriver(dut, "running_counter__base_r", dut.clk)
    monitor = XLSChannelMonitor(dut, "running_counter__cnt_s", dut.clk)

    scoreboard = Scoreboard(dut, fail_immediately=True)
    scoreboard.add_interface(monitor, data_to_recv)

    total_of_packets = len(data_to_recv)
    terminate = Event("Last packet of data received")

    def terminate_cb(transaction):
        if monitor.stats.received_transactions == total_of_packets:
            terminate.set()

    monitor.add_callback(terminate_cb)
    monitor.bus.rdy.setimmediatevalue(1)

    return (clock, driver, terminate)


@cocotb.coroutine
async def reset(clk, rst, cycles=1):
    rst.setimmediatevalue(1)
    await ClockCycles(clk, cycles)
    rst.value = 0


@cocotb.test(timeout_time=10, timeout_unit="ms")
async def counter_test(dut):
    data_to_send = list_to_binary_value([0x100, 0x100, 0x100])
    data_to_recv = list_to_binary_value([0x102, 0x103, 0x104])

    (clock, driver, terminate) = init_sim(dut, data_to_send, data_to_recv)

    cocotb.start_soon(clock.start())
    await reset(dut.clk, dut.rst, 10)
    await driver.write(data_to_send)
    await terminate.wait()
