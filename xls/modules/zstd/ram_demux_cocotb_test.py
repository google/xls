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
from cocotb.binary import BinaryValue
from cocotb.clock import Clock
from cocotb.triggers import Event, ClockCycles, RisingEdge
from cocotb_bus.scoreboard import Scoreboard
from cocotbext.axi import axi_channels
from cocotbext.axi.axi_ram import AxiRamRead
from cocotbext.axi.sparse_memory import SparseMemory

import xls.modules.zstd.cocotb.channel as xlschannel
from xls.modules.zstd.cocotb import utils
from xls.modules.zstd.cocotb import xlsstruct

warnings.filterwarnings("ignore", category=DeprecationWarning)

ADDR_W = 10
DATA_W = 64
NUM_PARTITIONS = 64
SEL_W = 1


@xlsstruct.xls_dataclass
class SelReq(xlsstruct.XLSStruct):
    sel: SEL_W


@xlsstruct.xls_dataclass
class ReadReq(xlsstruct.XLSStruct):
    addr: ADDR_W
    mask: NUM_PARTITIONS


@xlsstruct.xls_dataclass
class ReadResp(xlsstruct.XLSStruct):
    data: DATA_W


@xlsstruct.xls_dataclass
class WriteReq(xlsstruct.XLSStruct):
    addr: ADDR_W
    data: DATA_W
    mask: NUM_PARTITIONS


@xlsstruct.xls_dataclass
class WriteResp(xlsstruct.XLSStruct):
    pass


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


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def test_mem_reader(dut):

    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    sel_resp_channel = xlschannel.XLSChannel(dut, "ram_demux__sel_resp_s", dut.clk)
    rd_resp_channel = xlschannel.XLSChannel(dut, "ram_demux__rd_resp_s", dut.clk)
    wr_resp_channel = xlschannel.XLSChannel(
        dut, "ram_demux__wr_resp_s", dut.clk, start_now=True
    )

    sel_driver = xlschannel.XLSChannelDriver(dut, "ram_demux__sel_req_r", dut.clk)
    rd_req_driver = xlschannel.XLSChannelDriver(dut, "ram_demux__rd_req_r", dut.clk)
    wr_req_driver = xlschannel.XLSChannelDriver(dut, "ram_demux__wr_req_r", dut.clk)

    dut.rst.setimmediatevalue(0)
    await ClockCycles(dut.clk, 10)
    dut.rst.setimmediatevalue(1)
    await ClockCycles(dut.clk, 10)
    dut.rst.setimmediatevalue(0)

    sel_resp_channel.rdy.setimmediatevalue(1)
    rd_resp_channel.rdy.setimmediatevalue(1)
    wr_resp_channel.rdy.setimmediatevalue(1)

    await sel_driver.send(SelReq(0))
    while True:
        await RisingEdge(dut.clk)
        if sel_resp_channel.rdy.value and sel_resp_channel.vld.value:
            break

    await wr_req_driver.send(WriteReq(addr=123, data=0x10, mask=0xFFFF_FFFF_FFFF_FFFF))
    while True:
        await RisingEdge(dut.clk)
        if wr_resp_channel.rdy.value and wr_resp_channel.vld.value:
            break

    await sel_driver.send(SelReq(1))
    while True:
        await RisingEdge(dut.clk)
        if sel_resp_channel.rdy.value and sel_resp_channel.vld.value:
            break

    await wr_req_driver.send(WriteReq(addr=256, data=0x3, mask=0xFFFF_FFFF_FFFF_FFFF))
    while True:
        await RisingEdge(dut.clk)
        if wr_resp_channel.rdy.value and wr_resp_channel.vld.value:
            break

    await sel_driver.send(SelReq(0))
    while True:
        await RisingEdge(dut.clk)
        if sel_resp_channel.rdy.value and sel_resp_channel.vld.value:
            break

    await rd_req_driver.send(ReadReq(addr=123, mask=0xFFFF_FFFF_FFFF_FFFF))
    while True:
        await RisingEdge(dut.clk)
        if rd_resp_channel.rdy.value and rd_resp_channel.vld.value:
            assert rd_resp_channel.data.value == 0x10
            break

    await sel_driver.send(SelReq(1))
    while True:
        await RisingEdge(dut.clk)
        if sel_resp_channel.rdy.value and sel_resp_channel.vld.value:
            break

    await rd_req_driver.send(ReadReq(addr=256, mask=0xFFFF_FFFF_FFFF_FFFF))
    while True:
        await RisingEdge(dut.clk)
        if rd_resp_channel.rdy.value and rd_resp_channel.vld.value:
            assert rd_resp_channel.data.value == 0x3
            break


if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).parent))

    toplevel = "ram_demux_wrapper"
    verilog_sources = [
        "xls/modules/zstd/ram_demux.v",
        "xls/modules/zstd/rtl/ram_1r1w.v",
        "xls/modules/zstd/rtl/ram_demux_wrapper.v",
    ]
    test_module = [pathlib.Path(__file__).stem]
    utils.run_test(toplevel, test_module, verilog_sources)
