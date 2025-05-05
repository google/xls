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

"""Helpers for setting up and running simulations using Icarus Verilog with Cocotb."""

import os
import pathlib

import cocotb
from cocotb.runner import check_results_file
from cocotb.runner import get_runner
from cocotb.triggers import ClockCycles
import cocotbext.axi.axi_channels as axi

from xls.common import runfiles


def setup_com_iverilog():
  iverilog_path = pathlib.Path(
    runfiles.get_path("iverilog", repository = "com_icarus_iverilog")
  )
  vvp_path = pathlib.Path(
    runfiles.get_path("vvp", repository = "com_icarus_iverilog")
  )
  os.environ["PATH"] += os.pathsep + str(iverilog_path.parent)
  os.environ["PATH"] += os.pathsep + str(vvp_path.parent)
  build_dir = pathlib.Path("sim_build").absolute()
  return build_dir


def run_test(toplevel, test_module, verilog_sources, timescale=("1ns", "1ps")):
  """Builds and runs a Cocotb testbench using Icarus Verilog."""
  build_dir = setup_com_iverilog()
  runner = get_runner("icarus")()
  build_args = []

  cmds_file = build_dir / pathlib.Path("cmds.f")
  cmds_file.parent.mkdir(parents=True, exist_ok=True)
  with open(cmds_file, "w") as f:
    f.write("+timescale+{}/{}\n".format(*timescale))
  build_args += ["-f", str(cmds_file)]

  dump_file = build_dir / pathlib.Path("cocotb_iverilog_dump.v")
  wave_file = build_dir / pathlib.Path(f"{toplevel}.fst")
  with open(dump_file, "w") as f:
    f.write("module cocotb_iverilog_dump();\n")
    f.write("initial begin\n")
    f.write(f'    $dumpfile("{wave_file}");\n')
    f.write(f"    $dumpvars(0, {toplevel});\n")
    f.write("end\n")
    f.write("endmodule\n")
  wave_file.parent.mkdir(parents=True, exist_ok=True)

  runner.build(
    verilog_sources=(verilog_sources + [str(dump_file)]),
    toplevel=toplevel,
    build_dir=build_dir,
    extra_args=build_args,
    defines={"SIMULATION": "1"},
  )

  try:
    results_xml = runner.test(
       toplevel=toplevel,
       py_module=test_module,
    )
  finally:
    check_results_file(results_xml)

@cocotb.coroutine
async def reset(clk, rst, cycles=1):
  """Cocotb coroutine that performs the reset."""
  rst.value = 1
  await ClockCycles(clk, cycles)
  rst.value = 0

def connect_axi_read_bus(dut, name=""):
  AXI_AR = "axi_ar"
  AXI_R = "axi_r"

  if name != "":
      name += "_"

  bus_axi_ar = axi.AxiARBus.from_prefix(dut, name + AXI_AR)
  bus_axi_r = axi.AxiRBus.from_prefix(dut, name + AXI_R)

  return axi.AxiReadBus(bus_axi_ar, bus_axi_r)

def connect_axi_write_bus(dut, name=""):
  AXI_AW = "axi_aw"
  AXI_W = "axi_w"
  AXI_B = "axi_b"

  if name != "":
      name += "_"

  bus_axi_aw = axi.AxiAWBus.from_prefix(dut, name + AXI_AW)
  bus_axi_b = axi.AxiBBus.from_prefix(dut, name + AXI_B)
  bus_axi_w = axi.AxiWBus.from_prefix(dut, name + AXI_W)

  return axi.AxiWriteBus(bus_axi_aw, bus_axi_w, bus_axi_b)

def connect_axi_bus(dut, name=""):
  bus_axi_read = connect_axi_read_bus(dut, name)
  bus_axi_write = connect_axi_write_bus(dut, name)

  return axi.AxiBus(bus_axi_write, bus_axi_read)
