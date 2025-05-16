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

from xls.common import runfiles


def setup_com_iverilog():
  iverilog_path = pathlib.Path(runfiles.get_path("iverilog", repository = "com_icarus_iverilog"))
  vvp_path = pathlib.Path(runfiles.get_path("vvp", repository = "com_icarus_iverilog"))
  os.environ["PATH"] += os.pathsep + str(iverilog_path.parent)
  os.environ["PATH"] += os.pathsep + str(vvp_path.parent)
  build_dir = pathlib.Path(os.environ['BUILD_WORKING_DIRECTORY'], "sim_build")
  return build_dir

def run_test(toplevel, test_module, verilog_sources):
  """Builds and runs a Cocotb testbench using Icarus Verilog."""
  build_dir = setup_com_iverilog()
  runner = get_runner("icarus")
  runner.build(
    verilog_sources=verilog_sources,
    hdl_toplevel=toplevel,
    timescale=("1ns", "1ps"),
    build_dir=build_dir,
    defines={"SIMULATION": "1"},
    waves=True,
  )

  results_xml = runner.test(
     hdl_toplevel=toplevel,
     test_module=test_module,
     waves=True,
  )
  check_results_file(results_xml)

@cocotb.coroutine
async def reset(clk, rst, cycles=1):
  """Cocotb coroutine that performs the reset."""
  rst.value = 1
  await ClockCycles(clk, cycles)
  rst.value = 0
