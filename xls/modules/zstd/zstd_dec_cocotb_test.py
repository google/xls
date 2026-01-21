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

import cocotb
import pathlib
from xls.modules.zstd.cocotb import data_generator
from xls.modules.zstd.zstd_dec_cocotb_common import (
  randomized_testing_routine,
  run_test,
  test_csr,
  test_reset,
)

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
  await randomized_testing_routine(dut, test_cases, block_type)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def zstd_rle_frames_test(dut):
  test_cases = 5
  block_type = data_generator.BlockType.RLE
  await randomized_testing_routine(dut, test_cases, block_type)


@cocotb.test(timeout_time=5000, timeout_unit="ms")
async def zstd_compressed_frames_test(dut):
    test_cases = 1
    block_type = data_generator.BlockType.COMPRESSED
    literal_type = data_generator.LiteralType.RAW
    await randomized_testing_routine(dut, test_cases, block_type, literal_type)


if __name__ == "__main__":
    test_module = [pathlib.Path(__file__).stem]
    run_test(test_module, sim="icarus")
