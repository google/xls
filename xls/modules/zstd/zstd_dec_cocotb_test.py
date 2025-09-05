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
  pregenerated_testing_routine,
  run_test,
  test_csr,
  test_reset,
  FseTableRecord
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


# Tests with pregenerated inputs
#
# block type and literal type in arguments and file names reflect what was used
# for generating them. The file names also contain the value of seed but the
# bazel environment is not hermetic in terms of reproducible results with the
# same seed value
# The tests are disabled by default as none of them passes currently.
# Use them to verify progress in specific parts of the decoder.

# TODO the workdir / data relation is weird. How to pass this better?
PREGENERATED_FILES_DIR = "../xls/modules/zstd/data/"


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_raw_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_raw_1.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_raw_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_raw_2.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_rle_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_rle_1.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def pregenerated_compressed_rle_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_rle_2.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def pregenerated_compressed_random_1(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_random_1.zst"
  test_cases = 1

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

  expected_huffman_codes = [
    [
      {"code": 0x00, "length": 8, "symbol": 0x31},
      {"code": 0x01, "length": 8, "symbol": 0x35},
      {"code": 0x20, "length": 3, "symbol": 0x39},
      {"code": 0x02, "length": 8, "symbol": 0x6E},
      {"code": 0x03, "length": 8, "symbol": 0x72},
      {"code": 0x08, "length": 6, "symbol": 0x76},
      {"code": 0x40, "length": 2, "symbol": 0x7A},
      {"code": 0x04, "length": 8, "symbol": 0xAF},
      {"code": 0x05, "length": 8, "symbol": 0xB3},
      {"code": 0x0C, "length": 6, "symbol": 0xB7},
      {"code": 0x80, "length": 1, "symbol": 0xBB},
      {"code": 0x06, "length": 8, "symbol": 0xF0},
      {"code": 0x07, "length": 8, "symbol": 0xF4},
      {"code": 0x10, "length": 4, "symbol": 0xF8},
    ],
    [
      {"code": 0x02, "length": 2, "symbol": 0x01},
      {"code": 0x00, "length": 3, "symbol": 0x66},
      {"code": 0x04, "length": 1, "symbol": 0x9C},
      {"code": 0x01, "length": 3, "symbol": 0xCB},
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

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def pregenerated_compressed_random_2(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_random_2.zst"
  await pregenerated_testing_routine(dut, input_name)


# Tests with predefined FSE tables and Huffman-encoded literals


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_107958(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_107958.zst"
  )

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

  await pregenerated_testing_routine(
    dut,
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
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_210872(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_210872.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_299289(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_299289.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_319146(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_319146.zst"
  )

  expected_huffman_codes = [
    [
      {"code": 0x300, "length": 2, "symbol": 0x07},
      {"code": 0x60, "length": 5, "symbol": 0x0E},
      {"code": 0x0A, "length": 9, "symbol": 0x15},
      {"code": 0x00, "length": 10, "symbol": 0x1C},
      {"code": 0x180, "length": 3, "symbol": 0x25},
      {"code": 0x30, "length": 6, "symbol": 0x2C},
      {"code": 0x0C, "length": 9, "symbol": 0x33},
      {"code": 0x01, "length": 10, "symbol": 0x3A},
      {"code": 0xC0, "length": 4, "symbol": 0x43},
      {"code": 0x18, "length": 7, "symbol": 0x4A},
      {"code": 0x02, "length": 10, "symbol": 0x51},
      {"code": 0x100, "length": 4, "symbol": 0x61},
      {"code": 0x20, "length": 7, "symbol": 0x68},
      {"code": 0x03, "length": 10, "symbol": 0x6F},
      {"code": 0x80, "length": 5, "symbol": 0x7F},
      {"code": 0x10, "length": 8, "symbol": 0x86},
      {"code": 0x04, "length": 10, "symbol": 0x8D},
      {"code": 0x200, "length": 3, "symbol": 0x96},
      {"code": 0x40, "length": 6, "symbol": 0x9D},
      {"code": 0x0E, "length": 9, "symbol": 0xA4},
      {"code": 0x05, "length": 10, "symbol": 0xAB},
      {"code": 0x280, "length": 3, "symbol": 0xB4},
      {"code": 0x50, "length": 6, "symbol": 0xBB},
      {"code": 0x06, "length": 10, "symbol": 0xC2},
      {"code": 0x07, "length": 10, "symbol": 0xC9},
      {"code": 0x140, "length": 4, "symbol": 0xD2},
      {"code": 0x28, "length": 7, "symbol": 0xD9},
      {"code": 0x08, "length": 10, "symbol": 0xE0},
      {"code": 0xA0, "length": 5, "symbol": 0xF0},
      {"code": 0x14, "length": 8, "symbol": 0xF7},
      {"code": 0x09, "length": 10, "symbol": 0xFE},
    ]
  ]

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
  )


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_331938(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_331938.zst"
  )

  expected_huffman_codes = [
    [
      {"code": 0x00, "length": 9, "symbol": 0x13},
      {"code": 0x20, "length": 5, "symbol": 0x1B},
      {"code": 0x01, "length": 9, "symbol": 0x32},
      {"code": 0x10, "length": 6, "symbol": 0x3A},
      {"code": 0x100, "length": 2, "symbol": 0x42},
      {"code": 0x02, "length": 9, "symbol": 0x51},
      {"code": 0x18, "length": 6, "symbol": 0x59},
      {"code": 0x180, "length": 2, "symbol": 0x61},
      {"code": 0x08, "length": 7, "symbol": 0x78},
      {"code": 0x80, "length": 3, "symbol": 0x80},
      {"code": 0x0C, "length": 7, "symbol": 0x97},
      {"code": 0xC0, "length": 3, "symbol": 0x9F},
      {"code": 0x04, "length": 8, "symbol": 0xB6},
      {"code": 0x40, "length": 4, "symbol": 0xBE},
      {"code": 0x06, "length": 8, "symbol": 0xD5},
      {"code": 0x60, "length": 4, "symbol": 0xDD},
      {"code": 0x03, "length": 9, "symbol": 0xF4},
      {"code": 0x30, "length": 5, "symbol": 0xFC},
    ]
  ]

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
  )


@cocotb.test(timeout_time=350, timeout_unit="ms")
async def fse_huffman_literals_predefined_sequences_seed_333824(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "fse_huffman_literals_predefined_sequences_seed_333824.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


# Test cases crated manually to allow working with small sizes of inputs.


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def pregenerated_compressed_minimal(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_compressed_minimal.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def pregenerated_uncompressed(dut):
  input_name = PREGENERATED_FILES_DIR + "pregenerated_uncompressed.zst"
  await pregenerated_testing_routine(dut, input_name)


# Test cases with predefined FSE tables and RAW/RLE literals


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_406229(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_406229.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_411034(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_411034.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_413015(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_413015.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_436165(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_436165.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_464057(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_464057.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_predefined_sequences_seed_466803(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_predefined_sequences_seed_466803.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_422473(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_422473.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_436965(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_436965.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_predefined_sequences_seed_462302(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_predefined_sequences_seed_462302.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_raw_literals_predefined_sequences_seed_408158(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_raw_literals_predefined_sequences_seed_408158.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_raw_literals_predefined_sequences_seed_499212(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_raw_literals_predefined_sequences_seed_499212.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


# Tests with inputs that correspond to the values in arrays defined in
# data/*.x files


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def comp_frame_fse_comp(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_fse_comp.zst"

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

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_fse_lookups=expected_fse_lookups,
  )


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_fse_repeated(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_fse_repeated.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_huffman(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_huffman.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def comp_frame_huffman_fse(dut):
  input_name = PREGENERATED_FILES_DIR + "comp_frame_huffman_fse.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def raw_literals_compressed_sequences_seed_903062(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "raw_literals_compressed_sequences_seed_903062.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def raw_literals_rle_sequences_seed_700216(dut):
  input_name = PREGENERATED_FILES_DIR + "raw_literals_rle_sequences_seed_700216.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=1000, timeout_unit="ms")
async def rle_literals_compressed_sequences_seed_701326(dut):
  input_name = (
    PREGENERATED_FILES_DIR + "rle_literals_compressed_sequences_seed_701326.zst"
  )
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=200, timeout_unit="ms")
async def rle_literals_rle_sequences_seed_2(dut):
  input_name = PREGENERATED_FILES_DIR + "rle_literals_rle_sequences_seed_2.zst"
  await pregenerated_testing_routine(dut, input_name)


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_compressed_sequences_seed_400077(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_compressed_sequences_seed_400077.zst"
  )

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

  expected_huffman_codes = [
    [
      {"code": 0x00, "length": 4, "symbol": 0x00},
      {"code": 0x02, "length": 3, "symbol": 0x3D},
      {"code": 0x04, "length": 2, "symbol": 0x7A},
      {"code": 0x08, "length": 1, "symbol": 0xB7},
      {"code": 0x01, "length": 4, "symbol": 0xC3},
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

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
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

  expected_huffman_weights = [
    [
      0x11111111,
      0x11111111,
    ]
  ]

  expected_huffman_codes = [
    [
      {"code": 0x00, "length": 4, "symbol": 0x00},
      {"code": 0x01, "length": 4, "symbol": 0x01},
      {"code": 0x02, "length": 4, "symbol": 0x02},
      {"code": 0x03, "length": 4, "symbol": 0x03},
      {"code": 0x04, "length": 4, "symbol": 0x04},
      {"code": 0x05, "length": 4, "symbol": 0x05},
      {"code": 0x06, "length": 4, "symbol": 0x06},
      {"code": 0x07, "length": 4, "symbol": 0x07},
      {"code": 0x08, "length": 4, "symbol": 0x08},
      {"code": 0x09, "length": 4, "symbol": 0x09},
      {"code": 0x0A, "length": 4, "symbol": 0x0A},
      {"code": 0x0B, "length": 4, "symbol": 0x0B},
      {"code": 0x0C, "length": 4, "symbol": 0x0C},
      {"code": 0x0D, "length": 4, "symbol": 0x0D},
      {"code": 0x0E, "length": 4, "symbol": 0x0E},
      {"code": 0x0F, "length": 4, "symbol": 0x0F},
    ]
  ]

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
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

  expected_huffman_weights = [
    [
      0x11111111,
      0x11111111,
    ]
  ]

  expected_huffman_codes = [
    [
      {"code": 0x00, "length": 4, "symbol": 0x00},
      {"code": 0x01, "length": 4, "symbol": 0x01},
      {"code": 0x02, "length": 4, "symbol": 0x02},
      {"code": 0x03, "length": 4, "symbol": 0x03},
      {"code": 0x04, "length": 4, "symbol": 0x04},
      {"code": 0x05, "length": 4, "symbol": 0x05},
      {"code": 0x06, "length": 4, "symbol": 0x06},
      {"code": 0x07, "length": 4, "symbol": 0x07},
      {"code": 0x08, "length": 4, "symbol": 0x08},
      {"code": 0x09, "length": 4, "symbol": 0x09},
      {"code": 0x0A, "length": 4, "symbol": 0x0A},
      {"code": 0x0B, "length": 4, "symbol": 0x0B},
      {"code": 0x0C, "length": 4, "symbol": 0x0C},
      {"code": 0x0D, "length": 4, "symbol": 0x0D},
      {"code": 0x0E, "length": 4, "symbol": 0x0E},
      {"code": 0x0F, "length": 4, "symbol": 0x0F},
    ]
  ]

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
    expected_huffman_weights=expected_huffman_weights,
  )


@cocotb.test(timeout_time=2000, timeout_unit="ms")
async def treeless_huffman_literals_rle_sequences_seed_403927(dut):
  input_name = (
    PREGENERATED_FILES_DIR
    + "treeless_huffman_literals_rle_sequences_seed_403927.zst"
  )

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

  expected_huffman_codes = [
    [
      {"code": 0x0C, "length": 6, "symbol": 0x04},
      {"code": 0x00, "length": 8, "symbol": 0x07},
      {"code": 0x08, "length": 7, "symbol": 0x29},
      {"code": 0x01, "length": 8, "symbol": 0x2C},
      {"code": 0x0A, "length": 7, "symbol": 0x4E},
      {"code": 0x02, "length": 8, "symbol": 0x51},
      {"code": 0x80, "length": 1, "symbol": 0x70},
      {"code": 0x03, "length": 8, "symbol": 0x73},
      {"code": 0x04, "length": 8, "symbol": 0x76},
      {"code": 0x40, "length": 2, "symbol": 0x95},
      {"code": 0x05, "length": 8, "symbol": 0x98},
      {"code": 0x20, "length": 3, "symbol": 0xBA},
      {"code": 0x06, "length": 8, "symbol": 0xBD},
      {"code": 0x10, "length": 4, "symbol": 0xDF},
      {"code": 0x07, "length": 8, "symbol": 0xE2},
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

  await pregenerated_testing_routine(
    dut,
    input_name,
    expected_huffman_codes=expected_huffman_codes,
    expected_huffman_weights=expected_huffman_weights,
    expected_fse_huffman_lookups=expected_fse_huffman_lookups,
  )


# Tests with inputs generated randomly on test execution


@cocotb.test(timeout_time=5000, timeout_unit="ms")
async def zstd_compressed_frames_test(dut):
  test_cases = 1
  block_type = data_generator.BlockType.COMPRESSED
  literal_type = data_generator.LiteralType.RAW
  await randomized_testing_routine(dut, test_cases, block_type, literal_type)

if __name__ == "__main__":
  test_module = [pathlib.Path(__file__).stem]
  run_test(test_module, sim="icarus")
