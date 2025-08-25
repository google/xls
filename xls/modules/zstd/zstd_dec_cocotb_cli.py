# Copyright 2025 The XLS Authors
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


import sys
import cocotb
import os
import pathlib
from xls.modules.zstd.zstd_dec_cocotb_common import run_test, check_decoder_compliance
from xls.modules.zstd.zstd_dec_detailed_test import detailed_testing_routine
from multiprocessing import cpu_count

@cocotb.test(timeout_time=int(os.getenv("ZSTD_DEC_COCOTB_CLI_TIMEOUT", "5000")), timeout_unit="ms")
async def zstd_cli_test(dut):
    input_name = os.getenv("ZSTD_DEC_COCOTB_CLI_INPUT")
    print("input_name: ", input_name)
    await detailed_testing_routine(dut, input_name)

def usage():
      print(f"usage: {os.path.basename(sys.argv[0])} /abs/path/to/input.zst [timeout_is_ms]")
      sys.exit(1)

if __name__ == "__main__":
    help = "-h" in sys.argv or "--help" in sys.argv
    bad_params = len(sys.argv) not in (2,3)
    if bad_params or help:
        usage()

    if not os.path.isabs(sys.argv[1]):
        # bazel run changes the working dir to runfiles tree so relative paths won't work
        print(f"error: '{sys.argv[1]}' is not absolute path")
        usage()

    if not check_decoder_compliance(sys.argv[1]):
        print(f"error: '{sys.argv[1]}' is not suitable for the decoder parameters")
        sys.exit(1)
    
    # cocotb doesn't perserve global vars nor sys.argv
    # we work it around by passing arguments through env
    os.environ["ZSTD_DEC_COCOTB_CLI_INPUT"] = sys.argv[1]

    if len(sys.argv) == 3:
        os.environ["ZSTD_DEC_COCOTB_CLI_TIMEOUT"] = sys.argv[2]
    
    test_module = [pathlib.Path(__file__).stem]
    run_test(test_module, build_args=[
      "-Wno-fatal",
      "-Wwarn-ASSIGNIN",
      "--trace-fst", # trace in more space-efficient format than vcd
      "-O3",
      "--assert",
    ], sim="verilator")
