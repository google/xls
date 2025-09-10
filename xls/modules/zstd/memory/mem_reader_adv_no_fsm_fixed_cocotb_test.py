#!/usr/bin/env python
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

import sys

from pathlib import Path
from xls.modules.zstd.cocotb.utils import run_test

if __name__ == "__main__":
  sys.path.append(str(Path(__file__).parent))

  toplevel = "mem_reader_wrapper"
  verilog_sources = [
    "xls/modules/zstd/memory/manual_axi_reader_internal_r.v",
    "xls/modules/zstd/memory/manual_mem_reader_adv_no_fsm.v",
    "xls/modules/zstd/memory/rtl/mem_reader_wrapper.v",
  ]
  test_module = [ "mem_reader_cocotb" ]
  run_test(toplevel, test_module, verilog_sources)

