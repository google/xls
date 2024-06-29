#
# Copyright 2020 The XLS Authors
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
"""Tests for xls.tools.netlist_interpreter_main."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

XLS_TOOLS = 'xls/tools/'
NETLIST_INTERPRETER_MAIN = runfiles.get_path(
    XLS_TOOLS + 'netlist_interpreter_main'
)
CELL_LIBRARY = runfiles.get_path(XLS_TOOLS + 'testdata/simple_cell.lib')


def run_netlist_interpreter(netlist, module, input_data, output_type):
  result = subprocess.check_output([
      NETLIST_INTERPRETER_MAIN,
      '--netlist=' + runfiles.get_path(XLS_TOOLS + netlist),
      '--module_name=' + module,
      '--input=' + input_data,
      '--output_type=' + output_type,
      '--cell_library=' + CELL_LIBRARY,
  ])
  return result.decode('utf-8').strip()


class NetlistTranspilerMainTest(test_base.TestCase):

  def test_sqrt(self):
    res = run_netlist_interpreter(
        'testdata/sqrt.v', 'isqrt', 'bits[16]:100', 'bits[8]'
    )
    self.assertEqual(res, 'bits[8]:0xa')

  def test_ifte(self):
    res = run_netlist_interpreter(
        'testdata/ifte.v',
        'ifte',
        'bits[1]:1;bits[8]:0xaa;bits[8]:0xbb',
        'bits[8]',
    )
    self.assertEqual(res, 'bits[8]:0xaa')


if __name__ == '__main__':
  test_base.main()
