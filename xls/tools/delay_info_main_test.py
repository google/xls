#
# Copyright 2021 The XLS Authors
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
"""Tests for xls.tools.delay_info_main."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

DELAY_INFO_MAIN_PATH = runfiles.get_path('xls/tools/delay_info_main')

NOT_ADD_IR = """package not_add

top fn not_add(x: bits[32], y: bits[32]) -> bits[32] {
  sum: bits[32] = add(x, y)
  ret not_sum: bits[32] = not(sum)
}
"""
NOT_ADD_SCHEDULE = """
schedules {
  key: "not_add"
  value {
    function: "not_add"
    stages {
      stage: 0
      timed_nodes: { node: "x" node_delay_ps: 0 }
      timed_nodes: { node: "y" node_delay_ps: 0 }
    }
    stages {
      stage: 1
      timed_nodes: { node: "sum" node_delay_ps: 1 }
      timed_nodes: { node: "not_sum" node_delay_ps: 1 }
    }
  }
}
"""


class DelayInfoMainTest(test_base.TestCase):

  def test_without_schedule(self):
    """Test tool without specifying --schedule_path."""
    ir_file = self.create_tempfile(content=NOT_ADD_IR)

    optimized_ir = subprocess.check_output(
        [DELAY_INFO_MAIN_PATH, '--delay_model=unit', ir_file.full_path]
    ).decode('utf-8')
    self.assertEqual(
        optimized_ir,
        """# Critical path:
      2ps (+  1ps): not_sum: bits[32] = not(sum: bits[32], id=4)
      1ps (+  1ps): sum: bits[32] = add(x: bits[32], y: bits[32], id=3)
      0ps (+  0ps): y: bits[32] = param(y, id=2)

# Delay of all nodes:
x               :     0ps
y               :     0ps
sum             :     1ps
not_sum         :     1ps
""",
    )

  def test_with_schedule(self):
    """Test tool with specifying --schedule_path."""
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    schedule_file = self.create_tempfile(content=NOT_ADD_SCHEDULE)

    optimized_ir = subprocess.check_output([
        DELAY_INFO_MAIN_PATH,
        '--delay_model=unit',
        '--alsologtostderr',
        f'--schedule_path={schedule_file.full_path}',
        ir_file.full_path,
    ]).decode('utf-8')

    self.assertEqual(
        optimized_ir,
        """# Critical path for stage 0:
      1ps (+  1ps): tuple.7: (bits[32], bits[32]) = tuple(x: bits[32], y: bits[32], id=7)
      0ps (+  0ps): y: bits[32] = param(y, id=6)

# Critical path for stage 1:
      2ps (+  1ps): not_sum: bits[32] = not(sum: bits[32], id=11)
      1ps (+  1ps): sum: bits[32] = add(x: bits[32], y: bits[32], id=10)
      0ps (+  0ps): y: bits[32] = param(y, id=9)

# Delay of all nodes:
x               :     0ps
y               :     0ps
sum             :     1ps
not_sum         :     1ps
""",
    )


if __name__ == '__main__':
  test_base.main()
