#
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
"""Tests for xls/tools/gather_design_stats.py."""

import subprocess

from absl.testing import absltest
from xls.common import runfiles
from xls.common import test_base

_GATHER_DESIGN_STATS_PATH = runfiles.get_path('xls/tools/gather_design_stats')
_STA_LOG_PATH = runfiles.get_path(
    'xls/tools/testdata/find_index_5ps_model_unit_verilog_sta_by_stage_sta.log'
)
_SYN_LOG_PATH = runfiles.get_path(
    'xls/tools/testdata/find_index_5ps_model_unit_verilog_synth_by_stage_yosys_output.log.gz'
)
_EXP_TEXTPROTO_PATH = runfiles.get_path(
    'xls/tools/testdata/find_index_5ps_model_unit_expected.textproto'
)


class GatherDesignStatsMainTest(test_base.TestCase):

  def test_degenerate_use(self):
    out_textproto_file = self.create_tempfile()
    subprocess.run(
        [
            _GATHER_DESIGN_STATS_PATH,
            '--out',
            out_textproto_file.full_path,
        ],
        check=True,
    )
    wc_output = subprocess.check_output([
        'wc',
        out_textproto_file.full_path,
    ]).decode('utf-8')
    self.assertIn('0 0 0', wc_output)

  def test_sample_use(self):
    out_textproto_file = self.create_tempfile()
    subprocess.run(
        [
            _GATHER_DESIGN_STATS_PATH,
            '--out',
            out_textproto_file.full_path,
            _SYN_LOG_PATH,
            _STA_LOG_PATH,
        ],
        check=True,
    )
    diff_output = subprocess.check_output([
        'diff',
        out_textproto_file.full_path,
        _EXP_TEXTPROTO_PATH,
    ]).decode('utf-8')
    self.assertEmpty(diff_output)


if __name__ == '__main__':
  absltest.main()
