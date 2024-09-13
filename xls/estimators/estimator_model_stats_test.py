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

"""Tests for xls/estimators/estimator_model_stats.py."""

import filecmp
import subprocess

from absl.testing import absltest
from xls.common import runfiles

_ESTIMATOR_MODEL_STATS_PATH = runfiles.get_path(
    'xls/estimators/estimator_model_stats'
)
_ASAP7_DELAY_DATA_POINTS_PATH = runfiles.get_path(
    'xls/estimators/testdata/asap7_delay_data_points.textproto'
)
_ASAP7_DELAY_MODEL_STATS_PATH = runfiles.get_path(
    'xls/estimators/testdata/asap7_delay_model_stats.csv'
)
_SKY130_DELAY_DATA_POINTS_PATH = runfiles.get_path(
    'xls/estimators/testdata/sky130_delay_data_points.textproto'
)
_SKY130_DELAY_MODEL_STATS_PATH = runfiles.get_path(
    'xls/estimators/testdata/sky130_delay_model_stats.csv'
)
_UNIT_DELAY_DATA_POINTS_PATH = runfiles.get_path(
    'xls/estimators/testdata/unit_delay_data_points.textproto'
)
_UNIT_DELAY_MODEL_STATS_PATH = runfiles.get_path(
    'xls/estimators/testdata/unit_delay_model_stats.csv'
)


class DelayModelStatsTest(absltest.TestCase):

  def test_asap7_delay_model_stats(self):
    """Reproduciable results for the ASAP7 delay model."""
    out_csv_file = self.create_tempfile()
    subprocess.run(
        [
            _ESTIMATOR_MODEL_STATS_PATH,
            '--output_csv',
            out_csv_file.full_path,
            _ASAP7_DELAY_DATA_POINTS_PATH,
        ],
        check=True,
    )
    self.assertTrue(
        filecmp.cmp(out_csv_file.full_path, _ASAP7_DELAY_MODEL_STATS_PATH)
    )

  def test_sky130_delay_model_stats(self):
    """Reproduciable results for the SKY130 delay model."""
    out_csv_file = self.create_tempfile()
    subprocess.run(
        [
            _ESTIMATOR_MODEL_STATS_PATH,
            '--output_csv',
            out_csv_file.full_path,
            _SKY130_DELAY_DATA_POINTS_PATH,
        ],
        check=True,
    )
    self.assertTrue(
        filecmp.cmp(out_csv_file.full_path, _SKY130_DELAY_MODEL_STATS_PATH)
    )

  def test_unit_delay_model_stats(self):
    """Reproduciable results for the unit delay model."""
    out_csv_file = self.create_tempfile()
    subprocess.run(
        [
            _ESTIMATOR_MODEL_STATS_PATH,
            '--output_csv',
            out_csv_file.full_path,
            _UNIT_DELAY_DATA_POINTS_PATH,
        ],
        check=True,
    )
    self.assertTrue(
        filecmp.cmp(out_csv_file.full_path, _UNIT_DELAY_MODEL_STATS_PATH)
    )


if __name__ == '__main__':
  absltest.main()
