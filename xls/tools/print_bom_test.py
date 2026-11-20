#
# Copyright 2026 The XLS Authors
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

import os
import shutil
import subprocess

from xls.common import runfiles
from xls.common import test_base

PRINT_BOM_PATH = runfiles.get_path('xls/tools/print_bom')
BOM_METRICS_PATH = runfiles.get_path('xls/tools/testdata/bom.metrics.textproto')


class PrintBomTest(test_base.TestCase):

  def _stage_metrics_dir(self) -> tuple[str, str]:
    temp_dir = self.create_tempdir().full_path
    metrics_path = os.path.join(temp_dir, 'bom.metrics.textproto')
    shutil.copyfile(BOM_METRICS_PATH, metrics_path)
    return temp_dir, metrics_path

  def test_rejects_positional_arguments(self):
    temp_dir, metrics_path = self._stage_metrics_dir()

    # Even with a valid --root_path, extra positional arguments must fail with
    # usage guidance instead of being silently ignored.
    result = subprocess.run(
        [PRINT_BOM_PATH, f'--root_path={temp_dir}', metrics_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('positional arguments', result.stderr.lower())
    self.assertIn('--root_path', result.stderr)
    self.assertIn('--file_pattern', result.stderr)

    # Passing only a positional path should print usage rather than failing
    # with a misleading "No protobuf files" error.
    result = subprocess.run(
        [PRINT_BOM_PATH, metrics_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('positional arguments', result.stderr.lower())
    self.assertIn('--root_path', result.stderr)
    self.assertNotIn('No protobuf files', result.stderr)

  def test_table_and_csv_output(self):
    temp_dir, _ = self._stage_metrics_dir()

    table_output = subprocess.check_output(
        [
            PRINT_BOM_PATH,
            f'--root_path={temp_dir}',
            r'--file_pattern=.*bom\.metrics\.textproto',
        ],
        text=True,
    )
    self.assertIn('BOM_KIND_COMPARISON', table_output)
    self.assertIn('BOM_KIND_ADDER', table_output)

    csv_output = subprocess.check_output(
        [
            PRINT_BOM_PATH,
            f'--root_path={temp_dir}',
            '--output_as=csv',
            '--op_kind=BOM_KIND_ADDER',
        ],
        text=True,
    )
    self.assertEqual(
        csv_output,
        'Kind,Op,Input Width,Output Width,Count\nBOM_KIND_ADDER,add,8,8,1\n',
    )

  def test_help_describes_usage_and_file_pattern(self):
    result = subprocess.run(
        [PRINT_BOM_PATH, '--help'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    self.assertIn('--root_path', output)
    self.assertIn('--file_pattern', output)
    self.assertIn('Full-path regex', output)
    self.assertIn('.textproto', output)


if __name__ == '__main__':
  test_base.main()
