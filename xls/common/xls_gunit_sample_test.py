# Copyright 2022 The XLS Authors
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
"""Tests for xls_gunit_sample."""

import subprocess
import sys

from absl.testing import absltest
from xls.common import runfiles


class XlsGunitSampleTest(absltest.TestCase):

  def test_filter_flag_works(self):
    """Tests there is a flag to filter tests on xls_gunit_main-linked targets.

    Unfortunately, this flag is named different things in different
    environments; e.g. internal vs OSS.
    """
    xls_gunit_sample_path = runfiles.get_path('xls/common/xls_gunit_sample')
    try:
      output = subprocess.check_output(
          [xls_gunit_sample_path, '--gtest_filter=*SomeTestName'],
          encoding='utf-8',
      )
      print('Success: gtest_filter flag present', file=sys.stderr)
    except subprocess.CalledProcessError:
      output = subprocess.check_output(
          [xls_gunit_sample_path, '--gunit_filter=*SomeTestName'],
          encoding='utf-8',
      )
      print('Success: gunit_filter flag present', file=sys.stderr)
    self.assertIn('XlsGunitSampleTest.SomeTestName', output)


if __name__ == '__main__':
  absltest.main()
