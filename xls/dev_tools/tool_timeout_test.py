#
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

import subprocess

from absl.testing import absltest
from xls.common import runfiles

_TEST_BINARY = runfiles.get_path("xls/dev_tools/tool_timeout_test_main")

_LOGGING_FLAGS = []


class ToolTimeoutTest(absltest.TestCase):

  def test_timeout_will_trigger(self):
    res = subprocess.run(
        [_TEST_BINARY, "--timeout=4s"] + _LOGGING_FLAGS,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    self.assertEqual(res.returncode, 1, str(res))
    self.assertRegex(res.stderr, b"timeout", str(res))

  def test_timeout_does_not_trigger(self):
    res = subprocess.run(
        [_TEST_BINARY, "--timeout=10m", "--wait_for=4s"] + _LOGGING_FLAGS,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    self.assertEqual(res.returncode, 0, str(res))
    self.assertRegex(res.stdout, b"Waited for", str(res))

  def test_no_timeout_does_not_trigger(self):
    res = subprocess.run(
        [_TEST_BINARY, "--wait_for=4s"] + _LOGGING_FLAGS,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    self.assertEqual(res.returncode, 0, str(res))
    self.assertRegex(res.stdout, b"Waited for", str(res))


if __name__ == "__main__":
  absltest.main()
