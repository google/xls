# Lint as: python3
#
# Copyright 2020 Google LLC
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

"""Testing layer that supplements the test case class.

This is an adapter that helps paper over differences between Google-internal and
OSS/external testing facilities.
"""

# pylint: disable=useless-super-delegation

import os
import sys
import tempfile
from xls.common.python import init_xls
from typing import Sized, Optional

from absl.testing import absltest as unittest


class XlsTestCase(unittest.TestCase):

  def assertEndsWith(self, s: str, suffix: str) -> None:
    """Asserts that s ends with the given suffix."""
    self.assertTrue(s.endswith(suffix))

  def assertLen(self, sequence: Sized, target: int) -> None:
    """Assert that the len of sequence is target."""
    assert len(sequence) == target, 'Got sequence of length %d, want %d' % (
       len(sequence), target)

  def assertEmpty(self, sequence: Sized) -> None:
    """Assert that the len of sequence is target."""
    assert len(sequence) == 0, 'Got non-empty sequence, got len %d: %s' % (
       len(sequence), sequence)

  def assertNotEmpty(self, sequence: Sized) -> None:
    """Assert that the len of sequence is target."""
    assert len(sequence) != 0, 'Got empty sequence, want length != 0: %s' % (
       sequence)

  def create_tempfile(self, content: Optional[str] = None):
    f = tempfile.NamedTemporaryFile()
    f.full_path = os.path.realpath(f.name)
    if content:
      f.write(content.encode('utf-8'))
      f.flush()
    return f

  def create_tempdir(self) -> str:
    return tempfile.mkdtemp()


def create_named_output_text_file(name: str):
  return tempfile.NamedTemporaryFile(suffix=name, delete=False).name


def main():
  init_xls.init_xls(sys.argv)
  unittest.main()
