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
"""Tests for ir_converter_main."""

import os
import subprocess
import textwrap
from typing import Dict, Optional

from xls.common import runfiles
from xls.common import test_base


class IrConverterMainTest(test_base.TestCase):

  A_DOT_X = 'fn f() -> u32 { u32:42 }'
  B_DOT_X = 'fn f() -> u32 { u32:64 }'
  IR_CONVERTER_MAIN_PATH = runfiles.get_path('xls/dslx/ir_converter_main')

  def _ir_convert(self,
                  dslx_contents: Dict[str, str],
                  package_name: Optional[str] = None) -> str:
    tempdir = self.create_tempdir()
    tempfiles = []
    for filename, contents in dslx_contents.items():
      path = os.path.join(tempdir, filename)
      with open(path, 'w') as f:
        f.write(contents)
      tempfiles.append(path)
    cmd = [self.IR_CONVERTER_MAIN_PATH] + tempfiles
    if package_name is not None:
      cmd.append('--package_name=' + package_name)
    return subprocess.check_output(cmd, encoding='utf-8')

  def test_a_dot_x(self) -> None:
    self.assertEqual(
        self._ir_convert({'a.x': self.A_DOT_X}),
        textwrap.dedent("""\
    package a

    file_number 0 "fake_file.x"

    fn __a__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=42, id=1, pos=[(0,0,20)])
    }
    """))

  def test_b_dot_x(self) -> None:
    self.assertEqual(
        self._ir_convert({'b.x': self.B_DOT_X}),
        textwrap.dedent("""\
    package b

    file_number 0 "fake_file.x"

    fn __b__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=64, id=1, pos=[(0,0,20)])
    }
    """))

  def test_multi_file(self) -> None:
    self.assertEqual(
        self._ir_convert({
            'a.x': self.A_DOT_X,
            'b.x': self.B_DOT_X
        },
                         package_name='my_entry'),
        textwrap.dedent("""\
    package my_entry

    file_number 0 "fake_file.x"

    fn __a__f() -> bits[32] {
      ret literal.1: bits[32] = literal(value=42, id=1, pos=[(0,0,20)])
    }

    fn __b__f() -> bits[32] {
      ret literal.2: bits[32] = literal(value=64, id=2, pos=[(0,0,20)])
    }
    """))


if __name__ == '__main__':
  test_base.main()
