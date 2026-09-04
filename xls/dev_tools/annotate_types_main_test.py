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

"""Tests for annotate_types_main binary."""

import subprocess as subp
import textwrap

from absl.testing import absltest
from xls.common import runfiles

_ANNOTATE_TYPES_MAIN_PATH = runfiles.get_path(
    'xls/dev_tools/annotate_types_main'
)


class AnnotateTypesMainTest(absltest.TestCase):

  def test_basic_stdout(self):
    contents = textwrap.dedent("""\
    fn f() -> u32 {
        let x = u32:42;
        x
    }
    """)
    want = textwrap.dedent("""\
    fn f() -> u32 {
        let x: u32 = u32:42;
        x
    }
    """)
    f = self.create_tempfile(content=contents)
    output = subp.check_output(
        [_ANNOTATE_TYPES_MAIN_PATH, f.full_path], encoding='utf-8'
    )
    self.assertEqual(output, want)

  def test_in_place_flag(self):
    contents = textwrap.dedent("""\
    fn f() -> u32 {
        let x = u32:42;
        x
    }
    """)
    want = textwrap.dedent("""\
    fn f() -> u32 {
        let x: u32 = u32:42;
        x
    }
    """)
    f = self.create_tempfile(content=contents)
    subp.check_call([_ANNOTATE_TYPES_MAIN_PATH, '--in_place', f.full_path])
    self.assertEqual(f.read_text(), want)

  def test_short_in_place_flag(self):
    contents = textwrap.dedent("""\
    fn f() -> u32 {
        let x = u32:42;
        x
    }
    """)
    want = textwrap.dedent("""\
    fn f() -> u32 {
        let x: u32 = u32:42;
        x
    }
    """)
    f = self.create_tempfile(content=contents)
    subp.check_call([_ANNOTATE_TYPES_MAIN_PATH, '-i', f.full_path])
    self.assertEqual(f.read_text(), want)

  def test_output_file_flag(self):
    contents = textwrap.dedent("""\
    fn f() -> u32 {
        let x = u32:42;
        x
    }
    """)
    want = textwrap.dedent("""\
    fn f() -> u32 {
        let x: u32 = u32:42;
        x
    }
    """)
    in_file = self.create_tempfile(content=contents)
    out_file = self.create_tempfile()
    subp.check_call([
        _ANNOTATE_TYPES_MAIN_PATH,
        f'--output_file={out_file.full_path}',
        in_file.full_path,
    ])
    self.assertEqual(out_file.read_text(), want)

  def test_stdin_input(self):
    contents = textwrap.dedent("""\
    fn f() -> u32 {
        let x = u32:42;
        x
    }
    """)
    want = textwrap.dedent("""\
    fn f() -> u32 {
        let x: u32 = u32:42;
        x
    }
    """)
    p = subp.run(
        [_ANNOTATE_TYPES_MAIN_PATH, '-'],
        input=contents,
        encoding='utf-8',
        stdout=subp.PIPE,
        stderr=subp.PIPE,
        check=True,
    )
    self.assertEqual(p.stdout, want)

  def test_error_in_place_on_stdin(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output(
          [_ANNOTATE_TYPES_MAIN_PATH, '--in_place', '-'],
          encoding='utf-8',
          stderr=subp.PIPE,
      )
    self.assertIn(
        'Cannot annotate stdin with in-place annotation',
        str(e.exception.stderr),
    )

  def test_error_in_place_and_output_file(self):
    f = self.create_tempfile(content='fn f() -> u32 { u32:42 }')
    out = self.create_tempfile()
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output(
          [
              _ANNOTATE_TYPES_MAIN_PATH,
              '--in_place',
              f'--output_file={out.full_path}',
              f.full_path,
          ],
          encoding='utf-8',
          stderr=subp.PIPE,
      )
    self.assertIn(
        'Cannot specify both --in_place and --output_file',
        str(e.exception.stderr),
    )

  def test_error_multiple_files_not_in_place(self):
    f1 = self.create_tempfile(content='fn f() -> u32 { u32:42 }')
    f2 = self.create_tempfile(content='fn g() -> u32 { u32:42 }')
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output(
          [_ANNOTATE_TYPES_MAIN_PATH, f1.full_path, f2.full_path],
          encoding='utf-8',
          stderr=subp.PIPE,
      )
    self.assertIn(
        'Cannot have multiple input files when --in_place is not specified',
        str(e.exception.stderr),
    )

  def test_multi_file_in_place(self):
    f1 = self.create_tempfile(content='fn f() -> u32 { let a = u32:1; a }\n')
    f2 = self.create_tempfile(content='fn g() -> u32 { let b = u32:2; b }\n')
    subp.check_call(
        [_ANNOTATE_TYPES_MAIN_PATH, '--in_place', f1.full_path, f2.full_path]
    )
    self.assertEqual(
        f1.read_text(), 'fn f() -> u32 { let a: u32 = u32:1; a }\n'
    )
    self.assertEqual(
        f2.read_text(), 'fn g() -> u32 { let b: u32 = u32:2; b }\n'
    )


if __name__ == '__main__':
  absltest.main()
