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
"""Tests that DSLX blocks in reference documentation are valid."""

import dataclasses
import re
import subprocess as subp
import sys
from typing import List, Dict, Union

from xls.common import runfiles
from absl.testing import absltest
from absl.testing import parameterized

# Implementation note: from inspecting pymarkdown it appears that scraping for
# DSL code blocks via this regex has the same power as using the Markdown parser
# directly, since these fence constructs can only appear at the top level.
_DSLX_RE = re.compile('^```dslx$(.*?)^```$', re.MULTILINE | re.DOTALL)

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')

_INPUT_FILES = [
    'docs_src/dslx_reference.md',
    'docs_src/dslx_type_system.md',
    'docs_src/tutorials/hello_xls.md',
    'docs_src/tutorials/float_to_int.md',
    'docs_src/tutorials/intro_to_parametrics.md',
]

_INTERP_ATTR_RE = re.compile(
    r'#\[interp_main_(?P<key>\S+) = "(?P<value>\S+)"\]')


def get_examples() -> List[Dict[str, Union[int, str]]]:
  """Returns DSLX blocks in the reference Markdown as dictionary records."""
  examples = []
  for input_file in _INPUT_FILES:
    contents = runfiles.get_contents_as_text(input_file)
    f = input_file.replace('/', '_').replace('.', '_')
    for i, m in enumerate(_DSLX_RE.finditer(contents)):
      dslx_block = m.group(1)
      examples.append(
          dict(testcase_name=f'dslx_block_{f}_{i}', i=i, dslx_block=dslx_block))

  return examples


@dataclasses.dataclass
class StrippedExample:
  dslx: str
  flags: List[str]


def strip_attributes(dslx: str) -> StrippedExample:
  flags: List[str] = []
  while True:
    m = _INTERP_ATTR_RE.search(dslx)
    if not m:
      break
    flags.append('--{}={}'.format(m.group('key'), m.group('value')))
    dslx = _INTERP_ATTR_RE.sub('', dslx, count=1)
  return StrippedExample(dslx, flags)


class DocumentationTest(parameterized.TestCase):

  def test_strip_attributes(self):
    text = """\
// Some comment
#[interp_main_flag = "stuff"]

// Another comment
#[interp_main_other_flag = "thing"]
"""
    example = strip_attributes(text)
    self.assertEqual(example.flags, ['--flag=stuff', '--other_flag=thing'])
    self.assertEqual(example.dslx, """\
// Some comment


// Another comment

""")

  @parameterized.named_parameters(get_examples())
  def test_dslx_blocks(self, i: int, dslx_block: str) -> None:
    """Runs the given DSLX block as a DSLX test file."""
    example = strip_attributes(dslx_block)
    x_file = self.create_tempfile(
        file_path=f'doctest_{i}.x', content=example.dslx)
    cmd = [_INTERP_PATH] + example.flags + [x_file.full_path]
    print('Running command:', subp.list2cmdline(cmd), file=sys.stderr)
    p = subp.run(cmd, check=False, stderr=subp.PIPE, encoding='utf-8')
    if p.returncode != 0:
      print('== DSLX block:')
      print(example.dslx)
      print('== stderr:')
      print(p.stderr)
    self.assertEqual(p.returncode, 0)


if __name__ == '__main__':
  absltest.main()
