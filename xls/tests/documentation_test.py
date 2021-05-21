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

import re
import subprocess as subp
from typing import List, Dict, Union

from xls.common import runfiles
from absl.testing import absltest
from absl.testing import parameterized

# Implementation note: from inspecting pymarkdown it appears that scraping for
# DSL code blocks via this regex has the same power as using the Markdown parser
# directly, since these fence constructs can only appear at the top level.
_DSLX_RE = re.compile('^```dslx$(.*?)^```$', re.MULTILINE | re.DOTALL)

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')


def get_examples() -> List[Dict[str, Union[int, str]]]:
  """Returns DSLX blocks in the reference Markdown as dictionary records."""
  contents = runfiles.get_contents_as_text('docs_src/dslx_reference.md')
  examples = []
  for i, m in enumerate(_DSLX_RE.finditer(contents)):
    dslx_block = m.group(1)
    examples.append(
        dict(testcase_name=f'dslx_block_{i}', i=i, dslx_block=dslx_block))
  return examples


class DocumentationTest(parameterized.TestCase):

  @parameterized.named_parameters(get_examples())
  def test_dslx_blocks(self, i: int, dslx_block: str) -> None:
    """Runs the given DSLX block as a DSLX test file."""
    x_file = self.create_tempfile(
        file_path=f'doctest_{i}.x', content=dslx_block)
    p = subp.run([_INTERP_PATH, x_file.full_path],
                 check=False,
                 stderr=subp.PIPE,
                 encoding='utf-8')
    if p.returncode != 0:
      print(p.stderr)
    self.assertEqual(p.returncode, 0)


if __name__ == '__main__':
  absltest.main()
