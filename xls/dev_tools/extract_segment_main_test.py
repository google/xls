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
import textwrap
from typing import Iterable, Optional, Union

from absl.testing import absltest
from xls.common import runfiles

_EXTRACT_SEGMENT_MAIN = runfiles.get_path('xls/dev_tools/extract_segment_main')

_PROC_IR = """package foobar

top proc NextNodesAsTuples(a: bits[32], b: bits[32], c: bits[32], d: bits[32], init={0, 0, 0, 0}) {
  node0: bits[32] = add(a, b)
  node1: bits[1] = eq(b, c)
  node2: bits[32] = add(c, d)
  node3: bits[32] = add(a, d)
  node4: bits[32] = add(b, c)
  next0: () = next_value(param=a, value=node0, predicate=node1)
  next1: () = next_value(param=b, value=node2)
  next2: () = next_value(param=c, value=node3)
  next3: () = next_value(param=d, value=node4)
}
"""


class ExtractSegmentMainTest(absltest.TestCase):

  def _do_extract(
      self,
      ir: str,
      *,
      sources: Optional[Iterable[Union[str, int]]] = None,
      sinks: Optional[Iterable[Union[str, int]]] = None,
  ) -> str:
    """Runs extract_segment_main with the given IR and args.

    Args:
      ir: The IR to extract from.
      sources: The source nodes to extract.
      sinks: The sink nodes to extract.

    Returns:
      The output of extract_segment_main.
    """
    ir_file = self.create_tempfile(content=ir)
    args = [_EXTRACT_SEGMENT_MAIN]
    if sinks:
      args.append(f"--sink_nodes={','.join(str(s) for s in sinks)}")
    if sources:
      args.append(f"--source_nodes={','.join(str(s) for s in sources)}")
    args.append(ir_file.full_path)

    res = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if res.returncode != 0:
      raise AssertionError(textwrap.dedent(f"""Failed to run {args}.
                           stdout:
                           {res.stdout.decode()}
                           stderr:
                           {res.stderr.decode()}
                           """))
    return res.stdout.decode()

  def test_source_by_name(self):
    result = self._do_extract(_PROC_IR, sources=['d'])
    self.assertNotIn('node0', result)
    self.assertNotIn('node1', result)
    self.assertIn('node2', result)
    self.assertIn('node3', result)
    self.assertIn('node4', result)
    self.assertNotIn('next0', result)
    self.assertIn('next1', result)
    self.assertIn('next2', result)
    self.assertIn('next3', result)

  def test_source_by_id(self):
    result = self._do_extract(_PROC_IR, sources=['d'])
    self.assertNotIn('node0', result)
    self.assertNotIn('node1', result)
    self.assertIn('node2', result)
    self.assertIn('node3', result)
    self.assertIn('node4', result)
    self.assertNotIn('next0', result)
    self.assertIn('next1', result)
    self.assertIn('next2', result)
    self.assertIn('next3', result)

  def test_sinks_by_name(self):
    result = self._do_extract(_PROC_IR, sinks=['next0', 'node1'])
    self.assertIn('node0', result)
    self.assertIn('node1', result)
    self.assertNotIn('node2', result)
    self.assertNotIn('node3', result)
    self.assertNotIn('node4', result)
    self.assertIn('next0', result)
    self.assertNotIn('next1', result)
    self.assertNotIn('next2', result)
    self.assertNotIn('next3', result)


if __name__ == '__main__':
  absltest.main()
