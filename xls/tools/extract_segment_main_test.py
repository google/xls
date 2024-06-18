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

_EXTRACT_SEGMENT_MAIN = runfiles.get_path("xls/tools/extract_segment_main")

_PROC_IR = """package foobar

top proc NextNodesAsTuples(a: bits[32], b: bits[32], c: bits[32], d: bits[32], init={0, 0, 0, 0}) {
  add.5: bits[32] = add(a, b, id=5)
  eq.6: bits[1] = eq(b, c, id=6)
  add.8: bits[32] = add(c, d, id=8)
  add.10: bits[32] = add(a, d, id=10)
  add.12: bits[32] = add(b, c, id=12)
  next_value.7: () = next_value(param=a, value=add.5, predicate=eq.6, id=7)
  next_value.9: () = next_value(param=b, value=add.8, id=9)
  next_value.11: () = next_value(param=c, value=add.10, id=11)
  next_value.13: () = next_value(param=d, value=add.12, id=13)
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
    self.assertEqual(
        self._do_extract(_PROC_IR, sources=["d"]),
        """package extracted_package

top fn extracted_func(param_for_a_id1: bits[32], param_for_b_id2: bits[32], param_for_c_id3: bits[32], d: bits[32], param_for_add_12_id12: bits[32]) -> ((bits[32], bits[32]), (bits[32], bits[32]), (bits[32], bits[32])) {
  add.5: bits[32] = add(param_for_c_id3, d, id=5)
  add.6: bits[32] = add(param_for_a_id1, d, id=6)
  next_value_9: (bits[32], bits[32]) = tuple(param_for_b_id2, add.5, id=8)
  next_value_11: (bits[32], bits[32]) = tuple(param_for_c_id3, add.6, id=9)
  next_value_13: (bits[32], bits[32]) = tuple(d, param_for_add_12_id12, id=10)
  ret tuple.11: ((bits[32], bits[32]), (bits[32], bits[32]), (bits[32], bits[32])) = tuple(next_value_9, next_value_11, next_value_13, id=11)
}
""",
    )

  def test_source_by_id(self):
    self.assertEqual(
        self._do_extract(_PROC_IR, sources=[5]),
        """package extracted_package

top fn extracted_func(param_for_a_id1: bits[32], param_for_b_id2: bits[32], param_for_eq_6_id6: bits[1]) -> (bits[32], bits[32], bits[1]) {
  add.3: bits[32] = add(param_for_a_id1, param_for_b_id2, id=3)
  ret next_value_7: (bits[32], bits[32], bits[1]) = tuple(param_for_a_id1, add.3, param_for_eq_6_id6, id=5)
}
""",
    )

  def test_sinks_by_name(self):
    self.assertEqual(
        self._do_extract(_PROC_IR, sinks=["next_value.7", "eq.6"]),
        """package extracted_package

top fn extracted_func(a: bits[32], b: bits[32], c: bits[32]) -> ((bits[32], bits[32], bits[1]), bits[1]) {
  add.4: bits[32] = add(a, b, id=4)
  eq.5: bits[1] = eq(b, c, id=5)
  next_value_7: (bits[32], bits[32], bits[1]) = tuple(a, add.4, eq.5, id=6)
  ret tuple.7: ((bits[32], bits[32], bits[1]), bits[1]) = tuple(next_value_7, eq.5, id=7)
}
""",
    )


if __name__ == "__main__":
  absltest.main()
