#
# Copyright 2023 The XLS Authors
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

"""The main routine for the python-side of the ECO flow."""

import os
import pathlib
import sys
from typing import Sequence

from absl import app
from absl import flags

from xls.eco import ir_diff
from xls.eco import ir_diff_utils
from xls.eco import ir_patch_gen
from xls.eco import xls_ir_to_networkx


_BEFORE_IR_PATH = flags.DEFINE_string(
    "before_ir",
    None,
    "Path to the 'before' IR file.",
    required=True,
)
_AFTER_IR_PATH = flags.DEFINE_string(
    "after_ir",
    None,
    "Path to the 'after' IR file.",
    required=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    "o",
    None,
    "Path to the output patch file.",
    required=False,
)

_TIMEOUT = flags.DEFINE_integer(
    "t",
    None,
    "Timeout limit (in seconds) for finding optimized edit paths, if not set, "
    "the script defaults to finding the optimal edit paths.",
    required=False,
)

_RECURSION_LIMIT = flags.DEFINE_integer(
    "r",
    None,
    "Recursion limit for the IR diff algorithm. If not set, the script defaults"
    " to system recursion limit.",
    required=False,
)


def main(argv: Sequence[str]) -> int:
  del argv  # Unused.

  if _RECURSION_LIMIT.value is not None:
    sys.setrecursionlimit(_RECURSION_LIMIT.value)

  # Parse IR files
  before_ir_graph = xls_ir_to_networkx.read_xls_ir_to_networkx(
      pathlib.Path(_BEFORE_IR_PATH.value)
  )
  after_ir_graph = xls_ir_to_networkx.read_xls_ir_to_networkx(
      pathlib.Path(_AFTER_IR_PATH.value)
  )
  # Compute IR diff
  # Initialize edit_paths to None to avoid potential UnboundLocalError.
  edit_paths = None
  if _TIMEOUT.value is None:
    edit_paths = ir_diff.find_optimal_edit_paths(
        before_ir_graph, after_ir_graph
    )
    print(f"Found the optimal edit paths:\t path cost: {edit_paths.cost}")
  else:
    costs = []
    durations = []
    for i, edit_paths in enumerate(
        ir_diff.find_optimized_edit_paths(
            before_ir_graph, after_ir_graph, _TIMEOUT.value
        )
    ):
      if not edit_paths:
        continue
      costs.append(edit_paths.cost)
      durations.append(edit_paths.duration)
      print(
          f"Found {i+1} edit paths\tpath cost: {edit_paths.cost}\telapsed time:"
          f" {edit_paths.duration:.2f}s"
      )
      if _OUTPUT_PATH.value:
        ir_diff_utils.plot_optimized_edit_paths_cost_vs_time(
            costs,
            durations,
            output_path=os.path.join(
                _OUTPUT_PATH.value, "optimized_edit_paths_benchmark.png"
            ),
        )
  if edit_paths is None:
    raise ValueError(
        "edit paths was not set, find_optimized_edit_paths timed out before"
        " producing a result."
    )
  interpreted_edit_paths_path = None
  if _OUTPUT_PATH.value:
    interpreted_edit_paths_path = os.path.join(
        _OUTPUT_PATH.value, "interpreted_edit_paths.txt"
    )
  ir_diff_utils.interpret_edit_paths(
      edit_paths=edit_paths, output_path=interpreted_edit_paths_path
  )
  if _OUTPUT_PATH.value:
    patch = ir_patch_gen.IrPatch(
        edit_paths,
        before_ir_graph,
        after_ir_graph,
        os.path.join(_OUTPUT_PATH.value, "patch.bin"),
    )
    patch.write_proto()
  return 0


if __name__ == "__main__":
  app.run(main)
