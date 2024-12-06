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
import sys
from typing import Sequence

from absl import app
from absl import flags

from xls.eco import ir2nx
from xls.eco import ir_diff
from xls.eco import ir_diff_utils
from xls.eco import ir_patch_gen


# Define flags using absl
_IR_INPUT_PATHS = flags.DEFINE_multi_string(
    "i",
    None,
    "Path to the IR files to diff. --i first_ir --i second_ir",
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

_GENERATE_PATCH = flags.DEFINE_bool(
    "p",
    False,
    "Whether to generate a patch file from the optimized edit paths.",
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

  # Extract IR paths from input flag (assuming space-separated)
  if len(_IR_INPUT_PATHS.value) != 2:
    raise ValueError(
        "Expected exactly two IR files, got {}.".format(
            len(_IR_INPUT_PATHS.value)
        )
    )
  if _GENERATE_PATCH.value and _OUTPUT_PATH.value is None:
    raise ValueError(
        "Output path must be specified if patch generation is requested."
    )

  first_ir_path, second_ir_path = _IR_INPUT_PATHS.value
  # Parse IR files
  first_ir = ir2nx.IrParser(first_ir_path)
  second_ir = ir2nx.IrParser(second_ir_path)
  if _RECURSION_LIMIT.value is not None:
    sys.setrecursionlimit(_RECURSION_LIMIT.value)
  # Compute IR diff
  graph_diff = ir_diff.IrDiff(first_ir.graph, second_ir.graph)
  if _TIMEOUT.value is None:
    graph_diff.find_optimal_edit_paths()
    print(
        f"Found the optimal edit paths:\t path cost: {graph_diff.path_costs[0]}"
    )
    ir_diff_utils.interpret_edit_paths(
        edit_paths=graph_diff.optimal_edit_paths,
    )
    if _OUTPUT_PATH.value:
      ir_diff_utils.interpret_edit_paths(
          edit_paths=graph_diff.optimal_edit_paths,
          output_path=os.path.join(
              _OUTPUT_PATH.value, "interpreted_edit_paths.txt"
          ),
      )
      if _GENERATE_PATCH.value:
        patch = ir_patch_gen.IrPatch(
            graph_diff.optimal_edit_paths,
            graph_diff.graph0,
            graph_diff.graph1,
            os.path.join(_OUTPUT_PATH.value, "patch.bin"),
        )
        patch.write_proto()
  else:
    for i, edit_paths in enumerate(
        graph_diff.find_optimized_edit_paths(_TIMEOUT.value)
    ):
      if not edit_paths:
        continue
      else:
        print(
            f"Found {i+1} edit paths\tcost: {edit_paths[2]}\telapsed time:"
            f" {graph_diff.optimized_timestamps[i]:.2f}s"
        )
        if _OUTPUT_PATH.value:
          ir_diff_utils.plot_optimized_edit_paths_cost_vs_time(
              graph_diff.path_costs,
              graph_diff.optimized_timestamps,
              output_path=os.path.join(
                  _OUTPUT_PATH.value, "optimized_edit_paths_benchmark.png"
              ),
          )
          ir_diff_utils.interpret_edit_paths(
              edit_paths=graph_diff.optimized_edit_paths,
              output_path=os.path.join(
                  _OUTPUT_PATH.value, "interpreted_edit_paths.txt"
              ),
          )
          if _GENERATE_PATCH.value:
            patch = ir_patch_gen.IrPatch(
                edit_paths,
                graph_diff.graph0,
                graph_diff.graph1,
                os.path.join(_OUTPUT_PATH.value, f"patch_{i}.bin"),
            )
            patch.write_proto()
        else:
          ir_diff_utils.interpret_edit_paths(
              edit_paths=graph_diff.optimized_edit_paths
          )
  return 0


if __name__ == "__main__":
  app.run(main)
