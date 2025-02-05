# Copyright 2025 The XLS Authors
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

"""Read XLS IR into a networkx graph.

Uses the xls_ir_to_cytoscape utility to read the IR into a JSON format and then
parses that into a networkx graph.
"""

import json
import pathlib
import subprocess

from absl import logging
import networkx as nx

from xls.common import runfiles


XLS_IR_TO_CYTOSCAPE_PATH = runfiles.get_path(
    "xls/eco/xls_ir_to_cytoscape"
)


def read_xls_ir_to_networkx(ir_path: pathlib.Path) -> nx.MultiDiGraph:
  """Reads XLS IR into a networkx graph."""
  json_text = subprocess.check_output([XLS_IR_TO_CYTOSCAPE_PATH, ir_path])
  ir_json = json.loads(json_text)
  logging.vlog(3, "Loaded JSON: %s", ir_json)
  return nx.cytoscape_graph(ir_json)
