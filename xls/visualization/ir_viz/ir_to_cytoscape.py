#
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

"""Dump all information about an ir into a cytoscape json.

This is equivalent to importing the edge and node csvs from ir_to_csvs into
cytoscape.
"""

from collections.abc import Mapping, Sequence
import json
import subprocess
import sys
from typing import Any

from absl import app
from absl import flags
import networkx as nx

from xls.common import runfiles
from xls.visualization.ir_viz import visualization_pb2

_OUTPUT = flags.DEFINE_string(
    'output',
    default=None,
    help='file to write cytoscape JSON data to.',
    required=True,
)
_DELAY_MODEL = flags.DEFINE_string(
    'delay_model',
    default=None,
    help='delay model to use',
    required=True,
)
_ENTRY_NAME = flags.DEFINE_string(
    'entry_name',
    default=None,
    help='top',
    required=False,
)
_PIPELINE_STAGES = flags.DEFINE_integer(
    'pipeline_stages',
    default=None,
    help='pipeline stages to use',
    required=False,
)

XLS_IR_TO_PROTO = runfiles.get_path('xls/visualization/ir_viz/ir_to_proto_main')


def edge_attrs(edge: visualization_pb2.Edge) -> Mapping[str, Any]:
  """Converts an Edge proto to a dictionary of attributes for cytoscape.

  Args:
    edge: The Edge proto to convert.

  Returns:
    A dictionary of attributes for cytoscape.
  """
  return {
      'key': edge.id,
      'on_critical_path': edge.on_critical_path,
      'bit_width': edge.bit_width,
      'type': edge.type,
  }


def node_attrs(node: visualization_pb2.Node) -> Mapping[str, Any]:
  """Converts a Node proto to a dictionary of attributes for cytoscape.

  Args:
    node: The Node proto to convert.

  Returns:
    A dictionary of attributes for cytoscape.
  """
  res = {
      'label': node.name,
      'ir': node.ir,
      'opcode': node.opcode,
      'all_locs': '\n'.join(
          f'{loc.file.strip()}:{loc.line}' for loc in node.loc
      ),
  }
  forced = {'area_um', 'value', 'known_bits', 'initial_value'}
  for f in visualization_pb2.NodeAttributes.DESCRIPTOR.fields:
    if node.attributes.HasField(f.name) or f.name in forced:
      res[f.name] = getattr(node.attributes, f.name)
  return res


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError('Too many command-line arguments.')

  pipeline_args = []
  if _PIPELINE_STAGES.value is not None:
    pipeline_args.append(f'--pipeline_stages={_PIPELINE_STAGES.value}')
  entry_args = []
  if _ENTRY_NAME.value is not None:
    entry_args.append(f'--entry_name={_ENTRY_NAME.value}')
  run_result = subprocess.run(
      [
          XLS_IR_TO_PROTO,
          argv[1],
          f'--delay_model={_DELAY_MODEL.value}',
          '--binary_format',
      ]
      + pipeline_args
      + entry_args,
      check=False,
      capture_output=True,
  )
  if run_result.returncode != 0:
    print(f'stderr: {run_result.stderr}', file=sys.stderr)
    run_result.check_returncode()
  proto = visualization_pb2.Package.FromString(run_result.stdout)

  graph = nx.MultiDiGraph()
  for fb in proto.function_bases:
    for node in fb.nodes:
      graph.add_node(
          node.id, function=fb.name, function_kind=fb.kind, **node_attrs(node)
      )
    for edge in fb.edges:
      graph.add_edge(edge.source_id, edge.target_id, **edge_attrs(edge))

  with open(_OUTPUT.value, 'w') as out:
    out.write(json.dumps(nx.readwrite.json_graph.cytoscape_data(graph)))


if __name__ == '__main__':
  app.run(main)
