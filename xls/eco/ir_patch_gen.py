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

"""A library for constructing proto-based patches from IR edit paths."""

import os

from xls.eco import ir_diff
from xls.eco import ir_patch_pb2


class IrPatch:
  """Represents an IR edit patch for converting between two IR graphs.

  This class processes edit paths between two NetworkX graphs and constructs a
  corresponding protocol buffer message for exporting the edits.

  Attributes:
    ir_edit_paths_pb: A message representing the edit patch.
    serialized_proto: A byte string containing the serialized IrEditPaths
      message.
    export_path: The base path for writing serialized IrEditPaths messages to
      files.
    id: A unique identifier for each edit path.
  """

  def __init__(
      self, edit_paths: ir_diff.OptimizedEditPaths, graph0, graph1, export_path
  ):
    """Initializes an IrPatch object.

    Args:
      edit_paths: An `OptimizedEditPaths` object containing node and edge edit
        paths.
      graph0: The golden NetworkX graph.
      graph1: The revised NetworkX graph.
      export_path: The base path for writing serialized IrEditPaths messages to
        files.
    """

    self.ir_edit_paths_pb = ir_patch_pb2.IrPatchProto()
    self.serialized_proto = None
    self.export_path = export_path
    self.id = 0
    for node_edit_path in edit_paths.node_edit_paths:
      if node_edit_path[0] != node_edit_path[1]:
        pb_edit_path = self.ir_edit_paths_pb.edit_paths.add()
        pb_edit_path.id = self.id
        self.id += 1
        pb_node_edit_path = pb_edit_path.node_edit_path
        if node_edit_path[0] is not None and node_edit_path[1] is None:
          pb_edit_path.operation = ir_patch_pb2.Operation.DELETE
          pb_node_edit_path.node.name = node_edit_path[0]
          self._export_pb_node_attributes(
              pb_node_edit_path.node, graph0.nodes[node_edit_path[0]]
          )
        elif node_edit_path[0] is None and node_edit_path[1] is not None:
          pb_edit_path.operation = ir_patch_pb2.Operation.INSERT
          pb_node_edit_path.node.name = node_edit_path[1]
          self._export_pb_node_attributes(
              pb_node_edit_path.node, graph1.nodes[node_edit_path[1]]
          )
          if 'ret' in graph1.graph and graph1.graph['ret'] == node_edit_path[1]:
            return_node = self.ir_edit_paths_pb.return_node
            return_node.name = node_edit_path[1]
            self._export_pb_node_attributes(
                return_node, graph1.nodes[node_edit_path[1]]
            )
        else:
          pb_edit_path.operation = ir_patch_pb2.Operation.UPDATE
          pb_node_edit_path.node.name, pb_node_edit_path.updated_node.name = (
              node_edit_path[0],
              node_edit_path[1],
          )
          self._export_pb_node_attributes(
              pb_node_edit_path.node, graph0.nodes[node_edit_path[0]]
          )
          self._export_pb_node_attributes(
              pb_node_edit_path.updated_node, graph1.nodes[node_edit_path[1]]
          )
          if 'ret' in graph1.graph and graph1.graph['ret'] == node_edit_path[1]:
            return_node = self.ir_edit_paths_pb.return_node
            return_node.name = node_edit_path[1]
            self._export_pb_node_attributes(
                return_node, graph1.nodes[node_edit_path[1]]
            )
    for edge_edit_path in edit_paths.edge_edit_paths:
      if edge_edit_path[0] != edge_edit_path[1]:
        pb_edit_path = self.ir_edit_paths_pb.edit_paths.add()
        pb_edit_path.id = self.id
        self.id += 1
        pb_edge_edit_path = pb_edit_path.edge_edit_path
        if (
            edge_edit_path[0] is None and edge_edit_path[1] is not None
        ):  # Insert edge
          pb_edit_path.operation = ir_patch_pb2.Operation.INSERT
          pb_edge_edit_path.edge.from_node = edge_edit_path[1][0]
          pb_edge_edit_path.edge.to_node = edge_edit_path[1][1]
          pb_edge_edit_path.edge.index = edge_edit_path[1][2]
        elif (
            edge_edit_path[1] is None and edge_edit_path[0] is not None
        ):  # Delete edge
          pb_edit_path.operation = ir_patch_pb2.Operation.DELETE
          pb_edge_edit_path.edge.from_node = edge_edit_path[0][0]
          pb_edge_edit_path.edge.to_node = edge_edit_path[0][1]
          pb_edge_edit_path.edge.index = edge_edit_path[0][2]
        elif (
            edge_edit_path[0] is not None
            and edge_edit_path[1] is not None
            and edge_edit_path[0] != edge_edit_path[1]
        ):  # Update edge
          pb_edit_path.operation = ir_patch_pb2.Operation.UPDATE
          pb_edge_edit_path.edge.from_node = edge_edit_path[0][0]
          pb_edge_edit_path.edge.to_node = edge_edit_path[0][1]
          pb_edge_edit_path.edge.index = edge_edit_path[0][2]
          pb_edge_edit_path.updated_edge.from_node = edge_edit_path[1][0]
          pb_edge_edit_path.updated_edge.to_node = edge_edit_path[1][1]
          pb_edge_edit_path.updated_edge.index = edge_edit_path[1][2]
    self.serialized_proto = self.ir_edit_paths_pb.SerializeToString()

  def _export_pb_node_attributes(self, pb_node, nx_node):
    """Populates a protocol buffer node message (`pb_node`) with attributes from a NetworkX node (`nx_node`).

    This function handles common node attributes and converts data types to
    proto messages when necessary. It also sets operation-specific unique
    arguments based on the `nx_node['op']` value.

    Args:
      pb_node: The target protocol buffer node message to populate.
      nx_node: The source NetworkX node dictionary containing attributes.

    Raises:
      Warning: If the `nx_node` lacks required operand data types.
    """
    if 'id' in nx_node and nx_node['id'] is not None:
      pb_node.id = nx_node['id']
    data_type = nx_node['data_type']
    pb_node.data_type.CopyFrom(data_type.to_proto())
    pb_node.op = nx_node['op']
    # node_attributes = (nx_node['cost_attributes']['node_attributes']]
    # if 'node_attributes' in nx_node['cost_attributes']else {})
    if 'operand_data_type' in nx_node:
      operand_data_type = nx_node['operand_data_type']
      pb_node.operand_data_types.append(operand_data_type.to_proto())
    elif 'operand_data_types' in nx_node:
      for operand_data_type in nx_node['operand_data_types']:
        pb_node.operand_data_types.append(operand_data_type.to_proto())
    if nx_node['op'] == 'literal':
      pb_unique_args = pb_node.unique_args.add()
      node_value = nx_node['value']
      pb_unique_args.value.CopyFrom(node_value.to_proto(data_type))
    elif nx_node['op'] == 'bit_slice':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.start = nx_node['start']
    elif nx_node['op'] == 'tuple_index':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.index = nx_node['index']
    elif nx_node['op'] == 'array_index' or nx_node['op'] == 'array_update':
      if nx_node['assumed_in_bounds'] is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.assumed_in_bounds = nx_node['assumed_in_bounds']
    elif nx_node['op'] == 'one_hot':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.lsb_prio = nx_node['lsb_prio']
    elif nx_node['op'] == 'sign_ext':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.new_bit_count = nx_node['new_bit_count']
    elif nx_node['op'] == 'sel' or nx_node['op'] == 'priority_sel':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.has_default_value = nx_node['has_default_value']
    elif nx_node['op'] == 'receive':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = nx_node['channel']
      if 'blocking' in nx_node:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.blocking = nx_node['blocking']
    elif nx_node['op'] == 'send':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = nx_node['channel']
    elif nx_node['op'] == 'state_read':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.index = nx_node['index']
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.state_element = nx_node['state_element']
      pb_unique_args = pb_node.unique_args.add()
      node_init_value = nx_node['init']
      pb_unique_args.init.CopyFrom(node_init_value.to_proto(data_type))

  def write_proto(self) -> None:
    """Writes the serialized protocol buffer message to a specified file.

    This function serializes the internal protocol buffer message and writes it
    to the specified binary file.
    """
    # Extract the directory path from the export path
    directory_path = os.path.dirname(self.export_path)

    if not os.path.exists(directory_path):
      os.makedirs(directory_path)

    with open(self.export_path, 'wb') as f:
      f.write(self.serialized_proto)
    print(
        f'Wrote proto: {self.export_path}\t file size:'
        f' {os.path.getsize(self.export_path)} bytes'
    )
