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
import struct

from xls.eco import ir_patch_pb2
from xls.eco import xls_types
from xls.ir import xls_type_pb2
from xls.ir import xls_value_pb2


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

  def __init__(self, graph_edit_paths, graph0, graph1, export_path):
    """Initializes an IrPatch object.

    Args:
      graph_edit_paths: A tuple containing node and edge edit paths. It can be
        either optimized or optimal edit paths.
      graph0: The golden NetworkX graph.
      graph1: The revised NetworkX graph.
      export_path: The base path for writing serialized IrEditPaths messages to
        files.
    """

    self.ir_edit_paths_pb = ir_patch_pb2.IrPatchProto()
    self.serialized_proto = None
    self.export_path = export_path
    self.id = 0
    edit_paths = graph_edit_paths
    if isinstance(graph_edit_paths[0][0][0], list):
      edit_paths = graph_edit_paths[0][0]  # conpensate for optimal edit paths
    for node_edit_path in edit_paths[0]:
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
            self._export_pb_node_attributes(
                return_node, graph1.nodes[node_edit_path[1]]
            )
    for edge_edit_path in edit_paths[1]:
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

    This function handles common node attributes and converts data types (using
    `self._node_dtype_to_proto`) when necessary. It also sets operation-specific
    unique arguments based on the `nx_node['op']` value.

    Args:
      pb_node: The target protocol buffer node message to populate.
      nx_node: The source NetworkX node dictionary containing attributes.

    Raises:
      Warning: If the `nx_node` lacks required operand data types.
    """
    if 'id' in nx_node and nx_node['id'] is not None:
      pb_node.id = nx_node['id']
    pb_node.data_type.CopyFrom(
        self._node_dtype_to_proto(pb_node, nx_node['data_type'])
    )
    pb_node.op = nx_node['op']
    if 'operand_data_type' in nx_node:
      pb_node.operand_data_types.append(
          self._node_dtype_to_proto(pb_node, nx_node['operand_data_type'])
      )
    elif 'operand_data_types' in nx_node:
      for operand_data_type in nx_node['operand_data_types']:
        pb_node.operand_data_types.append(
            self._node_dtype_to_proto(pb_node, operand_data_type)
        )
    if nx_node['op'] == 'literal':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.value.CopyFrom(self._node_value_to_proto(nx_node['value']))
    elif nx_node['op'] == 'bit_slice':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.start = nx_node['start']
    elif nx_node['op'] == 'tuple_index':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.index = nx_node['index']
    elif nx_node['op'] == 'one_hot':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.lsb_prio = nx_node['lsb_prio']
    elif nx_node['op'] == 'sign_ext':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.new_bit_count = nx_node['new_bit_count']
    elif nx_node['op'] == 'sel':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.has_default_value = nx_node['has_default_value']
    elif nx_node['op'] == 'receive':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = nx_node['channel']
      if nx_node['blocking'] is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.blocking = nx_node['blocking']
    elif nx_node['op'] == 'send':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = nx_node['channel']

  def _node_dtype_to_proto(self, pb_node, dtype):
    """Converts a `DataType` object to an `xls.TypeProto` message.

    Args:
      pb_node: The `ir_patch_pb2.Node` message to fill.
      dtype: The `DataType` object to convert.

    Returns:
      An `xls.TypeProto` message filled with the corresponding data.
    """

    type_pb = xls_type_pb2.TypeProto()
    if isinstance(dtype, xls_types.TokenType):
      type_pb.type_enum = xls_type_pb2.TypeProto.TypeEnum.TOKEN
    elif isinstance(dtype, xls_types.TupleType):
      type_pb.type_enum = xls_type_pb2.TypeProto.TypeEnum.TUPLE
      type_pb.tuple_elements.extend(
          self._node_dtype_to_proto(pb_node, t) for t in dtype.tuple_elements
      )
    elif isinstance(dtype, xls_types.ArrayType):
      type_pb.type_enum = xls_type_pb2.TypeProto.TypeEnum.ARRAY
      type_pb.array_size = dtype.array_size
      if dtype.array_element is not None:
        type_pb.array_element.MergeFrom(
            self._node_dtype_to_proto(pb_node, dtype.array_element)
        )
    elif isinstance(dtype, xls_types.BitsType):
      type_pb.type_enum = xls_type_pb2.TypeProto.TypeEnum.BITS
      type_pb.bit_count = (
          dtype.bit_count or 0
      )  # Set default for missing bit_count
    else:
      raise ValueError(f'Unsupported data type: {dtype}')

    return type_pb

  def _integer_to_bytes(self, integer, bit_count):
    """Converts an integer to bytes data as specified in the proto message.

    Args:
        integer: The integer to convert.
        bit_count: The number of bits in the integer.

    Returns:
        The bytes data representing the integer.
    """

    # Ensure the bit count is valid
    if bit_count <= 0:
      raise ValueError('Bit count must be positive')

    # Calculate the number of bytes required
    byte_count = (bit_count + 7) // 8

    # Convert the integer to bytes in little-endian order
    bytes_data = struct.pack(
        '<{}B'.format(byte_count),
        *(integer >> (8 * i) & 0xFF for i in range(byte_count)),
    )

    return bytes_data

  def _node_value_to_proto(self, value):
    """Converts a `Value` object to an `xls.ValueProto` message.

    Args:
      value: The `Value` object to convert.

    Returns:
      An `xls.ValueProto` message filled with the corresponding data.
    """
    pb = xls_value_pb2.ValueProto()
    if isinstance(value, xls_types.TokenValue):
      pb.token = xls_value_pb2.ValueProto.Token()
    elif isinstance(value, xls_types.BitsValue):

      pb.bits.CopyFrom(
          xls_value_pb2.ValueProto.Bits(
              bit_count=value.bit_count,
              data=self._integer_to_bytes(value.data, value.bit_count),
          )
      )
    elif isinstance(value, xls_types.TupleValue):
      pb.tuple.CopyFrom(
          xls_value_pb2.ValueProto.Tuple(
              elements=[self._node_value_to_proto(e) for e in value.elements]
          )
      )
    elif isinstance(value, xls_types.ArrayValue):
      pb.array.CopyFrom(
          xls_value_pb2.ValueProto.Array(
              elements=[self._node_value_to_proto(e) for e in value.elements]
          )
      )
    else:
      raise ValueError(f'Unsupported value type: {value}')
    return pb

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
