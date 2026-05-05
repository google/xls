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


# TODO: DEPRECATE - This Python patch generator will be replaced
# by patch_ir.h/cc (C++ implementation) which provides:
# - Better type safety and error handling
# - Direct integration with XLS IR infrastructure
# - More efficient patch application
# Please use //xls/contrib/eco:patch_ir_main instead.


"""A library for constructing proto-based patches from IR edit paths."""

import json
import os

from xls.contrib.eco import ir_diff
from xls.contrib.eco import ir_patch_pb2
from xls.contrib.eco import xls_types
from xls.contrib.eco import xls_values


def _split_top_level(value: str) -> list[str]:
  """Splits a comma-delimited string while preserving nested IR type/value text."""
  parts = []
  current = []
  depth = 0
  for char in value:
    if char in "([":
      depth += 1
    elif char in ")]":
      depth = max(0, depth - 1)
    if char == "," and depth == 0:
      part = "".join(current).strip()
      if part:
        parts.append(part)
      current = []
      continue
    current.append(char)
  part = "".join(current).strip()
  if part:
    parts.append(part)
  return parts


def _parse_cost_attribute_string(cost_attributes: str) -> dict[str, object]:
  """Parses the legacy pipe-delimited cost attribute representation."""
  result = {}
  for token in cost_attributes.split("|"):
    key, sep, value = token.partition("=")
    if sep:
      result[key] = value
  return result


def _merge_node_attributes(attrs: dict[str, object]) -> None:
  """Flattens Cytoscape node_attributes into the main attribute map."""
  node_attributes = attrs.get("node_attributes")
  if node_attributes is None:
    return
  if isinstance(node_attributes, str):
    try:
      node_attributes = json.loads(node_attributes)
    except json.JSONDecodeError:
      return
  if not isinstance(node_attributes, dict):
    return
  for key, value in node_attributes.items():
    attrs.setdefault(key, value)
  if "initial_value" in attrs and "init" not in attrs:
    attrs["init"] = attrs["initial_value"]
  if "state_param_index" in attrs and "index" not in attrs:
    attrs["index"] = attrs["state_param_index"]


def _collect_node_attributes(nx_node) -> dict[str, object]:
  """Collects legacy ir2nx and Cytoscape-backed NetworkX attributes."""
  attrs = {}
  cost_attributes = nx_node.get("cost_attributes", {})
  if isinstance(cost_attributes, dict):
    attrs.update(cost_attributes)
  elif isinstance(cost_attributes, str):
    try:
      parsed = json.loads(cost_attributes)
      if isinstance(parsed, dict):
        attrs.update(parsed)
      else:
        attrs.update(_parse_cost_attribute_string(cost_attributes))
    except json.JSONDecodeError:
      attrs.update(_parse_cost_attribute_string(cost_attributes))
  _merge_node_attributes(attrs)

  # Cost attributes carry operation-specific values. Top-level Cytoscape
  # fields such as "value" may just be display labels, so keep them as fallback.
  for key, value in nx_node.items():
    if key != "cost_attributes":
      attrs.setdefault(key, value)
  return attrs


def _as_int(value) -> int:
  if isinstance(value, bool):
    return int(value)
  if isinstance(value, int):
    return value
  return int(str(value), 0)


def _as_bool(value) -> bool:
  if isinstance(value, bool):
    return value
  if isinstance(value, int):
    return bool(value)
  lowered = str(value).strip().lower()
  if lowered in ("true", "1"):
    return True
  if lowered in ("false", "0"):
    return False
  raise ValueError(f"Cannot parse bool: {value}")


def _as_data_type(value):
  if hasattr(value, "to_proto"):
    return value
  return xls_types.parse_data_type(str(value))


def _strip_typed_value_prefix(value: str) -> str:
  text = value.strip()
  prefix, sep, suffix = text.partition(":")
  if not sep:
    return text
  prefix = prefix.strip()
  if prefix == "token" or prefix.startswith(("bits[", "(", "[")):
    return suffix.strip()
  return text


def _parse_value_for_type(value, data_type):
  if hasattr(value, "to_proto"):
    return value
  if isinstance(data_type, xls_types.TokenType):
    return xls_values.TokenValue()

  text = _strip_typed_value_prefix(str(value))
  if isinstance(data_type, xls_types.BitsType):
    return xls_values.BitsValue(data=_as_int(text))
  if isinstance(data_type, xls_types.ArrayType):
    if not (text.startswith("[") and text.endswith("]")):
      raise ValueError(f"Expected array value, got: {value}")
    return xls_values.ArrayValue(
        elements=[
            _parse_value_for_type(element, data_type.array_element)
            for element in _split_top_level(text[1:-1])
        ]
    )
  if isinstance(data_type, xls_types.TupleType):
    if text == "()":
      return xls_values.TupleValue(elements=[])
    if not (text.startswith("(") and text.endswith(")")):
      raise ValueError(f"Expected tuple value, got: {value}")
    elements = _split_top_level(text[1:-1])
    return xls_values.TupleValue(
        elements=[
            _parse_value_for_type(element, element_type)
            for element, element_type in zip(elements, data_type.tuple_elements)
        ]
    )
  return xls_values.parse_value(text)[1]


def _append_operand_data_type(pb_node, value) -> None:
  pb_node.operand_data_types.append(_as_data_type(value).to_proto())


def _append_operand_data_types(pb_node, value) -> None:
  if isinstance(value, (list, tuple)):
    for operand_data_type in value:
      _append_operand_data_type(pb_node, operand_data_type)
  else:
    for operand_data_type in _split_top_level(str(value)):
      _append_operand_data_type(pb_node, operand_data_type)


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
    attrs = _collect_node_attributes(nx_node)
    if 'id' in attrs and attrs['id'] is not None:
      pb_node.id = _as_int(attrs['id'])
    data_type = _as_data_type(attrs.get('data_type', attrs['dtype_str']))
    pb_node.data_type.CopyFrom(data_type.to_proto())
    pb_node.op = attrs['op']
    if 'operand_data_type' in attrs:
      _append_operand_data_type(pb_node, attrs['operand_data_type'])
    elif 'operand_dtype_str' in attrs:
      _append_operand_data_type(pb_node, attrs['operand_dtype_str'])
    elif 'operand_data_types' in attrs:
      _append_operand_data_types(pb_node, attrs['operand_data_types'])
    elif 'operand_dtype_strs' in attrs:
      _append_operand_data_types(pb_node, attrs['operand_dtype_strs'])
    if attrs['op'] == 'literal':
      pb_unique_args = pb_node.unique_args.add()
      node_value = _parse_value_for_type(
          attrs.get('value_str', attrs['value']), data_type
      )
      pb_unique_args.value.CopyFrom(node_value.to_proto(data_type))
    elif attrs['op'] == 'bit_slice':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.start = _as_int(attrs['start'])
    elif attrs['op'] == 'tuple_index':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.index = _as_int(attrs['index'])
    elif attrs['op'] == 'array_index' or attrs['op'] == 'array_update':
      if attrs.get('assumed_in_bounds') is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.assumed_in_bounds = _as_bool(
            attrs['assumed_in_bounds']
        )
    elif attrs['op'] == 'one_hot':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.lsb_prio = _as_bool(attrs['lsb_prio'])
    elif attrs['op'] == 'sign_ext':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.new_bit_count = _as_int(attrs['new_bit_count'])
    elif attrs['op'] == 'sel' or attrs['op'] == 'priority_sel':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.has_default_value = _as_bool(
          attrs.get('has_default_value', attrs.get('has_default', False))
      )
    elif attrs['op'] == 'receive':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = attrs['channel']
      if 'blocking' in attrs:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.blocking = _as_bool(attrs['blocking'])
    elif attrs['op'] == 'send':
      pb_unique_args = pb_node.unique_args.add()
      pb_unique_args.channel = attrs['channel']
    elif attrs['op'] == 'assert':
      message = attrs.get('message_', attrs.get('message'))
      if message is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.message = str(message)
      if attrs.get('label') is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.label = str(attrs['label'])
    elif attrs['op'] == 'trace':
      trace_format = attrs.get('xls_format', attrs.get('format'))
      if trace_format is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.format = str(trace_format)
      if attrs.get('verbosity') is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.verbosity = _as_int(attrs['verbosity'])
    elif attrs['op'] == 'state_read':
      if attrs.get('index') is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.index = _as_int(attrs['index'])
      if attrs.get('state_element') is not None:
        pb_unique_args = pb_node.unique_args.add()
        pb_unique_args.state_element = str(attrs['state_element'])
      if attrs.get('init') is not None:
        pb_unique_args = pb_node.unique_args.add()
        node_init_value = _parse_value_for_type(attrs['init'], data_type)
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
