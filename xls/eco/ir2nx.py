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

"""Parses XLS IR into a networkx graph utilizing regexes."""

import re
import networkx as nx
from xls.eco import xls_types
from xls.eco import xls_values


class IrParser:
  """This class parses XLS intermediate representation (IR) into a networkx graph."""

  def __init__(self, path):
    """Initializes the IrParser object.

    Args:
      path (str): The path to the IR file.
    """
    self.node_colors = {
        "P": "#ffffff",
        "LT": "#90ee90",
        "OP": "#ffcccb",
    }
    self._node_parsers = {
        "encode": self._parse_solo_node,
        "reverse": self._parse_solo_node,
        "not": self._parse_solo_node,
        "neg": self._parse_solo_node,
        "or_reduce": self._parse_solo_node,
        "xor_reduce": self._parse_solo_node,
        "and_reduce": self._parse_solo_node,
        "sub": self._parse_binary_node,
        "shll": self._parse_binary_node,
        "shrl": self._parse_binary_node,
        "shra": self._parse_binary_node,
        "ule": self._parse_binary_node,
        "ult": self._parse_binary_node,
        "slt": self._parse_binary_node,
        "ugt": self._parse_binary_node,
        "uge": self._parse_binary_node,
        "sge": self._parse_binary_node,
        "sle": self._parse_binary_node,
        "sign_ext": self._parse_ext_node,
        "zero_ext": self._parse_ext_node,
        "ne": self._set_commutative(self._parse_binary_node),
        "eq": self._set_commutative(self._parse_binary_node),
        "udiv": self._parse_binary_node,
        "sdiv": self._parse_binary_node,
        "umul": self._set_commutative(self._parse_binary_node),
        "smul": self._set_commutative(self._parse_binary_node),
        "smulp": self._set_commutative(self._parse_binary_node),
        "umulp": self._set_commutative(self._parse_binary_node),
        "add": self._set_commutative(self._parse_binary_node),
        "concat": self._parse_nary_node,
        "tuple": self._parse_nary_node,
        "array": self._parse_nary_node,
        "nor": self._set_commutative(self._parse_nary_node),
        "or": self._set_commutative(self._parse_nary_node),
        "xor": self._set_commutative(self._parse_nary_node),
        "and": self._set_commutative(self._parse_nary_node),
        "nand": self._set_commutative(self._parse_nary_node),
        "one_hot_sel": self._parse_sel_node,
        "sel": self._parse_sel_node,
        "priority_sel": self._parse_sel_node,
        "bit_slice": self._parse_bit_slice,
        "dynamic_bit_slice": self._parse_dynamic_bit_slice,
        "after_all": self._parse_after_all,
        "literal": self._parse_literal,
        "one_hot": self._parse_one_hot,
        "array_index": self._parse_array_index,
        "array_update": self._parse_array_update,
        "tuple_index": self._parse_tuple_index,
        "receive": self._parse_receive,
        "send": self._parse_send,
        "next_value": self._parse_next_value,
        "assert": self._parse_dbg_node,
        "trace": self._parse_dbg_node,
        "state_read": self._parse_state_read,
        "decode": self._parse_decode_node,
    }
    self._path = path
    self.graph = nx.MultiDiGraph()
    # Some nodes such as select can have single node as their cases;
    # multidigraph allows for this.
    self.graph.graph["name"] = self._path
    self._ir = self._read_local_ir(path)
    self._parse_ir()

  def _set_commutative(self, fn):
    """Returns a wrapper function that sets commutative=True for the given function."""

    def _wrapper(*args, **kwargs):
      kwargs["commutative"] = True
      return fn(*args, **kwargs)

    return _wrapper

  def _read_local_ir(self, path):
    """A Generator that Reads the IR file locally line by line and yields each stripped line.

    Args:
        path (str): The path to the IR file.

    Yields:
        str: Each stripped line from the IR file.
    """
    with open(path, "rt") as file:
      for line in file:
        stripped_line = line.strip()
        if stripped_line:
          yield stripped_line

  def _parse_ir(self):
    for line in self._ir:
      if line and not line.startswith(("package", "file_number", "chan")):
        self._parse_ir_line(line)

  def _parse_ir_line(self, line):
    """Parses a single line of IR code and adds the corresponding node to the parser graph.

    This function handles different line formats based on keywords and
    punctuation. It extracts node information like name, data type, and op type,
    and calls the appropriate parsing function from `self._node_parsers` to
    build the IR graph.

    Args:
        line (str): A single line of IR code to be parsed.
    """
    if re.search(r"^[}]$", line):
      return
    if "top fn" in line:
      self._parse_top_fn(line)
    elif "top proc" in line:
      self._parse_top_proc(line)
    else:
      (line_lhs, line_rhs) = [l.strip() for l in line.split("=", 1)]
      node_name = line_lhs.split(":")[0].strip()
      op = line_rhs.split("(", 1)[0].strip()
      if node_name.startswith("ret "):
        node_name = node_name.removeprefix("ret ")
        self.graph.graph["ret"] = node_name
      data_type_str = line_lhs.split(":")[1].strip()
      parser = self._node_parsers[op]
      parser(line_rhs, node_name, data_type_str)

  def _split_inits(self, inits):
    """Splits a comma-separated list of initial values into individual values."""
    res = [""]
    depth = 0
    for e in inits:
      if (not depth) and (e == ","):
        res.append("")
        continue
      if e in "([":
        depth += 1
      elif e in ")]":
        depth -= 1
      res[-1] = res[-1] + e.strip()
    return res

  def _split_params(self, line):
    """Splits a line defining the parameters of a top function or proc in the IR.

    This function handles nested parentheses and brackets to correctly identify
    individual parameters. It iterates over the characters in the line and
    builds a list based on commas and nesting levels to separate parameters.

    Args:
      line (str): A line containing the parameter list of a top function/proc.

    Returns:
          list: A list of strings representing individual parameters.
    """
    stack = []  # Stack to track nesting of parentheses/brackets
    result = []  # List to store extracted elements
    current_element = ""
    for char in line:
      if char in "([{":
        stack.append(char)  # Push opening parenthesis/bracket onto stack
        current_element += (
            char  # Add opening parenthesis/bracket to current element
        )
      elif char in ")]}":
        if not stack:
          raise ValueError("Unexpected closing parenthesis/bracket")
        stack.pop()  # Pop closing parenthesis/bracket from stack
        current_element += (
            char  # Add closing parenthesis/bracket to current element
        )
      elif char == ",":
        if not stack:
          result.append(current_element)  # Add entire current element
          current_element = ""
        else:
          current_element += char  # Include comma within parentheses/brackets
      else:
        current_element += char  # Add other characters to current element
    if current_element and current_element != " ":
      result.append(current_element)  # Add remaining element
    return result

  def _num2bits(self, num):
    return f"bits[{num}]"

  def _str2bool(self, string):
    if string is None:
      return None
    else:
      string = string.lower()
      if string == "true":
        return True
      elif string == "false":
        return False
      else:
        raise ValueError(f"Unknown bool: {string}")

  def _parse_top_fn(self, line):
    line = line.split("->")[0]  # Drop return type
    start_index = line.find("(") + 1  # Skip "top fn " and opening parenthesis
    end_index = line.rfind(")")  # Find the last closing parenthesis
    line = line[start_index:end_index]
    params = self._split_params(line)
    for param in params:
      self._parse_param(param)

  def _parse_top_proc(self, line):
    """Parses a line representing a top proc in the IR format and adds it to the graph.

    Args:
        line (str): The line representing the top proc in the IR format.
    """
    line = line.split("(", 1)[1]
    line, inits = line.split("init=")
    state_elements = self._split_params(line)
    state_elements = zip(
        state_elements, self._split_inits(inits[1:-4]), strict=True
    )
    for index, state_element in enumerate(state_elements):
      self._parse_state_element(state_element, index)

  def _parse_after_all(self, line, node_name, data_type_str):
    """Parses a line representing an "after_all" node in the IR format and adds it to the graph.

    Args:
        line (str): The line representing the "after_all" node in the IR format.
        node_name (str): The name of the "after_all" node extracted from the IR
          line.
        data_type_str (str): The string representation of the data type for the
          "after_all" node.

    Raises:
        ValueError: If the format of the "after_all" line is unexpected or
        invalid (e.g., missing parentheses or invalid separators).
    """
    regex = re.compile(
        r"after_all\((?P<operands>[^)]*?),?\s*id\s*=\s*(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "after_all"
    data_type = xls_types.parse_data_type(data_type_str)
    operands = (
        [operand.strip() for operand in details["operands"].split(",")]
        if details["operands"]
        else None
    )
    operand_data_types = (
        [self.graph.nodes[operand]["data_type"] for operand in operands]
        if operands
        else None
    )
    if operands:
      operand_dtype_strs = [
          self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
          for operand in operands
      ]
    else:
      operand_dtype_strs = None
    if operands:
      self.graph.add_node(
          node_name,
          op=op,
          data_type=data_type,
          operand_data_types=operand_data_types,
          id=int(details["id"]),
          color=self.node_colors["OP"],
          cost_attributes={
              "op": op,
              "dtype_str": data_type_str,
              "operand_dtype_strs": operand_dtype_strs,
          },
      )
      for i, operand in enumerate(operands):
        self._add_edge(operand, node_name, i)
    else:
      self.graph.add_node(
          node_name,
          op=op,
          data_type=data_type,
          id=int(details["id"]),
          color=self.node_colors["OP"],
          cost_attributes={
              "op": op,
              "dtype_str": data_type_str,
          },
      )

  def _map_bit_counts_to_value_types(self, data_type, value_type):
    """Maps bit counts from a DataType to a corresponding ValueType instance.

    Args:
        data_type: The root DataType to map.
        value_type: The ValueType instance to populate.

    Returns:
        The populated ValueType instance.
    """

    def _map(data_type, value_type):
      if isinstance(data_type, xls_types.BitsType):
        value_type.bit_count = data_type.bit_count
      elif isinstance(data_type, xls_types.ArrayType):
        for i in range(data_type.array_size):
          _map(data_type.array_element, value_type.elements[i])
      elif isinstance(data_type, xls_types.TupleType):
        for i, element in enumerate(data_type.tuple_elements):
          _map(element, value_type.elements[i])
      elif isinstance(data_type, xls_types.TokenType):
        pass

    _map(data_type, value_type)
    return value_type

  def _parse_literal(self, line, node_name, data_type_str):
    """Parses a line representing a literal node in the IR format and adds it to the graph.

    Args:
        line (str): The line representing the literal node in the IR format.
        node_name (str): The name of the literal node extracted from the IR
          line.
        data_type_str (str): The string representation of the data type for the
          literal node.
    """
    regex = re.compile(r"literal\(value=(?P<value>.+),\s+id=(?P<id>\d+)")
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "literal"
    data_type = xls_types.parse_data_type(data_type_str)
    value = xls_values.parse_value(details["value"])[1]
    value = self._map_bit_counts_to_value_types(data_type, value)
    self.graph.add_node(
        node_name,
        op=op,
        value=value,
        data_type=data_type,
        id=int(details["id"]),
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "value_str": details["value"],
        },
        color=self.node_colors["LT"],
    )

  def _parse_param(self, param_str):
    """Parses a line representing a param node in the IR format and adds it to the graph."""
    id_idx = param_str.find("id=")
    id_str = None
    if id_idx >= 0:
      id_str = param_str[id_idx:].strip("id=").strip()
      param_str = param_str[:id_idx].strip()
    node_name = param_str.split(":")[0].strip()
    op = "param"
    data_type_str = param_str.split(":")[1].strip()
    data_type = xls_types.parse_data_type(data_type_str)
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        id=int(id_str) if id_str is not None else None,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
        },
        color=self.node_colors["P"],
    )

  def _parse_state_element(self, state_element_str, index):
    """Parses a line representing a state element in the IR format and adds it to the graph."""
    state_element_str, init_str = state_element_str
    node_name = state_element_str.split(":")[0].strip()
    op = "state_read"
    data_type_str = state_element_str.split(":")[1].strip()
    data_type = xls_types.parse_data_type(data_type_str)
    init = xls_values.parse_value(init_str)[1]
    init = self._map_bit_counts_to_value_types(data_type, init)
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        state_element=node_name,
        index=index,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "init": init,
            "state_element": node_name,
            "index": index,
        },
        color=self.node_colors["P"],
        init=init,
    )

  def _parse_solo_node(self, line, node_name, data_type_str):
    """Parses a line representing a "Solo Node" in the IR format and adds it to the graph.

    Args:
        line (str): The line representing the Solo Node in the IR format.
        node_name (str): The name of the Solo Node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          Solo Node.

    Raises:
        ValueError: If the format of the "Solo Node" line is unexpected or
        invalid (e.g., missing parentheses or invalid separators).
    """
    op = line.split("(", 1)[0].strip()
    regex = re.compile(rf"{op}\((?P<operand>\w+(\.\d+)?),\s*id=(?P<id>\d+)")
    match = regex.match(line)
    details = match.groupdict() if match else None
    operand = details["operand"].strip()
    data_type = xls_types.parse_data_type(data_type_str)
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        id=int(details["id"]),
        color=self.node_colors["OP"],
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
        },
    )
    self._add_edge(operand, node_name, 0)

  def _parse_binary_node(
      self, line, node_name, data_type_str, commutative=False
  ):
    """Parses a line representing a node with two lhs and rhs operands and adds it to the graph.

    Args:
        line (str): The line representing the binary operation node in the IR
          format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.
        commutative (bool, optional): Whether to treat operands with the same
          data type as a single commutative operand (defaults to False).

    Raises:
        ValueError: If the format of the binary operation line is unexpected or
        invalid.
    """
    op = line.split("(", 1)[0].strip()
    regex = re.compile(
        rf"{op}\((?P<lhs>\w+(\.\d+)?),\s*(?P<rhs>\w+(\.\d+)?),\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    data_type = xls_types.parse_data_type(data_type_str)
    operands = [details["lhs"], details["rhs"]]
    operand_data_types = [
        self.graph.nodes[operand]["data_type"] for operand in operands
    ]
    dtype_strs = [
        self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
        for operand in operands
    ]
    operand_dtype_strs = (
        tuple(sorted(dtype_strs)) if commutative else tuple(dtype_strs)
    )
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        commutative=commutative,
        operand_data_types=operand_data_types,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_strs": operand_dtype_strs,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(details["lhs"], node_name, 0)
    self._add_edge(details["rhs"], node_name, 1)

  def _parse_nary_node(self, line, node_name, data_type_str, commutative=False):
    """Parses a line representing a multi-operand operation node and adds it to the graph.

    Args:
        line (str): The line representing the multi-operand operation node in
          the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.
        commutative (bool, optional): Whether to treat operands with the same
          data type as a single commutative operand (defaults to False).

    Raises:
        ValueError: If the format of the multi-operand operation line is
        unexpected or invalid.
    """
    op = line.split("(", 1)[0].strip()
    regex = re.compile(rf"{op}\((?P<operands>[^)]*?),\s*id=(?P<id>\d+)")
    match = regex.match(line)
    details = match.groupdict() if match else None
    data_type = xls_types.parse_data_type(data_type_str)
    operands = [operand.strip() for operand in details["operands"].split(",")]
    dtype_strs = [
        self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
        for operand in operands
    ]
    operand_dtype_strs = (
        tuple(sorted(dtype_strs)) if commutative else tuple(dtype_strs)
    )
    operand_data_types = [
        self.graph.nodes[operand]["data_type"] for operand in operands
    ]
    self.graph.add_node(
        node_name,
        op=op,
        commutative=commutative,
        data_type=data_type,
        operand_data_types=operand_data_types,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_strs": operand_dtype_strs,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    for i, operand in enumerate(operands):
      self._add_edge(operand, node_name, i)

  def _parse_sel_node(self, line, node_name, data_type_str):
    """Parses a line representing a select node and adds it to the graph.

    Args:
        line (str): The line representing the node with a selector in the IR
          format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the node with selector line is unexpected
        or invalid (e.g., missing parentheses or invalid separators).
    """
    op = line.split("(", 1)[0].strip()
    regex = re.compile(
        rf"{op}\((?P<selector>\w+(\.\d+)?),\s*cases=\[(?P<operands>[^)]+)\],\s*(default=(?P<default>\w+(\.\d+)?),\s*)?id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    operands = [item.strip() for item in details["operands"].split(",")]
    data_type = xls_types.parse_data_type(data_type_str)
    selector = details["selector"]
    default = details["default"] if details["default"] else None
    operand_dtype_strs = (
        self.graph.nodes[selector]["cost_attributes"]["dtype_str"],
        *(
            self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
            for operand in operands
        ),
        (
            self.graph.nodes[default]["cost_attributes"]["dtype_str"]
            if default is not None
            else None
        ),
    )
    operand_data_types = (
        [self.graph.nodes[selector]["data_type"]]
        + [self.graph.nodes[operand]["data_type"] for operand in operands]
        + (
            [self.graph.nodes[default]["data_type"]]
            if default is not None
            else []
        )
    )
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_types=operand_data_types,
        has_default_value=default is not None,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_strs": operand_dtype_strs,
            "has_default_value": default is not None,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(details["selector"], node_name, 0)
    for i, operand in enumerate(operands, start=1):
      self._add_edge(operand, node_name, i)
    if default is not None:
      self._add_edge(default, node_name, len(operands) + 1)

  def _parse_array_index(self, line, node_name, data_type_str):
    """Parses a line representing an array index node and adds it to the graph.

    Args:
      line (str): The line representing the array index node in the IR format.
      node_name (str): The name of the node extracted from the IR line.
      data_type_str (str): The string representation of the data type for the
        node.

    Raises:
      ValueError: If the format is invalid.
    """
    regex = re.compile(
        r"array_index\((?P<array>\w+(\.\d+)?),\s*indices=\[(?P<operands>[^)]+)\](?:,\s*assumed_in_bounds=(?P<assumed_in_bounds>true|false))?,\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "array_index"
    data_type = xls_types.parse_data_type(data_type_str)
    array = details["array"]
    operands = [operand.strip() for operand in details["operands"].split(",")]
    assumed_in_bounds = (
        self._str2bool(details["assumed_in_bounds"])
        if details["assumed_in_bounds"]
        else None
    )
    operand_dtype_strs = (
        self.graph.nodes[array]["cost_attributes"]["dtype_str"],
        *(
            self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
            for operand in operands
        ),
    )
    operand_data_types = [self.graph.nodes[array]["data_type"]] + [
        self.graph.nodes[operand]["data_type"] for operand in operands
    ]
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        assumed_in_bounds=assumed_in_bounds,
        operand_data_types=operand_data_types,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_strs": operand_dtype_strs,
            "assumed_in_bounds": assumed_in_bounds,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(details["array"], node_name, 0)
    for i, operand in enumerate(operands, start=1):
      self._add_edge(operand, node_name, i)

  def _parse_array_update(self, line, node_name, data_type_str):
    """Parses a line representing an array update node and adds it to the graph.

    Args:
      line (str): The line representing the array update node in the IR format.
      node_name (str): The name of the node extracted from the IR line.
      data_type_str (str): The string representation of the data type for the
        node.

    Raises:
      ValueError: If the line format is invalid.
    """
    regex = re.compile(
        r"array_update\((?P<array>\w+(\.\d+)?)\s*,\s*(?P<value>\w+(\.\d+)?)\s*,\s*indices\s*=\s*\[(?P<operands>[^)]+)\],\s*(assumed_in_bounds\s*=\s*(?P<assumed_in_bounds>\w+)\s*,\s*)?id\s*=\s*(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "array_update"
    data_type = xls_types.parse_data_type(data_type_str)
    array = details["array"]
    value = details["value"]
    assumed_in_bounds = self._str2bool(details["assumed_in_bounds"])
    operands = [operand.strip() for operand in details["operands"].split(",")]
    operand_dtype_strs = (
        self.graph.nodes[value]["cost_attributes"]["dtype_str"],
        self.graph.nodes[array]["cost_attributes"]["dtype_str"],
        *(
            self.graph.nodes[operand]["cost_attributes"]["dtype_str"]
            for operand in operands
        ),
    )
    operand_data_types = (
        [self.graph.nodes[array]["data_type"]]
        + [self.graph.nodes[value]["data_type"]]
        + [self.graph.nodes[operand]["data_type"] for operand in operands]
    )
    self.graph.add_node(
        node_name,
        op=op,
        value=value,
        data_type=data_type,
        operand_data_types=operand_data_types,
        assumed_in_bounds=assumed_in_bounds,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_strs": operand_dtype_strs,
            "assumed_in_bounds": assumed_in_bounds,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(details["array"], node_name, 0)
    self._add_edge(details["value"], node_name, 1)
    for i, operand in enumerate(operands, start=2):
      self._add_edge(operand, node_name, i)

  def _parse_bit_slice(self, line, node_name, data_type_str):
    """Parses a line representing a bit slice node and adds it to the graph.

    Args:
        line (str): The line representing the bit slice node in the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the bit slice line is unexpected or
        invalid.
    """
    regex = re.compile(
        r"bit_slice\((?P<operand>\w+(\.\d+)?),\s*start=(?P<start>\d+),\s*width=(?P<width>\d+),\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "bit_slice"
    data_type = xls_types.parse_data_type(data_type_str)
    operand = details["operand"]
    start = int(details["start"])
    width = int(details["width"])
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        start=start,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "start": start,
            "width": width,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(operand, node_name, 0)

  def _parse_dynamic_bit_slice(self, line, node_name, data_type_str):
    """Parses a line representing a dynamic bit slice node and adds it to the graph.

    Args:
        line (str): The line representing the dynamic bit slice node in the IR
          format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.
    """
    regex = re.compile(
        r"dynamic_bit_slice\((?P<operand>\w+(.\d+)?)\s*,\s*(?P<start>\w+(.\d+)?)\s*,\s*width=(?P<width>\d+)\s*,\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "dynamic_bit_slice"
    data_type = xls_types.parse_data_type(data_type_str)
    operand = details["operand"]
    start = details["start"]
    width = int(details["width"])
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        start=start,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "start": start,
            "width": width,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(operand, node_name, 0)
    self._add_edge(start, node_name, 1)

  def _parse_tuple_index(self, line, node_name, data_type_str):
    """Parses a line representing a tuple index node and adds it to the graph.

    This function extracts information from a line in the IR format that defines
    a tuple indexing operation. It then adds a corresponding node to the IR
    graph.

    Args:
        line (str): The line representing the tuple index node in the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the tuple index line is unexpected or
        invalid.
    """
    regex = re.compile(
        r"tuple_index\((?P<operand>\w+(\.\d+)?),\s+index=(?P<index>\d+),\s+id=(?P<id>\d+)"
    )
    match = regex.match(line)
    op = "tuple_index"
    details = match.groupdict() if match else None
    data_type = xls_types.parse_data_type(data_type_str)
    operand = details["operand"]
    index = int(details["index"])
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        index=index,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "index": index,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(operand, node_name, 0)

  def _parse_one_hot(self, line, node_name, data_type_str):
    """Parses a line representing a one-hot node and adds it to the graph.

    Args:
        line (str): The line representing the one-hot node in the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the one-hot line is unexpected or invalid.
    """
    regex = re.compile(
        r"one_hot\((?P<operand>\w+(\.\d+)?),\s*lsb_prio=(?P<lsb_prio>\w+),\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "one_hot"
    data_type = xls_types.parse_data_type(data_type_str)
    operand = details["operand"]
    lsb_prio = self._str2bool(details["lsb_prio"])
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        lsb_prio=lsb_prio,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "lsb_prio": lsb_prio,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(operand, node_name, 0)

  def _parse_ext_node(self, line, node_name, data_type_str):
    """Parses a line representing a sign extension node and adds it to the graph.

    Args:
        line (str): The line representing the sign extension node in the IR
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the sign extension line is unexpected or
        invalid.
    """
    data_type = xls_types.parse_data_type(data_type_str)
    op = line.split("(", 1)[0].strip()
    regex = re.compile(
        rf"{op}\((?P<operand>\w+(\.\d+)?),\s*new_bit_count=(?P<new_bit_count>\d+),\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    operand = details["operand"]
    new_bit_count = int(details["new_bit_count"])
    self.graph.add_node(
        node_name,
        op=op,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        new_bit_count=new_bit_count,
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "new_bit_count": new_bit_count,
        },
        id=int(details["id"]),
        color=self.node_colors["OP"],
    )
    self._add_edge(operand, node_name, 0)

  def _parse_receive(self, line, node_name, data_type_str):
    """Parses a line representing a "receive" operation and adds it as a node to the graph.

    Args:
      line: The string representation of the "receive" operation to be parsed.
      node_name: The name to be assigned to the created node in the graph.
      data_type_str: The string representing the data type of the receive
        operation.
    """
    regex = re.compile(
        r"receive\((?P<token>\w+(\.\d+)?),\s*(predicate\s*=(?P<predicate>\w+(\.\d+)?),\s*)?(blocking=(?P<blocking>\w+),\s*)?channel=\s*(?P<channel>\w+(\.\d+)?),\s*id\s*=\s*(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "receive"
    data_type = xls_types.parse_data_type(data_type_str)
    channel = details["channel"]
    predicate = details["predicate"] if details["predicate"] else None
    blocking = (
        self._str2bool(details["blocking"]) if details["blocking"] else False
    )
    token_data_type = self.graph.nodes[details["token"]]["data_type"]
    node_args = {
        "data_type": data_type,
        "op": op,
        "channel": channel,
        "blocking": blocking,
        "cost_attributes": {
            "op": op,
            "dtype_str": data_type_str,
            "channel": channel,
            "blocking": blocking,
        },
        "id": int(details["id"]),
        "color": self.node_colors["OP"],
    }
    if predicate:
      predicate_data_type = self.graph.nodes[details["predicate"]]["data_type"]
      node_args["operand_data_types"] = [token_data_type, predicate_data_type]
      # Add operand_dtype_str for both token and predicate if predicate exists
      node_args["cost_attributes"]["operand_dtype_str"] = (
          self.graph.nodes[details["token"]]["cost_attributes"]["dtype_str"],
          self.graph.nodes[details["predicate"]]["cost_attributes"][
              "dtype_str"
          ],
      )
    else:
      node_args["operand_data_type"] = token_data_type
      # Add operand_dtype_str for only the token if no predicate
      node_args["cost_attributes"]["operand_dtype_str"] = self.graph.nodes[
          details["token"]
      ]["cost_attributes"]["dtype_str"]
    self.graph.add_node(node_name, **node_args)
    self._add_edge(details["token"], node_name, 0)
    if predicate:
      self._add_edge(predicate, node_name, 1)

  def _parse_send(self, line, node_name, data_type_str):
    """Parses a line representing a send node and adds it to the graph.

    Args:
        line (str): The line representing the send node in the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.

    Raises:
        ValueError: If the format of the send line is unexpected or invalid.
    """
    regex = re.compile(
        r"send\((?P<token>\w+(\.\d+)?),\s*(?P<data>\w+(\.\d+)?),\s*(predicate\s*=(?P<predicate>\w+(\.\d+)?),\s*)?channel=(?P<channel>\w+(\.\d+)?\s*),\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "send"
    data_type = xls_types.parse_data_type(data_type_str)
    channel = details["channel"]
    data = details["data"]
    predicate = details["predicate"] if details["predicate"] else None
    token_data_type = self.graph.nodes[details["token"]]["data_type"]
    node_args = {
        "data_type": data_type,
        "op": op,
        "channel": channel,
        "cost_attributes": {
            "op": op,
            "dtype_str": data_type_str,
            "channel": channel,
        },
        "id": int(details["id"]),
        "color": self.node_colors["OP"],
    }
    if predicate:
      predicate_data_type = self.graph.nodes[details["predicate"]]["data_type"]
      node_args["operand_data_types"] = [token_data_type, predicate_data_type]
      # Add operand_dtype_str for both token and predicate if predicate exists
      node_args["cost_attributes"]["operand_dtype_str"] = (
          self.graph.nodes[details["token"]]["cost_attributes"]["dtype_str"],
          self.graph.nodes[details["predicate"]]["cost_attributes"][
              "dtype_str"
          ],
      )
    else:
      node_args["operand_data_type"] = token_data_type
      # Add operand_dtype_str for only the token if no predicate
      node_args["cost_attributes"]["operand_dtype_str"] = self.graph.nodes[
          details["token"]
      ]["cost_attributes"]["dtype_str"]
    self.graph.add_node(node_name, **node_args)
    self._add_edge(details["token"], node_name, 0)
    self._add_edge(data, node_name, 1)
    if predicate:
      self._add_edge(predicate, node_name, 2)

  def _parse_state_read(self, line, node_name, data_type_str):
    """populates the id and name field of an existing state_read node.

    Args:
      line: The string representation of the "state_read" operation to be
        parsed.
      node_name: The name to be assigned to the created node in the graph.
      data_type_str: The string representing the data type of the "state_read"
        operation.
    """
    regex = re.compile(
        r"state_read\(state_element\s*=\s*(?P<state_element>\w+)\s*,\s*id\s*=\s*(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    data_type = xls_types.parse_data_type(data_type_str)
    state_element = details["state_element"]
    assert self.graph.nodes[state_element]["op"] == "state_read"
    assert self.graph.nodes[state_element]["data_type"] == data_type
    assert (
        self.graph.nodes[state_element]["cost_attributes"]["state_element"]
        == state_element
    )
    self.graph.nodes[state_element]["id"] = int(details["id"])
    if state_element != node_name:
      mapping = {state_element: node_name}
      nx.relabel_nodes(self.graph, mapping, copy=False)

  def _parse_next_value(self, line, node_name, data_type_str):
    """Parses a line representing a "next_value" operation and adds it as a node to the graph.

    Args:
      line: The string representation of the "next_value" operation to be
        parsed.
      node_name: The name to be assigned to the created node in the graph.
      data_type_str: The string representing the data type of the "next_value"
        operation.
    """
    regex = re.compile(
        r"next_value\(param=(?P<param>\w+(\.\d+)?),\s*value=(?P<value>\w+(\.\d+)?),\s*(predicate=(?P<predicate>\w+(\.\d+)?),\s*)?id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    op = "next_value"
    data_type = xls_types.parse_data_type(data_type_str)
    param = details["param"]
    value = details["value"]
    predicate = details["predicate"] if details["predicate"] else None
    node_args = {
        "op": op,
        "data_type": data_type,
        "cost_attributes": {
            "op": op,
            "dtype_str": data_type_str,
        },
        "id": int(details["id"]),
        "color": self.node_colors["OP"],
    }
    if predicate:
      node_args["operand_data_types"] = [
          self.graph.nodes[param]["data_type"],
          self.graph.nodes[value]["data_type"],
          self.graph.nodes[predicate]["data_type"],
      ]
      node_args["cost_attributes"]["operand_dtype_str"] = (
          self.graph.nodes[param]["cost_attributes"]["dtype_str"],
          self.graph.nodes[value]["cost_attributes"]["dtype_str"],
          self.graph.nodes[predicate]["cost_attributes"]["dtype_str"],
      )
    else:
      node_args["operand_data_types"] = [
          self.graph.nodes[param]["data_type"],
          self.graph.nodes[value]["data_type"],
      ]
      node_args["cost_attributes"]["operand_dtype_str"] = [
          self.graph.nodes[param]["cost_attributes"]["dtype_str"],
          self.graph.nodes[value]["cost_attributes"]["dtype_str"],
      ]
    self.graph.add_node(node_name, **node_args)
    self._add_edge(details["param"], node_name, 0)
    self._add_edge(details["value"], node_name, 1)
    if predicate:
      self._add_edge(predicate, node_name, 2)

  def _parse_decode_node(self, line, node_name, data_type_str):
    """Parses a line representing a decode node and adds it to the graph.

    Args:
        line (str): The line representing the decode node in the IR format.
        node_name (str): The name of the node extracted from the IR line.
        data_type_str (str): The string representation of the data type for the
          node.
    """
    op = line.split("(", 1)[0].strip()
    regex = re.compile(
        rf"{op}\((?P<operand>\w+(\.\d+)?),\s*width\s*=\s*(?P<width>\w+(\.\d+)?)\s*,\s*id=(?P<id>\d+)"
    )
    match = regex.match(line)
    details = match.groupdict() if match else None
    operand = details["operand"].strip()
    width = int(details["width"])
    data_type = xls_types.parse_data_type(data_type_str)
    self.graph.add_node(
        node_name,
        op=op,
        width=width,
        data_type=data_type,
        operand_data_type=self.graph.nodes[operand]["data_type"],
        id=int(details["id"]),
        color=self.node_colors["OP"],
        cost_attributes={
            "op": op,
            "dtype_str": data_type_str,
            "operand_dtype_str": self.graph.nodes[operand]["cost_attributes"][
                "dtype_str"
            ],
            "width": width,
        },
    )
    self._add_edge(operand, node_name, 0)

  def _parse_dbg_node(self, line, node_name, data_type_str):
    """This is a software-only operation and has no representation in the generated hardware.

    Skipping...

    Args:
      line: Not used.
      node_name: Not used.
      data_type_str: Not used.
    """

  def _add_edge(self, source, sink, index):
    """Adds an edge to the graph."""
    cost_attributes = {
        "source_data_type": self.graph.nodes[source]["cost_attributes"][
            "dtype_str"
        ],
        "sink_data_type": self.graph.nodes[sink]["cost_attributes"][
            "dtype_str"
        ],
    }
    if (
        "commutative" not in self.graph.nodes[sink]
        or not self.graph.nodes[sink]["commutative"]
    ):
      cost_attributes["index"] = index
    # if edge already exists print a warning:
    self.graph.add_edge(
        source, sink, key=int(index), cost_attributes=cost_attributes
    )
