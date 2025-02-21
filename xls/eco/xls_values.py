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

"""Parses XLS values from the IR format.

Ideally, this would all be done by leveraging the existing C++ parser, but for
now we're using a regex-based approach.
"""

import abc
import dataclasses
import struct

from xls.eco import xls_types
from xls.ir import xls_value_pb2


class ValueBase(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def to_proto(self, tpe: xls_types.DataType) -> xls_value_pb2.ValueProto:
    """Converts the value to an `xls.ValueProto` message."""
    raise NotImplementedError()


def _integer_to_bytes(integer, bit_count):
  """Converts an integer to bytes data as specified in the proto message.

  Args:
      integer: The integer to convert.
      bit_count: The number of bits in the integer.

  Returns:
      The bytes data representing the integer.
  """

  # Ensure the bit count is valid
  if bit_count <= 0:
    raise ValueError("Bit count must be positive")

  # Calculate the number of bytes required
  byte_count = (bit_count + 7) // 8

  # Convert the integer to bytes in little-endian order
  bytes_data = struct.pack(
      "<{}B".format(byte_count),
      *(integer >> (8 * i) & 0xFF for i in range(byte_count)),
  )

  return bytes_data


@dataclasses.dataclass
class BitsValue(ValueBase):
  """Represents a bits value."""

  data: int

  def to_proto(self, tpe: xls_types.DataType) -> xls_value_pb2.ValueProto:
    """Converts the value to an `xls.ValueProto` message."""
    if not isinstance(tpe, xls_types.BitsType):
      raise ValueError(f"Expected bits data type, got: {tpe}")
    return xls_value_pb2.ValueProto(
        bits=xls_value_pb2.ValueProto.Bits(
            bit_count=tpe.bit_count,
            data=_integer_to_bytes(self.data, tpe.bit_count),
        )
    )


@dataclasses.dataclass
class TupleValue(ValueBase):
  """Represents a tuple value."""

  elements: list[ValueBase]

  def to_proto(self, tpe: xls_types.DataType) -> xls_value_pb2.ValueProto:
    """Converts the value to an `xls.ValueProto` message."""
    if not isinstance(tpe, xls_types.TupleType):
      raise ValueError(f"Expected tuple data type, got: {tpe}")
    if len(self.elements) != len(tpe.tuple_elements):
      raise ValueError(
          f"Expected {len(tpe.tuple_elements)} elements, got:"
          f" {len(self.elements)}"
      )
    return xls_value_pb2.ValueProto(
        tuple=xls_value_pb2.ValueProto.Tuple(
            elements=[
                element.to_proto(tpe)
                for (element, tpe) in zip(self.elements, tpe.tuple_elements)
            ]
        )
    )


@dataclasses.dataclass
class ArrayValue(ValueBase):
  """Represents an array value."""

  elements: list[ValueBase]

  def to_proto(self, tpe: xls_types.DataType) -> xls_value_pb2.ValueProto:
    """Converts the value to an `xls.ValueProto` message."""
    if not isinstance(tpe, xls_types.ArrayType):
      raise ValueError(f"Expected array data type, got: {tpe}")
    if len(self.elements) != tpe.array_size:
      raise ValueError(
          f"Expected {tpe.array_size} elements, got: {len(self.elements)}"
      )
    return xls_value_pb2.ValueProto(
        array=xls_value_pb2.ValueProto.Array(
            elements=[
                element.to_proto(tpe.array_element) for element in self.elements
            ]
        )
    )


@dataclasses.dataclass
class TokenValue(ValueBase):
  """Represents a token value."""

  def to_proto(self, tpe: xls_types.DataType) -> xls_value_pb2.ValueProto:
    """Converts the value to an `xls.ValueProto` message."""
    if not isinstance(tpe, xls_types.TokenType):
      raise ValueError(f"Expected token data type, got: {tpe}")
    return xls_value_pb2.ValueProto(token=xls_value_pb2.ValueProto.Token())


def parse_value(value_string: str, idx=0) -> tuple[int, ValueBase]:
  """Parses a value from a string representation.

  Args:
    value_string (str): The string representation of the value.
    idx (int): The current index in the string.

  Returns:
    int, ValueType: A value and the data type of the value.

  Raises:
    ValueError: If the value is not properly formatted.
  """
  while idx < len(value_string):
    if value_string[idx] == "[":
      return parse_value_array(value_string, idx + 1)
    elif value_string[idx] == "(":
      return parse_value_tuple(value_string, idx + 1)
    elif value_string[idx].isdigit():
      return parse_value_bits(value_string, idx)
    elif value_string[idx] == "t":
      if value_string[idx : idx + 5] != "token":
        raise ValueError(
            f"Unknown dtype at index {idx}: {value_string[idx:idx+5]}"
        )
      return (idx + 5, TokenValue())
    else:
      raise ValueError(f"Unknown dtype at index {idx}: {value_string[idx]}")
  raise ValueError("Unexpected end of value")


def parse_value_array(value_string: str, idx):
  """Parses an array value from a string representation.

  Args:
    value_string (str): The string representation of the array value.
    idx (int): The current index in the string.

  Returns:
    list[int], ValueType: A list of values and the data type of the
    array value.

  Raises:
    ValueError: If the array value is not properly formatted.
  """
  elements = []
  while idx < len(value_string):
    if value_string[idx] == "]":
      return idx + 1, ArrayValue(elements=elements)
    elif value_string[idx] in " ,":
      idx += 1
    else:
      idx, element = parse_value(value_string, idx)
      elements.append(element)
  raise ValueError("Unexpected end of array")


def parse_value_tuple(value_string: str, idx):
  """Parses a tuple value from a string representation.

  Args:
    value_string (str): The string representation of the tuple value.
    idx (int): The current index in the string.

  Returns:
    Tuple[int, ValueType]: A tuple of values and the data type
    of the tuple value.

  Raises:
    ValueError: If the tuple value is not properly formatted.
  """
  elements = []
  while idx < len(value_string):
    if value_string[idx] == ")":
      return idx + 1, TupleValue(elements=elements)
    if value_string[idx] in " ,":
      idx += 1
    else:
      idx, element = parse_value(value_string, idx)
      elements.append(element)
  raise ValueError("Unexpected end of tuple")


def parse_value_bits(value_string: str, idx):
  """Parses a bits value from a string representation.

  Args:
    value_string (str): The string representation of the bits value.
    idx (int): The current index in the string.

  Returns:
    int, ValueType: A list of values and the data type of the
    bits value.

  Raises:
    ValueError: If the bits value is not properly formatted.
  """
  start_idx = idx
  while idx < len(value_string) and value_string[idx].isdigit():
    idx += 1
  return idx, BitsValue(data=int(value_string[start_idx:idx]))
