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

"""A collection of dataclasses to represent data types similar to XLS types.

Also, we implement a simple parser for the IR format of these types. Ideally,
this would all be done by leveraging the existing C++ parser, but for now we're
using a regex-based approach.
"""

import abc
import dataclasses
import re

from xls.ir import xls_type_pb2


class DataType(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def to_proto(self) -> xls_type_pb2.TypeProto:
    """Converts the data type to an `xls.TypeProto` message."""
    raise NotImplementedError()


@dataclasses.dataclass
class BitsType(DataType):
  bit_count: int

  def to_proto(self) -> xls_type_pb2.TypeProto:
    """Converts the data type to an `xls.TypeProto` message."""
    return xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.BITS,
        bit_count=self.bit_count,
    )


@dataclasses.dataclass
class ArrayType(DataType):
  array_size: int
  array_element: DataType

  def to_proto(self) -> xls_type_pb2.TypeProto:
    """Converts the data type to an `xls.TypeProto` message."""
    return xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.ARRAY,
        array_size=self.array_size,
        array_element=self.array_element.to_proto(),
    )


@dataclasses.dataclass
class TupleType(DataType):
  tuple_elements: list[DataType]

  def to_proto(self) -> xls_type_pb2.TypeProto:
    """Converts the data type to an `xls.TypeProto` message."""
    return xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.TUPLE,
        tuple_elements=[element.to_proto() for element in self.tuple_elements],
    )


@dataclasses.dataclass
class TokenType(DataType):

  def to_proto(self) -> xls_type_pb2.TypeProto:
    """Converts the data type to an `xls.TypeProto` message."""
    return xls_type_pb2.TypeProto(
        type_enum=xls_type_pb2.TypeProto.TypeEnum.TOKEN
    )


def parse_data_type(dtype_string):
  """Parses a string representation of a data type in the IR format and returns a `DataType` object.

  Args:
      dtype_string (str): A string representing the data type in the IR format.

  Returns:
      DataType: A `DataType` object containing information about the parsed
      data type.
  Raises:
      ValueError: If an unrecognized data type format is encountered.
  """
  match = re.search(r"id=.*$", dtype_string)
  dtype_string = dtype_string if not match else dtype_string[: match.start()]
  dtype_string = dtype_string.strip()
  if dtype_string == "()":
    return parse_tuple_type(None)
  elif re.search(r"^(?!bits\[\d+\]$|token$).*\[\d+\]$", dtype_string):
    return parse_array_type(dtype_string)
  elif dtype_string.startswith("bits"):
    return parse_bits_type(dtype_string)
  elif dtype_string == "token":
    return TokenType()
  elif dtype_string.startswith("(") and dtype_string.endswith(")"):
    return parse_tuple_type(dtype_string[1:-1])
  else:
    raise ValueError(f"Unknown dtype: {dtype_string}")


def parse_bits_type(bits_type_string):
  match = re.search(r"bits\[(\d+)\]", bits_type_string)
  if match:
    return BitsType(bit_count=int(match.group(1)))
  else:
    raise ValueError(f"Unknown dtype: {bits_type_string}")


def parse_array_type(array_type_string):
  last_bracket_pos = array_type_string.rfind("[")
  type_part = array_type_string[:last_bracket_pos]
  size_part = array_type_string[last_bracket_pos + 1 : -1]
  size = int(size_part)
  array_element = parse_data_type(type_part)
  return ArrayType(array_size=size, array_element=array_element)


def parse_tuple_type(tuple_type_string):
  """Parses a string representing a tuple data type in the IR format and returns a string representation.

  This function assumes the tuple data type is specified directly within the
  `dtype_string` using parentheses to enclose the element data types separated
  by commas.

  Args:
      tuple_type_string (str): A string representing the data type in the IR
        format, expected to be in the format "tuple(type1, type2, ...)".

  Returns:
      str: A string representation of the tuple data type, following the
      format "tuple(type1, type2, ...)".
  Raises:
      ValueError: If the `dtype_string` format is not valid for a tuple type
                  (e.g., missing parentheses or invalid separators).
  """
  elements = []
  depth = 0
  last_split = 0
  if tuple_type_string is None:
    return TupleType(tuple_elements=elements)
  for i, char in enumerate(tuple_type_string):
    if char == "(":
      depth += 1
    elif char == ")":
      depth -= 1
    elif char == "," and depth == 0:
      elements.append(parse_data_type(tuple_type_string[last_split:i].strip()))
      last_split = i + 1
  elements.append(parse_data_type(tuple_type_string[last_split:].strip()))
  return TupleType(tuple_elements=elements)
