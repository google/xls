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

"""A collection of dataclasses to represent data types similar to XLS types."""

import dataclasses


class DataType:
  pass


@dataclasses.dataclass
class BitsType(DataType):
  bit_count: int | None = None


@dataclasses.dataclass
class ArrayType(DataType):
  array_size: int | None = None
  array_element: DataType | None = None


@dataclasses.dataclass
class TupleType(DataType):
  tuple_elements: list[DataType] | None = None


@dataclasses.dataclass
class TokenType(DataType):
  pass


class ValueType:
  pass


@dataclasses.dataclass
class BitsValue(ValueType):
  data: int | None = None
  bit_count: int | None = None


@dataclasses.dataclass
class TupleValue(ValueType):
  elements: list[ValueType] | None = None


@dataclasses.dataclass
class ArrayValue(ValueType):
  elements: list[ValueType] | None = None


@dataclasses.dataclass
class TokenValue(ValueType):
  pass
