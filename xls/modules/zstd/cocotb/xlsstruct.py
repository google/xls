# Copyright 2024 The XLS Authors
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

import random
from dataclasses import asdict, astuple, dataclass, fields

from cocotb.binary import BinaryValue


class TruncationError(Exception):
  pass

def xls_dataclass(cls):
  """
  Class decorator for XLS structs.
  Usage:

  @xls_dataclass
  class MyStruct(XLSStruct):
    ...
  """
  return dataclass(cls, repr=False)

@dataclass
class XLSStruct:
  """
  Represents XLS struct on the Python side, allowing serialization/deserialization
  to/from common formats and usage with XLS{Driver, Monitor}.

  The intended way to use this class is to inherit from it, specify the fields with
  <field>: <width> [= <default value>] syntax and decorate the inheriting class with
  @XLSDataclass. Objects of this class can be instantiated and used like usual
  dataclass objects, with a few extra methods and properties available. They can also
  be passed as arguments to XLSChannelDriver.send and will be serialized to expected
  bit vector. Class can be passed to XLSChannelMonitor ``struct`` constructor argument
  to automatically deserialize all transfers to the provided struct.

  Example:

    from xlsstruct import XLSDataclass, XLSStruct

    @XLSDataclass
    class MyStruct(XLSStruct):
      data: 32
      ok: 1
      id: 4 = 0

    monitor = XLSChannelMonitor(dut, CHANNEL_PREFIX, dut.clk, MyStruct)

    driver = XLSChannelDriver(dut, CHANNEL_PREFIX, dut.clk)
    driver.send(MyStruct(
      data = 0xdeadbeef,
      ok = 1,
      id = 3,
    ))
    # struct fields can also be randomized
    driver.send(MyStruct.randomize())
  """

  @classmethod
  def _masks(cls):
    """
    Returns a list of field-sized bitmasks.

    For example for fields of widths 2, 3, 4
    returns [2'b11, 3'b111, 4'b1111].
    """
    masks = []
    for field in fields(cls):
      width = field.type
      masks += [(1 << width) - 1]
    return masks

  @classmethod
  def _positions(cls):
    """
    Returns a list of start positions in a bit vector for
    struct's fields.

    For example for fields of widths 1, 2, 3, 4, 5, 6
    returns [20, 18, 15, 11, 6, 0]
    """
    positions = []
    for i, field in enumerate(fields(cls)):
      width = field.type
      if i == 0:
        positions += [cls.total_width - width]
      else:
        positions += [positions[i-1] - width]
    return positions

  @classmethod
  @property
  def total_width(cls):
    """
    Returns total bit width of the struct
    """
    return sum(field.type for field in fields(cls))

  @property
  def value(self):
    """
    Returns struct's value as a Python integer
    """
    value = 0
    masks = self._masks()
    positions = self._positions()
    for field_val, mask, pos in zip(astuple(self), masks, positions):
      if field_val > mask:
        raise TruncationError(f"Signal value is wider than its bit width")
      value |= (field_val & mask) << pos
    return value

  @property
  def binaryvalue(self):
    """
    Returns struct's value as a cocotb.binary.BinaryValue
    """
    return BinaryValue(self.binstr)

  @property
  def binstr(self):
    """
    Returns struct's value as a string with its binary representation
    """
    return f"{self.value:>0{self.total_width}b}"

  @property
  def hexstr(self):
    """
    Returns struct's value as a string with its hex representation
    (without leading "0x")
    """
    return f"{self.value:>0{self.total_width // 4}x}"

  @classmethod
  def from_int(cls, value):
    """
    Returns an instance of the struct from Python integer
    """
    instance = {}
    masks = cls._masks()
    positions = cls._positions()
    for field, mask, pos in zip(fields(cls), masks, positions):
      instance[field.name] = (value >> pos) & mask
    return cls(**instance)

  @classmethod
  def randomize(cls):
    """
    Returns an instance of the struct with all fields' values randomized
    """
    instance = {}
    for field in fields(cls):
      instance[field.name] = random.randrange(0, 2**field.type)
    return cls(**instance)

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    classname = self.__class__.__name__
    fields = [f"{name}={hex(value)}" for name, value in asdict(self).items()]
    return f"{classname}({', '.join(fields)})"
