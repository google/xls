# Lint as: python3
#
# Copyright 2020 The XLS Authors
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
"""Helper utilities for asserting DSLX interpreter/LLVM IR JIT equivalence."""

from typing import Iterable

from xls.dslx import bit_helpers
from xls.dslx.python import interp_value as dslx_value
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.ir.python import bits as ir_bits
from xls.ir.python import value as ir_value


class UnsupportedJitConversionError(Exception):
  """Raised when the JIT bindings throw an exception."""


class JitMiscompareError(Exception):
  """Raised when the JIT and DSLX interpreter give inconsistent results."""


def convert_interpreter_value_to_ir(
    interpreter_value: dslx_value.Value) -> ir_value.Value:
  """Recursively translates a DSLX Value into an IR Value."""
  if interpreter_value.is_bits() or interpreter_value.is_enum():
    return ir_value.Value(interpreter_value.get_bits())
  elif interpreter_value.is_array():
    ir_arr = []
    for e in interpreter_value.get_elements():
      ir_arr.append(convert_interpreter_value_to_ir(e))
    return ir_value.Value.make_array(ir_arr)
  elif interpreter_value.is_tuple():
    ir_tuple = []
    for e in interpreter_value.get_elements():
      ir_tuple.append(convert_interpreter_value_to_ir(e))
    return ir_value.Value.make_tuple(ir_tuple)
  else:
    raise UnsupportedJitConversionError(
        "Can't convert to JIT value: {}".format(interpreter_value))


def convert_args_to_ir(
    args: Iterable[dslx_value.Value]) -> Iterable[ir_value.Value]:
  ir_args = []
  for arg in args:
    ir_args.append(convert_interpreter_value_to_ir(arg))

  return ir_args


def bits_to_int(jit_bits: ir_bits.Bits, signed: bool) -> int:
  """Constructs the ir bits value by reading in a 64-bit value at a time."""
  assert isinstance(jit_bits, ir_bits.Bits), jit_bits
  bit_count = jit_bits.bit_count()
  bits_value = jit_bits.to_uint()

  return (bits_value if not signed else bit_helpers.from_twos_complement(
      bits_value, bit_count))


def compare_values(interpreter_value: dslx_value.Value,
                   jit_value: ir_value.Value) -> None:
  """Asserts equality between a DSLX Value and an IR Value.

  Recursively traverses the values (for arrays/tuples) and makes assertions
  about value and length properties.

  Args:
    interpreter_value: Value that resulted from DSL interpretation.
    jit_value: Value that resulted from JIT-compiled execution.

  Raises:
    JitMiscompareError: If the dslx_value and jit_value are not equivalent.
    UnsupportedJitConversionError: If there is not JIT-supported type equivalent
      for the interpreter value.
  """
  if interpreter_value.is_bits() or interpreter_value.is_enum():
    assert jit_value.is_bits(), f'Expected bits value: {jit_value!r}'

    jit_bits_value = jit_value.get_bits()
    assert isinstance(jit_bits_value, ir_bits.Bits), jit_bits_value
    bit_count = interpreter_value.get_bit_count()
    if bit_count != jit_bits_value.bit_count():
      raise JitMiscompareError(f'Inconsistent bit counts for value -- '
                               f'interp: {bit_count}, '
                               f'jit: {jit_bits_value.bit_count()}')

    interpreter_bits_value = interpreter_value.get_bits()
    if interpreter_bits_value != jit_bits_value:
      raise JitMiscompareError('Inconsistent bit values in return value -- '
                               'interp: {!r}, jit: {!r}'.format(
                                   interpreter_bits_value, jit_bits_value))

  elif interpreter_value.is_array():
    assert jit_value.is_array(), f'Expected array value: {jit_value!r}'

    interpreter_values = interpreter_value.get_elements()
    jit_values = jit_value.get_elements()
    interp_len = len(interpreter_values)
    jit_len = len(jit_values)
    if interp_len != jit_len:
      raise JitMiscompareError(f'Inconsistent array lengths in return value -- '
                               f'interp: {interp_len}, jit: {jit_len}')

    for interpreter_element, jit_element in zip(interpreter_values, jit_values):
      compare_values(interpreter_element, jit_element)
  elif interpreter_value.is_tuple():
    assert jit_value.is_tuple(), 'Expected tuple value: {jit_value!r}'

    interpreter_values = interpreter_value.get_elements()
    jit_values = jit_value.get_elements()
    interp_len = len(interpreter_values)
    jit_len = len(jit_values)
    if interp_len != jit_len:
      raise JitMiscompareError(f'Inconsistent tuple lengths in return value -- '
                               f'interp: {interp_len}, jit: {jit_len}')

    for interpreter_element, jit_element in zip(interpreter_values, jit_values):
      compare_values(interpreter_element, jit_element)
  else:
    raise UnsupportedJitConversionError(
        'No JIT-supported type equivalent: {}'.format(interpreter_value))


def ir_value_to_interpreter_value(value: ir_value.Value,
                                  dslx_type: ConcreteType) -> dslx_value.Value:
  """Converts an IR Value to an interpreter Value."""
  if value.is_bits():
    assert isinstance(dslx_type, BitsType), dslx_type
    ir_bits_val = value.get_bits()
    if dslx_type.get_signedness():
      return dslx_value.Value.make_sbits(ir_bits_val)
    return dslx_value.Value.make_ubits(ir_bits_val)
  elif value.is_array():
    assert isinstance(dslx_type, ArrayType), dslx_type
    return dslx_value.Value.make_array(
        tuple(
            ir_value_to_interpreter_value(e, dslx_type.element_type)
            for e in value.get_elements()))
  else:
    assert value.is_tuple()
    assert isinstance(dslx_type, TupleType), dslx_type
    return dslx_value.Value.make_tuple(
        tuple(
            ir_value_to_interpreter_value(e, t) for e, t in zip(
                value.get_elements(), dslx_type.get_unnamed_members())))
