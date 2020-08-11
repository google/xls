# Lint as: python3
#
# Copyright 2020 Google LLC
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

from absl import logging

from xls.dslx import bit_helpers
from xls.dslx.concrete_type import ArrayType
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import TupleType
from xls.dslx.interpreter import value as dslx_value
from xls.ir.python import bits as ir_bits
from xls.ir.python import number_parser
from xls.ir.python import value as ir_value

WORD_SIZE = 64  # type: int


class UnsupportedJitConversionError(Exception):
  """Raised when the JIT bindings throw an exception."""

class JitMiscompareError(Exception):
  """Raised when the JIT and DSLX interpreter give inconsistent results."""


def convert_interpreter_value_to_ir(
    interpreter_value: dslx_value.Value) -> ir_value.Value:
  """Recursively translates a DSLX Value into an IR Value."""
  if interpreter_value.is_bits() or interpreter_value.is_enum():
    return ir_value.Value(
        int_to_bits(interpreter_value.get_bits_value_check_sign(),
                    interpreter_value.get_bit_count()))
  elif interpreter_value.is_array():
    ir_arr = []
    for e in interpreter_value.array_payload.elements:
      ir_arr.append(convert_interpreter_value_to_ir(e))
    return ir_value.Value.make_array(ir_arr)
  elif interpreter_value.is_tuple():
    ir_tuple = []
    for e in interpreter_value.tuple_members:
      ir_tuple.append(convert_interpreter_value_to_ir(e))
    return ir_value.Value.make_tuple(ir_tuple)
  else:
    raise UnsupportedJitConversionError("Can't convert to JIT value: {}"
                                        .format(interpreter_value))


def convert_args_to_ir(
    args: Iterable[dslx_value.Value]) -> Iterable[ir_value.Value]:
  ir_args = []
  for arg in args:
    ir_args.append(convert_interpreter_value_to_ir(arg))

  return ir_args


def bits_to_int(jit_bits: ir_bits.Bits, signed: bool) -> int:
  """Constructs the ir bits value by reading in a 64-bit value at a time."""
  bit_count = jit_bits.bit_count()
  bits_value = 0
  word_number = 0
  while (word_number * 64) < bit_count:
    word_value = jit_bits.word_to_uint(word_number)
    bits_value = (word_value << (word_number * WORD_SIZE)) | bits_value
    word_number += 1

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
    UnsupportedJitConversionError: If there is not JIT-supported type equivalent
      for the interpreter value.
  """
  if interpreter_value.is_bits() or interpreter_value.is_enum():
    assert jit_value.is_bits(), f'Expected bits value: {jit_value!r}'

    jit_value = jit_value.get_bits()
    bit_count = interpreter_value.get_bit_count()
    if bit_count != jit_value.bit_count():
      raise JitMiscompareError(f'Inconsistent bit counts for value -- '
                               f'interp: {bit_count}, '
                               f'jit: {jit_value.bit_count()}')

    if interpreter_value.is_ubits():
      interpreter_bits_value = interpreter_value.get_bits_value()
      jit_bits_value = bits_to_int(jit_value, signed=False)
    else:
      interpreter_bits_value = interpreter_value.get_bits_value_signed()
      jit_bits_value = bits_to_int(jit_value, signed=True)

    if interpreter_bits_value != jit_bits_value:
      raise JitMiscompareError('Inconsistent bit values in return value -- '
                               'interp: {!r}, jit: {!r}'.format(
                               interpreter_bits_value, jit_bits_value))

  elif interpreter_value.is_array():
    assert jit_value.is_array(), f'Expected array value: {jit_value!r}'

    interpreter_values = interpreter_value.array_payload.elements
    jit_values = jit_value.get_elements()
    interp_len = len(interpreter_values)
    jit_len = len(jit_values)
    if interp_len != jit_len:
        raise JitMiscompareError(
            f'Inconsistent array lengths in return value -- '
            f'interp: {interp_len}, jit: {jit_len}')

    for interpreter_element, jit_element in zip(interpreter_values, jit_values):
      compare_values(interpreter_element, jit_element)
  elif interpreter_value.is_tuple():
    assert jit_value.is_tuple(), 'Expected tuple value: {jit_value!r}'

    interpreter_values = interpreter_value.tuple_members
    jit_values = jit_value.get_elements()
    interp_len = len(interpreter_values)
    jit_len = len(jit_values)
    if interp_len != jit_len:
        raise JitMiscompareError(
            f'Inconsistent tuple lengths in return value -- '
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
      return dslx_value.Value.make_sbits(ir_bits_val.bit_count(),
                                         bits_to_int(ir_bits_val, signed=True))
    return dslx_value.Value.make_ubits(ir_bits_val.bit_count(),
                                       bits_to_int(ir_bits_val, signed=False))
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
            ir_value_to_interpreter_value(e, t)
            for e, t in zip(value.get_elements(), t.get_unnamed_members())))


def int_to_bits(value: int, bit_count: int) -> ir_bits.Bits:
  """Converts a Python arbitrary precision int to a Bits type."""
  if bit_count <= WORD_SIZE:
    return ir_bits.UBits(value, bit_count) if value >= 0 else ir_bits.SBits(
        value, bit_count)
  return number_parser.bits_from_string(
      bit_helpers.to_hex_string(value, bit_count), bit_count=bit_count)
