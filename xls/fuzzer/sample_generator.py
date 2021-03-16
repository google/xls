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

"""Sample generator for fuzzing."""

from typing import Tuple, Text, Sequence

from xls.dslx.python import interpreter
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.interp_value import Tag
from xls.dslx.python.interp_value import Value
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample
from xls.ir.python import bits as ir_bits


def _generate_bit_value(bit_count: int, rng: ast_generator.RngState,
                        signed: bool) -> Value:
  assert isinstance(bit_count, int), bit_count
  assert isinstance(rng, ast_generator.RngState), rng
  bits = ast_generator.choose_bit_pattern(bit_count, rng)
  tag = Tag.SBITS if signed else Tag.UBITS
  return Value.make_bits(tag, bits)


def _generate_unbiased_argument(concrete_type: ConcreteType,
                                rng: ast_generator.RngState) -> Value:
  if isinstance(concrete_type, BitsType):
    bit_count = concrete_type.get_total_bit_count().value
    return _generate_bit_value(bit_count, rng, concrete_type.get_signedness())
  else:
    raise NotImplementedError(
        'Generate argument for type: {}'.format(concrete_type))


def generate_argument(arg_type: ConcreteType, rng: ast_generator.RngState,
                      prior: Sequence[Value]) -> Value:
  """Generates an argument value of the same type as the concrete type."""
  if isinstance(arg_type, TupleType):
    return Value.make_tuple(
        tuple(generate_argument(t, rng, prior)
              for t in arg_type.get_unnamed_members()))
  elif isinstance(arg_type, ArrayType):
    return Value.make_array(
        tuple(
            generate_argument(arg_type.get_element_type(), rng, prior)
            for _ in range(arg_type.size.value)))
  else:
    assert isinstance(arg_type, BitsType)
    if not prior or rng.random() < 0.5:
      return _generate_unbiased_argument(arg_type, rng)

  # Try to mutate a prior argument. If it happens to not be a bits type then
  # just generate an unbiased argument.
  index = rng.randrange(len(prior))
  if not prior[index].is_bits():
    return _generate_unbiased_argument(arg_type, rng)

  to_mutate = prior[index].get_bits()
  bit_count = arg_type.get_total_bit_count().value
  if bit_count > to_mutate.bit_count():
    addendum = _generate_bit_value(
        bit_count - to_mutate.bit_count(), rng, signed=False)
    assert addendum.get_bit_count() + to_mutate.bit_count() == bit_count
    to_mutate = to_mutate.concat(addendum.get_bits())
  else:
    to_mutate = to_mutate.slice(0, bit_count)

  assert to_mutate.bit_count() == bit_count, (to_mutate.bit_count(), bit_count)
  value = to_mutate.to_uint()
  mutation_count = rng.randrange_biased_towards_zero(bit_count)
  for _ in range(mutation_count):
    # Pick a random bit and flip it.
    bitno = rng.randrange(bit_count)
    value ^= 1 << bitno

  signed = arg_type.get_signedness()
  tag = Tag.SBITS if signed else Tag.UBITS
  return Value.make_bits(tag,
                         ir_bits.from_long(value=value, bit_count=bit_count))


def generate_arguments(arg_types: Sequence[ConcreteType],
                       rng: ast_generator.RngState) -> Tuple[Value, ...]:
  """Returns a tuple of randomly generated values of the given types."""
  args = []
  for arg_type in arg_types:
    args.append(generate_argument(arg_type, rng, args))
  return tuple(args)


def generate_codegen_args(use_system_verilog: bool,
                          rng: ast_generator.RngState) -> Tuple[Text, ...]:
  """Returns randomly generated arguments for running codegen.

  These arguments are flags which are passed to codegen_main for generating
  Verilog. Randomly chooses either a purely combinational module or a
  feed-forward pipeline of a randome length.

  Args:
    use_system_verilog: Whether to use SystemVerilog.
    rng: Random number generator state.

  Returns:
    Tuple of arguments to pass to codegen_main.
  """
  args = []
  if use_system_verilog:
    args.append('--use_system_verilog')
  else:
    args.append('--nouse_system_verilog')
  if rng.random() < 0.2:
    args.append('--generator=combinational')
  else:
    args.extend([
        '--generator=pipeline',
        '--pipeline_stages=' + str(rng.randrange(10) + 1)
    ])
  return tuple(args)


def generate_sample(rng: ast_generator.RngState,
                    ast_options: ast_generator.AstGeneratorOptions,
                    calls_per_sample: int,
                    default_options: sample.SampleOptions) -> sample.Sample:
  """Generates and returns a random Sample with the given options."""
  assert isinstance(rng, ast_generator.RngState), rng
  dslx_text = ast_generator.generate(ast_options, rng)

  # Note: we also re-parse here so we can get real positions in error messages.
  fn_type = interpreter.get_function_type(dslx_text, 'main')

  args_batch = tuple(
      generate_arguments(fn_type.params, rng) for _ in range(calls_per_sample))
  # The generated sample is DSLX so input_is_dslx must be true.
  options = default_options.replace(input_is_dslx=True)
  if options.codegen and not options.codegen_args:
    # Generate codegen args if codegen is given but no codegen args specified.
    options = options.replace(
        codegen_args=generate_codegen_args(options.use_system_verilog, rng))

  return sample.Sample(dslx_text, options, args_batch)
