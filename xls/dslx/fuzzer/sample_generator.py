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

"""Sample generator for fuzzing."""

import random
from typing import Tuple, Text, Sequence

from xls.dslx import parse_and_typecheck
from xls.dslx.concrete_type import ArrayType
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import TupleType
from xls.dslx.fuzzer import ast_generator
from xls.dslx.fuzzer import sample
from xls.dslx.interpreter.value import Value

Random = random.Random
del random  # Avoids accidentally calling module functions, use object methods.


def _generate_bit_value(bit_count: int, rng: Random, signed: bool) -> Value:
  p = rng.random()
  if p < 0.9:  # Most of the time, use some interesting bit pattern.
    value = rng.choice(ast_generator.make_bit_patterns(bit_count))
  else:  # 10% of the time use a completely random value.
    value = rng.randrange(0, 2**bit_count)
  constructor = Value.make_sbits if signed else Value.make_ubits
  return constructor(value=value, bit_count=bit_count)


def _generate_unbiased_argument(concrete_type: ConcreteType,
                                rng: Random) -> Value:
  if isinstance(concrete_type, BitsType):
    bit_count = concrete_type.get_total_bit_count()
    return _generate_bit_value(bit_count, rng, concrete_type.get_signedness())
  else:
    raise NotImplementedError(
        'Generate argument for type: {}'.format(concrete_type))


def randrange_biased_towards_zero(limit: int, rng: Random) -> int:
  return int(rng.triangular(0, limit - 1, 0))


def generate_argument(arg_type: ConcreteType, rng: Random,
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
            for _ in range(arg_type.size)))
  else:
    assert isinstance(arg_type, BitsType)
    if not prior or rng.random() < 0.5:
      return _generate_unbiased_argument(arg_type, rng)

  to_mutate = rng.choice(prior)
  bit_count = arg_type.get_total_bit_count()
  if bit_count > to_mutate.get_bit_count():
    to_mutate = to_mutate.bits_payload.concat(
        _generate_bit_value(
            bit_count - to_mutate.get_bit_count(), rng,
            signed=False).bits_payload)
  else:
    to_mutate = to_mutate.bits_payload.slice(0, bit_count, lsb_is_0=False)

  assert to_mutate.bit_count == bit_count
  value = to_mutate.value
  mutation_count = randrange_biased_towards_zero(bit_count, rng)
  for _ in range(mutation_count):
    # Pick a random bit and flip it.
    bitno = rng.randrange(bit_count)
    value ^= 1 << bitno

  signed = arg_type.get_signedness()
  constructor = Value.make_sbits if signed else Value.make_ubits
  return constructor(value=value, bit_count=bit_count)


def generate_arguments(arg_types: Sequence[ConcreteType],
                       rng: Random) -> Tuple[Value, ...]:
  """Returns a tuple of randomly generated values of the given types."""
  args = []
  for arg_type in arg_types:
    args.append(generate_argument(arg_type, rng, args))
  return tuple(args)


def generate_codegen_args(use_system_verilog: bool,
                          rng: Random) -> Tuple[Text, ...]:
  """Returns randomly generated arguments for running codegen.

  These arguments are flags which are passed to codegen_main for generating
  Verilog. Randomly chooses either a purely combinational module or a
  feed-forward pipeline of a randome length.

  Args:
    use_system_verilog: Whether to use SystemVerilog.
    rng: Random number generator.

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
        '--generator=pipeline', '--pipeline_stages=' + str(rng.randint(1, 10))
    ])
  return tuple(args)


def generate_sample(rng: Random, ast_options: ast_generator.AstGeneratorOptions,
                    calls_per_sample: int,
                    default_options: sample.SampleOptions) -> sample.Sample:
  """Generates and returns a random Sample with the given options."""
  ast_gen = ast_generator.AstGenerator(
      rng, ast_options, codegen_ops_only=default_options.codegen)
  f, m = ast_gen.generate_function_in_module(fname='main', mname='test')
  dslx_text = m.format()

  # Re-parse so we can get real positions in error messages.
  m, node_to_type = parse_and_typecheck.parse_text_fakefs(
      dslx_text,
      m.name,
      print_on_error=True,
      f_import=None,
      filename='/fake/test.x')
  f = m.get_function('main')

  arg_types = tuple(node_to_type[p.type_] for p in f.params)
  args_batch = tuple(
      generate_arguments(arg_types, rng) for _ in range(calls_per_sample))
  # The generated sample is DSLX so input_is_dslx must be true.
  options = default_options._replace(input_is_dslx=True)
  if options.codegen and not options.codegen_args:
    # Generate codegen args if codegen is given but no codegen args specified.
    options = options._replace(
        codegen_args=generate_codegen_args(options.use_system_verilog, rng))

  return sample.Sample(dslx_text, options, args_batch)
