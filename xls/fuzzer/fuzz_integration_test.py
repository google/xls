#
# Copyright 2022 The XLS Authors
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
"""Tests for xls.fuzzer.run_fuzz."""

import datetime
import os
import random
import sys
import tempfile

from absl import flags
from absl import logging

from absl.testing import absltest
from xls.fuzzer import cli_helpers
from xls.fuzzer import run_fuzz
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python.cpp_sample import SampleOptions

_USE_NONDETERMINISTIC_SEED = flags.DEFINE_bool(
    'use_nondeterministic_seed', False,
    'Use a non-deterministic seed for the random number generator. '
    'If specified, the flag --seed is ignored')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed value for generation')
_SAMPLE_COUNT = flags.DEFINE_integer('sample_count', 10,
                                     'Number of samples to generate')
_DURATION = flags.DEFINE_string(
    'duration', None, 'Duration to run the fuzzer for. '
    'Examples: 60s, 30m, 5h')
_CALLS_PER_SAMPLE = flags.DEFINE_integer(
    'calls_per_sample', 128,
    'Arguments to generate per sample. The value is valid when '
    '`--generate_proc` is `false`.')
_PROC_TICKS = flags.DEFINE_integer(
    'proc_ticks', 128,
    'The number of ticks to execute when generating a proc. The value is valid '
    'when `--generate_proc` is `true`.')
_SIMULATE = flags.DEFINE_boolean('simulate', False, 'Run Verilog simulation.')
_SIMULATOR = flags.DEFINE_string(
    'simulator', None, 'Verilog simulator to use. For example: "iverilog".')
_MAX_WIDTH_BITS_TYPES = flags.DEFINE_integer(
    'max_width_bits_types', 64,
    'The maximum width of bits types in the generated samples.')
_MAX_WIDTH_AGGREGATE_TYPES = flags.DEFINE_integer(
    'max_width_aggregate_types', 1024,
    'The maximum width of aggregate types (tuples and arrays) in the generated '
    'samples.')
_FORCE_FAILURE = flags.DEFINE_bool(
    'force_failure', False,
    'Forces the samples to fail. Can be used to test failure code paths.')
_USE_SYSTEM_VERILOG = flags.DEFINE_bool(
    'use_system_verilog', False,
    'Whether to generate SystemVerilog or Verilog.')
_SAMPLE_TIMEOUT = flags.DEFINE_string(
    'sample_timeout', None,
    'Maximum time to run each sample before timing out. '
    'Examples: 10s, 2m, 1h')
_GENERATE_PROC = flags.DEFINE_boolean(
    'generate_proc', default=False, help='Generate a proc sample.')

# The maximum number of failures before the test aborts.
MAX_FAILURES = 10


def _get_crasher_directory() -> str:
  """Returns the directory in which to write crashers.

  Crashers are written to the undeclared outputs directory, if it is
  available. Otherwise a temporary directory is created.
  """
  if 'TEST_UNDECLARED_OUTPUTS_DIR' in os.environ:
    crasher_dir = os.path.join(os.environ['TEST_UNDECLARED_OUTPUTS_DIR'],
                               'crashers')
    os.mkdir(crasher_dir)
    return crasher_dir
  else:
    return tempfile.mkdtemp(prefix='run_fuzz_crashers')


class FuzzIntegrationTest(absltest.TestCase):

  def test(self):
    """Runs the fuzzer based on flag values."""
    crasher_dir = _get_crasher_directory()
    if _USE_NONDETERMINISTIC_SEED.value:
      seed = random.randrange(sys.maxsize)
      logging.info('Random seed (generated nondeterministically): %s', seed)
    else:
      seed = _SEED.value
      logging.info('Random seed specified via flag: %s', seed)

    rng = ast_generator.ValueGenerator(seed)
    generator_options = ast_generator.AstGeneratorOptions(
        emit_gate=not _SIMULATE.value,
        generate_proc=_GENERATE_PROC.value,
        max_width_bits_types=_MAX_WIDTH_BITS_TYPES.value,
        max_width_aggregate_types=_MAX_WIDTH_AGGREGATE_TYPES.value)

    if _SAMPLE_TIMEOUT.value is None:
      timeout_seconds = None
    else:
      timeout_seconds = cli_helpers.parse_duration(
          _SAMPLE_TIMEOUT.value).seconds

    sample_options = SampleOptions(
        input_is_dslx=True,
        ir_converter_args=['--top=main'],
        convert_to_ir=True,
        calls_per_sample=0 if _GENERATE_PROC.value else _CALLS_PER_SAMPLE.value,
        proc_ticks=_PROC_TICKS.value if _GENERATE_PROC.value else 0,
        use_jit=True,
        codegen=_SIMULATE.value,
        simulate=_SIMULATE.value,
        simulator=_SIMULATOR.value,
        use_system_verilog=_USE_SYSTEM_VERILOG.value,
        timeout_seconds=timeout_seconds)

    crasher_count = 0
    sample_count = 0

    def now():
      # Add a timezone so duration is accurately tracked across daylight savings
      # time changes.
      return datetime.datetime.now(tz=datetime.timezone.utc)

    start = now()
    duration = None if _DURATION.value is None else cli_helpers.parse_duration(
        _DURATION.value)

    def keep_going() -> bool:
      if duration is None:
        done = sample_count >= _SAMPLE_COUNT.value
        if done:
          logging.info('Generated target number of samples. Exiting.')
      else:
        done = now() - start >= duration
        if done:
          logging.info('Ran for target duration of %s. Exiting.', duration)
      return not done

    while keep_going():
      logging.info('Running sample %d', sample_count)
      sample_count += 1
      with tempfile.TemporaryDirectory(prefix='run_fuzz_') as run_dir:
        try:
          run_fuzz.generate_sample_and_run(
              rng,
              generator_options,
              sample_options,
              run_dir,
              crasher_dir=crasher_dir,
              force_failure=_FORCE_FAILURE.value)

        except sample_runner.SampleError as e:
          logging.error('Sample failed: %s', str(e))
          crasher_count += 1

      if crasher_count >= MAX_FAILURES:
        break

    if crasher_count > 0:
      if crasher_count == MAX_FAILURES:
        msg = f'Fuzzing stopped after finding {crasher_count} failures'
      else:
        msg = f'Fuzzing found {crasher_count} failures'
      self.fail(
          f'{msg}. Generated {sample_count} total samples [seed = {seed}]')


if __name__ == '__main__':
  absltest.main()
