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

"""Multi-process fuzz driver program.

Collects crash samples into a directory of the user's choosing.
"""

import os

from absl import app
from absl import flags
import psutil

from xls.common import gfile
from xls.common import multiprocess
from xls.fuzzer import cli_helpers
from xls.fuzzer import run_fuzz_multiprocess
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample

_DURATION = flags.DEFINE_string('duration', None,
                                'Duration to run the sample generator for.')
_CALLS_PER_SAMPLE = flags.DEFINE_integer('calls_per_sample', 128,
                                         'Arguments to generate per sample.')
_CRASH_PATH = flags.DEFINE_string('crash_path', None,
                                  'Path at which to place crash data.')
_CODEGEN = flags.DEFINE_boolean('codegen', False, 'Run code generation.')
_EMIT_LOOPS = flags.DEFINE_boolean('emit_loops', True,
                                   'Emit loops in generator.')
_FORCE_FAILURE = flags.DEFINE_bool(
    'force_failure', False,
    'Forces the samples to fail. Can be used to test failure code paths.')
_GENERATE_PROC = flags.DEFINE_boolean(
    'generate_proc', default=False, help='Generate a proc sample.')
_MAX_WIDTH_AGGREGATE_TYPES = flags.DEFINE_integer(
    'max_width_aggregate_types', 1024,
    'The maximum width of aggregate types (tuples and arrays) in the generated '
    'samples.')
_MAX_WIDTH_BITS_TYPES = flags.DEFINE_integer(
    'max_width_bits_types', 64,
    'The maximum width of bits types in the generated samples.')
_PROC_TICKS = flags.DEFINE_integer(
    'proc_ticks', 100, 'Number ticks to execute the generated procs.')
_SAMPLE_COUNT = flags.DEFINE_integer('sample_count', None,
                                     'Number of samples to generate.')
_SAVE_TEMPS_PATH = flags.DEFINE_string(
    'save_temps_path', None, 'Path of directory in which to save temporary '
    'files. These temporary files include DSLX, IR, and arguments. A '
    'separate numerically-named subdirectory is created for each sample.')
_SEED = flags.DEFINE_integer('seed', None, 'Seed value for generation')
_SIMULATE = flags.DEFINE_boolean('simulate', False, 'Run Verilog simulation.')
_SIMULATOR = flags.DEFINE_string(
    'simulator', None, 'Verilog simulator to use. For example: "iverilog".')
_SUMMARY_PATH = flags.DEFINE_string(
    'summary_path', None,
    'Directory in which to write the sample summary information. This records '
    'information about each generated sample including which XLS op types and '
    'widths. Information  is written in Protobuf format with one file per '
    'worker. Files are appended to by the worker.')
_TIMEOUT_SECONDS = flags.DEFINE_integer(
    'timeout_seconds', None,
    'The timeout value in seconds for each subcommand invocation. If not '
    'specified there is no timeout.')
_USE_LLVM_JIT = flags.DEFINE_boolean(
    'use_llvm_jit', True, 'Use LLVM JIT to evaluate IR. The interpreter is '
    'still invoked at least once on the IR even with this option enabled, but '
    'this option can be used to disable the JIT entirely.')
_USE_SYSTEM_VERILOG = flags.DEFINE_boolean(
    'use_system_verilog', True,
    'If true, emit SystemVerilog during codegen otherwise emit Verilog.')
_WORKER_COUNT = flags.DEFINE_integer(
    'worker_count', None, 'Number of workers to use for execution; defaults '
    'to number of physical cores detected.')

QUEUE_MAX_BACKLOG = 16


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _SIMULATE.value and not _CODEGEN.value:
    raise app.UsageError('Must specify --codegen when --simulate is given.')

  # Test that we can write to the crash and summary path.
  for path in (_CRASH_PATH.value, _SUMMARY_PATH.value):
    if path:
      gfile.make_dirs(path)
      with gfile.open(os.path.join(path, 'test'), 'w') as f:
        print('test', file=f)

  physical_core_count = psutil.cpu_count(logical=False)
  worker_count = _WORKER_COUNT.value or physical_core_count
  worker_count = max(worker_count, 1)  # Need at least one worker.

  duration_str = _DURATION.value
  duration = None if duration_str is None else cli_helpers.parse_duration(
      duration_str)

  generator_options = ast_generator.AstGeneratorOptions(
      emit_gate=not _CODEGEN.value,
      emit_loops=_EMIT_LOOPS.value,
      max_width_bits_types=_MAX_WIDTH_BITS_TYPES.value,
      max_width_aggregate_types=_MAX_WIDTH_AGGREGATE_TYPES.value,
      generate_proc=_GENERATE_PROC.value)

  sample_options = sample.SampleOptions(
      calls_per_sample=0 if _GENERATE_PROC.value else _CALLS_PER_SAMPLE.value,
      codegen=_CODEGEN.value,
      convert_to_ir=True,
      input_is_dslx=True,
      ir_converter_args=['--top=main'],
      optimize_ir=True,
      proc_ticks=_PROC_TICKS.value if _GENERATE_PROC.value else 0,
      simulate=_SIMULATE.value,
      simulator=_SIMULATOR.value,
      timeout_seconds=_TIMEOUT_SECONDS.value,
      use_jit=_USE_LLVM_JIT.value,
      use_system_verilog=_USE_SYSTEM_VERILOG.value)

  run_fuzz_multiprocess.parallel_generate_and_run_samples(
      worker_count,
      generator_options,
      sample_options=sample_options,
      seed=_SEED.value,
      top_run_dir=_SAVE_TEMPS_PATH.value,
      crasher_dir=_CRASH_PATH.value,
      summary_dir=_SUMMARY_PATH.value,
      sample_count=_SAMPLE_COUNT.value,
      duration=duration,
      force_failure=_FORCE_FAILURE.value)


if __name__ == '__main__':

  def real_main():  # Avoid defining things in global scope.
    flags.mark_flag_as_required('crash_path')
    multiprocess.run_main(main, None)

  real_main()
