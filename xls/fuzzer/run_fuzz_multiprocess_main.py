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

"""Multi-process fuzz driver program.

Collects crash samples into a directory of the user's choosing.
"""

import datetime
import multiprocessing as mp
import os
import random
import sys

from absl import app
from absl import flags
import psutil

from xls.common import gfile
from xls.common import multiprocess
from xls.fuzzer import cli_helpers
from xls.fuzzer import run_fuzz_multiprocess
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample

flags.DEFINE_integer('seed', 0, 'Seed value for generation')
flags.DEFINE_integer('sample_count', 1024, 'Number of samples to generate')
flags.DEFINE_string('duration', None,
                    'Duration to run the sample generator for')
flags.DEFINE_integer('calls_per_sample', 128,
                     'Arguments to generate per sample')
flags.DEFINE_string('crash_path', None, 'Path at which to place crash data')
flags.DEFINE_string(
    'save_temps_path', None, 'Path of directory in which to save temporary '
    'files. These temporary files include DSLX, IR, and arguments. A '
    'separate numerically-named subdirectory is created for each sample')
flags.DEFINE_integer(
    'worker_count', None, 'Number of workers to use for execution; defaults '
    'to number of physical cores detected')
flags.DEFINE_boolean('disallow_divide', True,
                     'Exclude generation of divide operator')
flags.DEFINE_boolean('emit_loops', True, 'Emit loops in generator')
flags.DEFINE_boolean(
    'use_llvm_jit', True, 'Use LLVM JIT to evaluate IR. The interpreter is '
    'still invoked at least once on the IR even with this option enabled, but '
    'this option can be used to disable the JIT entirely.')
flags.DEFINE_boolean('codegen', False, 'Run code generation')
flags.DEFINE_boolean('simulate', False, 'Run Verilog simulation.')
flags.DEFINE_string('simulator', None,
                    'Verilog simulator to use. For example: "iverilog".')
flags.DEFINE_boolean('execute', True, 'Execute IR (vs simply code generation)')
flags.DEFINE_boolean(
    'minimize_ir', True,
    'If a crasher is found, attempt to reduce the IR to find a minimal '
    'reproducer.')
flags.DEFINE_boolean('print_samples', False,
                     'Print generated samples (to stdout)')
flags.DEFINE_boolean(
    'short_samples', False,
    'Generate samples with small number of nested expressions')
flags.DEFINE_string(
    'summary_path', None,
    'Directory in which to write the sample summary information. This records '
    'information about each generated sample including which XLS op types and '
    'widths. Information  is written in Protobuf format with one file per '
    'worker. Files are appended to by the worker.')
flags.DEFINE_integer(
    'max_width_bits_types', 64,
    'The maximum width of bits types in the generated samples.')
flags.DEFINE_integer(
    'max_width_aggregate_types', 1024,
    'The maximum width of aggregate types (tuples and arrays) in the generated '
    'samples.')
flags.DEFINE_boolean(
    'use_system_verilog', True,
    'If true, emit SystemVerilog during codegen otherwise emit Verilog.')
FLAGS = flags.FLAGS

QUEUE_MAX_BACKLOG = 16


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.simulate and not FLAGS.codegen:
    raise app.UsageError('Must specify --codegen when --simulate is given.')

  # Test that we can write to the crash and summary path.
  for path in (FLAGS.crash_path, FLAGS.summary_path):
    if path:
      gfile.make_dirs(path)
      with gfile.open(os.path.join(path, 'test'), 'w') as f:
        print('test', file=f)

  start = datetime.datetime.now()

  physical_core_count = psutil.cpu_count(logical=False)
  worker_count = FLAGS.worker_count or physical_core_count
  worker_count = max(worker_count, 1)  # Need at least one worker.
  queues = (multiprocess.get_user_data() or
            [mp.Queue() for _ in range(worker_count)])
  queues = queues[:worker_count]
  print('-- Creating pool of {} workers; physical core count {}'.format(
      worker_count, physical_core_count))
  workers = []
  for i in range(worker_count):
    queue = None if multiprocess.has_user_data_support() else queues[i]

    target = run_fuzz_multiprocess.do_worker_task
    args = (i, queue, FLAGS.crash_path, FLAGS.summary_path,
            FLAGS.save_temps_path, FLAGS.minimize_ir)

    worker = multiprocess.Process(target=target, args=args)

    worker.start()
    workers.append(worker)

  duration_str = FLAGS.duration
  duration = None if duration_str is None else cli_helpers.parse_duration(
      duration_str)

  seed = FLAGS.seed
  if not seed:
    seed = random.randrange(0, 1 << 31)
    print('-- Using randomly generated seed:', seed)
    sys.stdout.flush()

  generator_options = ast_generator.AstGeneratorOptions(
      disallow_divide=FLAGS.disallow_divide,
      emit_loops=FLAGS.emit_loops,
      short_samples=FLAGS.short_samples,
      max_width_bits_types=FLAGS.max_width_bits_types,
      max_width_aggregate_types=FLAGS.max_width_aggregate_types)

  default_sample_options = sample.SampleOptions(
      convert_to_ir=True,
      optimize_ir=True,
      use_jit=FLAGS.use_llvm_jit,
      codegen=FLAGS.codegen,
      simulate=FLAGS.simulate,
      simulator=FLAGS.simulator,
      use_system_verilog=FLAGS.use_system_verilog)
  sample_count = run_fuzz_multiprocess.do_generator_task(
      queues,
      seed,
      generator_options,
      FLAGS.sample_count,
      FLAGS.calls_per_sample,
      default_sample_options=default_sample_options,
      duration=duration,
      print_samples=FLAGS.print_samples)

  for i, worker in enumerate(workers):
    print('-- Joining on worker {}'.format(i))
    worker.join()

  delta = datetime.datetime.now() - start
  elapsed = delta.total_seconds()
  print(
      '-- Elapsed end-to-end: {} = {:.2f} seconds; {:,} samples; {:.2f} samples/s'
      .format(delta, elapsed, sample_count, sample_count / elapsed))


if __name__ == '__main__':

  def real_main():  # Avoid defining things in global scope.
    flags.mark_flag_as_required('crash_path')
    queues = tuple(mp.Queue(QUEUE_MAX_BACKLOG) for _ in range(128))
    multiprocess.run_main(main, queues)

  real_main()
