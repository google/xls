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

"""Benchmarks the execution of a generated sample from the fuzzer."""

import tempfile
import timeit
from typing import Text

from absl import app
from absl import flags

from xls.fuzzer import run_fuzz
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample
from xls.fuzzer.run_fuzz_multiprocess import do_generator_task

flags.DEFINE_enum(
    'mode',
    'worker',
    enum_values=('worker', 'generator'),
    help='Which portion of the binary to benchmark.')
flags.DEFINE_boolean('codegen', False,
                     'Whether to run code generation after optimization.')
flags.DEFINE_boolean('simulate', False,
                     'Whether to simulate the generated Verilog.')
flags.DEFINE_boolean('execute', True, 'Whether to run DSLX/IR interpretation.')
FLAGS = flags.FLAGS

CALLS_PER_SAMPLE = 1024
DISALLOW_DIVIDE = True


def setup_worker():
  """Creates arguments to repeatedly pass to benchmark_worker."""
  rng = ast_generator.RngState(0)
  smp = ast_generator.generate_sample(
      ast_generator.AstGeneratorOptions(disallow_divide=DISALLOW_DIVIDE),
      CALLS_PER_SAMPLE,
      sample.SampleOptions(
          convert_to_ir=True,
          optimize_ir=True,
          codegen=FLAGS.codegen,
          simulate=FLAGS.simulate), rng)

  run_dir = tempfile.mkdtemp('run_fuzz_')
  return (run_dir, smp)


def benchmark_worker(run_dir: Text, smp: sample.Sample):
  print('Running!')
  run_fuzz.run_sample(smp, run_dir)


class _DummyQueue(object):

  def qsize(self) -> int:
    return 0

  def put_nowait(self, message) -> None:
    pass

  def put(self, message) -> None:
    pass


def setup_generator():
  return dict(
      queues=(_DummyQueue(),),
      seed=0,
      ast_generator_options=ast_generator.AstGeneratorOptions(
          disallow_divide=DISALLOW_DIVIDE),
      sample_count=16,
      duration=None,
      calls_per_sample=CALLS_PER_SAMPLE,
      print_samples=False)


def benchmark_generator(**kwargs):
  do_generator_task(**kwargs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.mode == 'generator':
    print(
        timeit.timeit(
            'benchmark_generator(**kwargs)',
            setup='from __main__ import setup_generator, benchmark_generator; '
            'kwargs = setup_generator()',
            number=20))
  else:
    print(
        timeit.timeit(
            'benchmark_worker(*args)',
            setup='from __main__ import setup_worker, benchmark_worker; '
            'args = setup_worker()',
            number=20))

if __name__ == '__main__':
  app.run(main)
