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

"""Multi-process fuzz driver library."""

import dataclasses
import datetime
import itertools
import os
import random
import shutil
import sys
import tempfile
from typing import Optional

import termcolor

from xls.common import gfile
from xls.common import multiprocess
from xls.fuzzer import run_fuzz
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample


@dataclasses.dataclass
class WorkerConfig:
  """Configuration for a fuzzer worker process."""
  worker_number: int
  ast_generator_options: ast_generator.AstGeneratorOptions
  sample_options: sample.SampleOptions
  crasher_dir: str
  seed: Optional[int]
  top_run_dir: Optional[str]
  summary_dir: Optional[str]
  sample_count: Optional[int]
  duration: Optional[datetime.timedelta]
  force_failure: bool


def _do_worker_task(config: WorkerConfig):
  """Runs worker task, generating and running samples."""
  crashers = 0
  print('--- Started worker {}'.format(config.worker_number))
  sys.stdout.flush()
  start = datetime.datetime.now()

  # Local file to write the summary information to before writing out to the
  # potentially remote (i.e. CNS) summary file. Avoids a potential CNS write
  # with every sample. Instead data is written out in batches.
  if config.summary_dir:
    summary_file = os.path.join(config.summary_dir,
                                'summary_%d.binarypb' % config.worker_number)
    summary_temp_file = tempfile.mkstemp(prefix='temp_summary_')[1]
  else:
    summary_file = None
    summary_temp_file = None

  if config.seed:
    # Set seed deterministically based on the worker number so different workers
    # generate different samples.
    rng = ast_generator.ValueGenerator(config.seed + config.worker_number)
  else:
    # Chose a nondeterministic seed.
    rng = ast_generator.ValueGenerator(random.randrange(0, 1 << 31))

  i = 0  # Silence pylint warning.
  for i in itertools.count():
    if config.sample_count is not None and i == config.sample_count:
      print('--- Worker {:3}: Ran {:5} samples. Exiting.'.format(
          config.worker_number, config.sample_count))
      break

    run_dir = None
    if config.top_run_dir:
      run_dir = os.path.join(config.top_run_dir,
                             'worker%d-sample%d' % (config.worker_number, i))
      os.makedirs(run_dir)
    else:
      run_dir = tempfile.mkdtemp(prefix='run_fuzz_')

    try:
      run_fuzz.generate_sample_and_run(
          rng,
          config.ast_generator_options,
          config.sample_options,
          run_dir,
          crasher_dir=config.crasher_dir,
          summary_file=summary_temp_file,
          force_failure=config.force_failure)
    except sample_runner.SampleError:
      termcolor.cprint(
          '--- Worker {} noted crasher #{} for sample number {}'.format(
              config.worker_number, crashers, i),
          color='red')
      crashers += 1

    if summary_file and i % 25 == 0:
      # Append the local temporary summary file to the actual, potentially
      # remote one, and delete the temporary file.
      with gfile.open(summary_temp_file, 'rb') as f:
        summaries = f.read()
      with gfile.open(summary_file, 'ab+') as f:
        f.write(summaries)
      gfile.remove(summary_temp_file)

    if not config.top_run_dir:
      # Remove temporary run directory.
      shutil.rmtree(run_dir)

    elapsed = (datetime.datetime.now() - start)
    if i != 0 and i % 16 == 0:
      metrics = []
      if config.sample_count is not None:
        metrics.append('{}/{} samples'.format(i, config.sample_count))
      else:
        metrics.append('{} samples'.format(i))
      metrics.append(' {:.2f} samples/s'.format(i / elapsed.total_seconds()))
      if config.duration is not None:
        metrics.append('running for {} (limit {})'.format(
            elapsed, config.duration))
      else:
        metrics.append('running for {}'.format(elapsed))

      print('--- Worker {:3}: {}'.format(config.worker_number,
                                         ', '.join(metrics)))
      sys.stdout.flush()

    if (config.duration is not None and elapsed >= config.duration):
      print('--- Worker {:3}: Ran for {}. Exiting.'.format(
          config.worker_number, config.duration))
      break

  elapsed = (datetime.datetime.now() - start).total_seconds()
  print(
      '--- Worker {:3} finished! {} samples; {} crashers; {:.2f} samples/s; ran for {}'
      .format(config.worker_number, i, crashers, i / elapsed, elapsed))
  sys.stdout.flush()


def parallel_generate_and_run_samples(
    worker_count: int,
    ast_generator_options: ast_generator.AstGeneratorOptions,
    sample_options: sample.SampleOptions,
    seed: Optional[int] = None,
    top_run_dir: Optional[str] = None,
    crasher_dir: Optional[str] = None,
    summary_dir: Optional[str] = None,
    sample_count: Optional[int] = None,
    duration: Optional[datetime.timedelta] = None,
    force_failure: bool = False):
  """Generate and run fuzzer samples on multiple processes.

  Args:
    worker_count: Number of processes.
    ast_generator_options: AST generator options (how to generate each sample).
    sample_options: Fuzz sample options (how to run each sample).
    seed: Optional random number generator seed. If not specified then a
      nondeterministic value is chose.
    top_run_dir: The directory to create the run directories in. If specified,
      then the run directories are not deleted after running the sample is
      complete. If not specified, then a ephemeral temporary directory is
      created for each sample.
    crasher_dir: The directory to write failing samples to.
    summary_dir: The directory to write summaries to.
    sample_count: The total number of samples to generate. If not specified the
      number of samples is unbounded unless limited by `duration`.
    duration: The total duration to run the fuzzer for.
    force_failure: If true, then every sample run is considered a failure.
      Useful for testing failure paths.
  """
  workers = []
  for i in range(worker_count):
    target = _do_worker_task
    config = WorkerConfig(
        worker_number=i,
        ast_generator_options=ast_generator_options,
        sample_options=sample_options,
        seed=seed,
        top_run_dir=top_run_dir,
        sample_count=(None if sample_count is None else
                      (sample_count + i) // worker_count),
        duration=duration,
        crasher_dir=crasher_dir,
        summary_dir=summary_dir,
        force_failure=force_failure)
    worker = multiprocess.Process(target=target, args=(config,))
    worker.start()
    workers.append(worker)

  for i, worker in enumerate(workers):
    print('-- Joining on worker {}'.format(i))
    worker.join()
