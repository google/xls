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
import subprocess
from typing import Optional

from xls.common import runfiles
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


def _run_worker(config: WorkerConfig):
  """Runs worker task, generating and running samples."""
  args = [
      runfiles.get_path('xls/fuzzer/run_fuzz_multiprocess_worker'),
      '--alsologtostderr',
      '--number=%d' % config.worker_number,
      '--ast_generator_options=%r' % config.ast_generator_options,
      '--sample_options=%r' % config.sample_options,
      '--crasher_dir=%s' % config.crasher_dir,
  ]
  if config.seed is not None:
    args.append('--seed=%d' % config.seed)
  if config.top_run_dir is not None:
    args.append('--top_run_dir=%s' % config.top_run_dir)
  if config.summary_dir is not None:
    args.append('--summary_dir=%s' % config.summary_dir)
  if config.sample_count is not None:
    args.append('--sample_count=%d' % config.sample_count)
  if config.duration is not None:
    args.append('--duration_seconds=%f' % config.duration.total_seconds())
  if config.force_failure:
    args.append('--force_failure')
  return subprocess.Popen(args)


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
  workers: list[subprocess.Popen[bytes]] = []
  for i in range(worker_count):
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
    worker = _run_worker(config)
    workers.append(worker)

  for i, worker in enumerate(workers):
    print('-- Waiting on worker {}'.format(i))
    worker.wait()
