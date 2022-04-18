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

import datetime
import enum
import hashlib
import itertools
import multiprocessing as mp
import os
import queue as queue_mod
import shutil
import sys
import tempfile
import time
from typing import Text, Optional, Tuple, NamedTuple

import termcolor

from xls.common import gfile
from xls.common import multiprocess
from xls.fuzzer import run_fuzz
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python.cpp_sample import Sample
from xls.fuzzer.python.cpp_sample import SampleOptions


class Command(enum.Enum):
  """Command sent from generator process to worker processes."""
  RUN = 1  # Run the accompanying payload.
  STOP = 2  # Terminate, no further work items.


class QueueMessage(NamedTuple):
  """An message in the multiprocess queue."""
  command: Command
  sample: Optional[Sample] = None
  sampleno: Optional[int] = None
  # The elapsed time to generate the sample in nanoseconds.
  generate_sample_ns: Optional[int] = None


def record_crasher(workerno: int, sampleno: int, minimize_ir: bool,
                   sample: Sample, run_dir: Text, crash_path: Text,
                   num_crashers: int, exception: sample_runner.SampleError):
  """Records and writes details of a failing test as a crasher."""
  print('--- Worker {} observed an exception for sampleno {}'.format(
      workerno, sampleno))

  # Try to prune down the IR to a minimal reproducer as long as it isn't a
  # timeout.
  if minimize_ir and not exception.is_timeout:
    print('--- Worker {} attempting to minimize IR'.format(workerno))
    minimized_ir_path = run_fuzz.minimize_ir(sample, run_dir)
    if minimized_ir_path:
      print('--- Worker {} minimized IR saved in {}'.format(
          workerno, os.path.basename(minimized_ir_path)))
    else:
      print('--- Worker {} unable to minimize IR'.format(workerno))

  # Create a directory under crash_path containing the entire contents of
  # the run directory along with a crasher file. Name of directory is the
  # first eight characters of the hash of the code sample.
  digest = hashlib.sha256(sample.input_text.encode('utf-8')).hexdigest()[:8]
  sample_crasher_dir = os.path.join(crash_path, digest)
  termcolor.cprint(
      '--- Worker {} noted crasher #{} for sampleno {} in {}'.format(
          workerno, num_crashers, sampleno, sample_crasher_dir),
      color='red')
  sys.stdout.flush()
  gfile.recursively_copy_dir(
      run_dir, sample_crasher_dir, preserve_file_mask=True)
  crasher_path = os.path.join(
      sample_crasher_dir,
      'crasher_{}_{}.x'.format(datetime.date.today().strftime('%Y-%m-%d'),
                               digest[:4]))
  with gfile.open(crasher_path, 'w') as f:
    f.write(sample.to_crasher(str(exception)))


def do_worker_task(workerno: int,
                   queue: Optional[mp.Queue],
                   crash_path: Text,
                   summary_path: Optional[Text] = None,
                   save_temps_path: Optional[Text] = None,
                   minimize_ir: bool = True) -> None:
  """Runs worker task, receiving commands from generator and executing them."""
  queue = queue or multiprocess.get_user_data()[workerno]
  crashers = 0
  calls = 0
  print('---- Started worker {}'.format(workerno))
  sys.stdout.flush()
  start = datetime.datetime.now()

  # Local file to write the summary information to before writing out to the
  # potentially remote (i.e. CNS) summary file. Avoids a potential CNS write
  # with every sample. Instead data is written out in batches.
  summary_file = os.path.join(summary_path, 'summary_%d.binarypb' %
                              workerno) if summary_path else None
  summary_temp_file = tempfile.mkstemp(
      prefix='temp_summary_')[1] if summary_path else None

  i = 0  # Silence pylint warning.
  for i in itertools.count():
    message = queue.get()
    if message.command == Command.STOP:
      break
    assert message.command == Command.RUN, message.command
    calls += len(message.sample.args_batch)
    run_dir = None
    if save_temps_path:
      run_dir = os.path.join(save_temps_path, str(message.sampleno))
      os.makedirs(run_dir)
    else:
      run_dir = tempfile.mkdtemp(prefix='run_fuzz_')

    try:
      run_fuzz.run_sample(
          message.sample,
          run_dir,
          summary_file=summary_temp_file,
          generate_sample_ns=message.generate_sample_ns)
    except sample_runner.SampleError as e:
      crashers += 1
      record_crasher(workerno, message.sampleno, minimize_ir, message.sample,
                     run_dir, crash_path, crashers, e)

    if summary_file and i % 25 == 0:
      # Append the local temporary summary file to the actual, potentially
      # remote one, and delete the temporary file.
      with gfile.open(summary_temp_file, 'rb') as f:
        summaries = f.read()
      with gfile.open(summary_file, 'ab+') as f:
        f.write(summaries)
      gfile.remove(summary_temp_file)

    if not save_temps_path:
      shutil.rmtree(run_dir)

    # TODO(leary): 2020-08-28 Turn this into an option.
    if i != 0 and i % 16 == 0:
      elapsed = (datetime.datetime.now() - start).total_seconds()
      print('---- Worker {:3}: {:8.2f} samples/s {:8.2f} calls/s'.format(
          workerno, i / elapsed, calls / elapsed))
      sys.stdout.flush()

  elapsed = (datetime.datetime.now() - start).total_seconds()
  print(
      '---- Worker {:3} finished! {:3} crashers; {:8.2f} samples/s; {:8.2f} calls/s'
      .format(workerno, crashers, i / elapsed, calls / elapsed))
  sys.stdout.flush()


def print_with_linenos(text: Text):
  for i, line in enumerate(text.splitlines(), 1):
    print('{:04d} {}'.format(i, line))


def do_generator_task(queues: Tuple[mp.Queue, ...],
                      seed: int,
                      ast_generator_options: ast_generator.AstGeneratorOptions,
                      sample_count: int,
                      calls_per_sample: int,
                      default_sample_options: SampleOptions,
                      duration: Optional[datetime.timedelta] = None,
                      print_samples: bool = False) -> int:
  """Makes DSLX text / args as fuzz samples and pushes them to workers."""
  start = datetime.datetime.now()
  i = 0
  queue_sweeps = 0
  rng = ast_generator.RngState(seed)
  while True:
    if duration:  # Note: duration overrides sample count.
      if datetime.datetime.now() - start >= duration:
        print('-- Hit target generator duration of {}'.format(duration))
        sys.stdout.flush()
        break
    elif i >= sample_count:
      print('-- Hit target sample_count of {}'.format(sample_count))
      sys.stdout.flush()
      break

    if i != 0 and i % len(queues) == 0:
      queue_sweeps += 1

      # Every 16 (arbitrary) sweeps through the queue we print out the generator
      # rate and flush stdout.
      if queue_sweeps & 0xf == 0:
        delta = datetime.datetime.now() - start
        elapsed = delta.total_seconds()

        print(f'-- Generating sample {i:8,d}; elapsed: {delta}; '
              f'aggregate generate samples/s: {i/elapsed:6.2f}')
        sys.stdout.flush()

    # Generate a command message.
    with sample_runner.Timer() as t:
      sample = ast_generator.generate_sample(ast_generator_options,
                                             calls_per_sample,
                                             default_sample_options, rng)

    if print_samples:
      print_with_linenos(sample.input_text)
    message = QueueMessage(
        command=Command.RUN,
        sample=sample,
        sampleno=i,
        generate_sample_ns=t.elapsed_ns)

    # Cycle through the queues seeing if we can find one to enqueue into. In the
    # common case where queues are not full it'll happen on the first one. This
    # helps avoid the case where a single worker gums up other (ready) workers
    # from receiving samples.
    queueno = i
    while True:
      queue = queues[queueno % len(queues)]
      try:
        queue.put_nowait(message)
      except queue_mod.Full:
        queueno += 1
      else:
        break

      if (queueno - i) % len(queues) == 0:
        # Avoid burning this core on spin polling all the time by sleeping for a
        # millisecond after we've visited all the queues.
        time.sleep(0.001)

    # Bump the generated sample count.
    i += 1

  print('-- Putting stop command in worker queues after generating {} samples'
        .format(i))
  sys.stdout.flush()

  for queue in queues:
    queue.put(QueueMessage(command=Command.STOP))

  print('-- Generator task complete')
  sys.stdout.flush()
  return i
