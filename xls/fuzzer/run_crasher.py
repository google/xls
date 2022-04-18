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

"""Tool for running 'crasher' fuzz samples."""

import shutil
import tempfile

from absl import app
from absl import flags

from xls.common import gfile
from xls.fuzzer import run_fuzz
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_sample as sample

flags.DEFINE_string(
    'run_dir', None, help='The directory to run the crasher in.')
flags.DEFINE_string(
    'simulator', None, 'Verilog simulator to use. If not specified, the value '
    'specified in the crasher file. If the simulator is '
    'not specified in either location, the default simulator '
    'is used.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) != 2:
    raise app.UsageError(
        'Invalid command-line arguments; want {} <crasher path>'.format(
            argv[0]))

  with gfile.open(argv[1], 'r') as f:
    smp = sample.Sample.deserialize(f.read())
  if FLAGS.simulator:
    smp = smp._replace(options=smp.options._replace(simulator=FLAGS.simulator))

  run_dir = FLAGS.run_dir if FLAGS.run_dir else tempfile.mkdtemp('run_crasher_')

  print(f'Running crasher in directory {run_dir}')
  try:
    run_fuzz.run_sample(smp, run_dir)
  except sample_runner.SampleError:
    print('FAILURE')
    return 1

  print('SUCCESS')
  if not FLAGS.run_dir:
    # Remove the directory if it is temporary.
    shutil.rmtree(run_dir)
  return 0


if __name__ == '__main__':
  app.run(main)
