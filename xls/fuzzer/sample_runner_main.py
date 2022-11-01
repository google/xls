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

r"""Sample runner program.

Runs a fuzzer code sample in the given run directory. Files are copied into the
run directory if they don't already reside there. If no run directory is
specified then a temporary directory is created.

sample_runner_main --options_file=OPT_FILE \
  --input_file=INPUT_FILE \
  --args_file=ARGS_FILE \
  [RUN_DIR]
"""

import os
import shutil
import tempfile

from typing import Text

from absl import app
from absl import flags

from xls.fuzzer import sample_runner

FLAGS = flags.FLAGS

_OPTIONS_FILE = flags.DEFINE_string('options_file', None,
                                    'File to load sample runner options from.')
_INPUT_FILE = flags.DEFINE_string('input_file', None, 'Code input file.')
_ARGS_FILE = flags.DEFINE_string(
    'args_file', None,
    'Optional arguments to use for interpretation and simulation.')
_IR_CHANNEL_NAMES_FILE = flags.DEFINE_string(
    'ir_channel_names_file', None,
    'Optional ir names of input channels for a proc.')


def maybe_copy_file(file_path: Text, dir_path: Text) -> Text:
  """Copies the file to the directory if it is not already in the directory.

  Args:
    file_path: Path of file to copy.
    dir_path: Directory to copy file to.

  Returns:
    Basename of the file.
  """
  if os.path.dirname(os.path.abspath(file_path)) != os.path.abspath(dir_path):
    shutil.copy(file_path, dir_path)
  return os.path.basename(file_path)


def run(run_dir: Text):
  """Runs the sample in the given run directory."""
  runner = sample_runner.SampleRunner(run_dir)
  input_filename = maybe_copy_file(_INPUT_FILE.value, run_dir)
  options_filename = maybe_copy_file(_OPTIONS_FILE.value, run_dir)
  args_filename = None
  ir_channel_names_filename = None
  if _ARGS_FILE.value:
    args_filename = maybe_copy_file(_ARGS_FILE.value, run_dir)
  if _IR_CHANNEL_NAMES_FILE.value:
    ir_channel_names_filename = maybe_copy_file(_IR_CHANNEL_NAMES_FILE.value,
                                                run_dir)
  runner.run_from_files(input_filename, options_filename, args_filename,
                        ir_channel_names_filename)


def main(argv):
  if len(argv) == 1:
    with tempfile.TemporaryDirectory(prefix='sample_runner_') as run_dir:
      run(run_dir)
  elif len(argv) == 2:
    run_dir = argv[1]
    if not os.path.isdir(run_dir):
      raise app.UsageError(f'{run_dir} is not a directory or does not exist.')
    run(run_dir)
  else:
    raise app.UsageError('Execpted at most one argument.')


if __name__ == '__main__':
  flags.mark_flag_as_required('options_file')
  flags.mark_flag_as_required('input_file')
  app.run(main)
