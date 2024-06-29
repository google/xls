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

"""Runs the opt_main binary (akin to a shell script)."""

import subprocess

from absl import app
from absl import flags

from xls.common import runfiles

FLAGS = flags.FLAGS
OPT_MAIN = 'xls/tools/opt_main'


def main(argv):
  if len(argv) != 2:
    raise app.UsageError('Too many command-line arguments.')

  subprocess.check_call(
      [runfiles.get_path(OPT_MAIN), runfiles.get_path(argv[1])]
  )


if __name__ == '__main__':
  app.run(main)
