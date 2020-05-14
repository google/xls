# Lint as: python3
#
# Copyright 2020 Google LLC
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

"""Converts the DSL text to (optimized) IR and runs it as a benchmark.

Usage: $prog <dslx-text>
"""

import functools
import subprocess
import tempfile

from absl import app
from absl import flags

from xls.common import runfiles
from xls.dslx import import_routines
from xls.tools import convert_helpers

FLAGS = flags.FLAGS
BENCHMARK_MAIN_PATH = 'xls/tools/benchmark_main'


def main(argv):
  if len(argv) != 2:
    raise app.UsageError('Too many command-line arguments.')

  import_cache = {}
  f_import = functools.partial(import_routines.do_import, cache=import_cache)

  text = argv[1]
  p = convert_helpers.convert_dslx_to_package(
      text, name='cli', f_import=f_import)
  entry = p.get_function_names()[0]
  opt_ir = convert_helpers.optimize_and_dump(p)
  benchmark_main_path = runfiles.get_path(BENCHMARK_MAIN_PATH)
  with tempfile.NamedTemporaryFile() as f:
    f.write(opt_ir.encode('utf-8'))
    f.flush()
    subprocess.check_call(
        [benchmark_main_path, f.name, '--entry={}'.format(entry)])


if __name__ == '__main__':
  app.run(main)
