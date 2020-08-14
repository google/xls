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

"""Runs the interpreter on a syntax file, running all tests.

Tests are contained within the module as top-level "test"-keyword delimited
constructs, so they are discovered after parsing and executed.
"""

import os
import sys

from absl import app
from absl import flags

from xls.dslx.interpreter import parse_and_interpret

FLAGS = flags.FLAGS
flags.DEFINE_boolean('trace_all', False, help='Trace every expression.')
flags.DEFINE_boolean('compare_jit', True, help='Run functions alongside JIT.')


def main(argv):
  if len(argv) > 3:
    raise app.UsageError('Too many command-line arguments. {}'.format(argv))

  path = argv[1]

  test_only = os.environ.get('TESTBRIDGE_TEST_ONLY')
  sys.exit(
      parse_and_interpret.parse_and_test_path(
          path,
          raise_on_error=False,
          test_filter=test_only,
          trace_all=FLAGS.trace_all,
          compare_jit=FLAGS.compare_jit))


if __name__ == '__main__':
  app.run(main)
