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

"""Command line runner that parses file to AST and optionally typechecks it."""

import functools
import os
import pprint

from absl import app

from xls.dslx import import_routines
from xls.dslx import parse_and_typecheck


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  path = argv[1]
  with open(path) as f:
    text = f.read()

  name = os.path.basename(path)
  name, _ = os.path.splitext(name)

  import_cache = {}
  f_import = functools.partial(import_routines.do_import, cache=import_cache)

  module = parse_and_typecheck.parse_text(
      text,
      name,
      filename=path,
      print_on_error=True,
      f_import=f_import)
  pprint.pprint(module)


if __name__ == '__main__':
  app.run(main)
