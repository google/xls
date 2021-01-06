# Lint as: python3
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

"""Command line utility for converting an input DSLX file into IR.

The IR is suitable for feeding to the XLS backend.
"""

import os
import sys

from absl import app
from absl import flags

from xls.common.python import init_xls
from xls.dslx import import_helpers
from xls.dslx import ir_converter
from xls.dslx import parser_helpers
from xls.dslx.python import cpp_parser
from xls.dslx.python import cpp_typecheck
from xls.dslx.span import PositionalError


flags.DEFINE_string(
    'entry', None,
    'Entry function name for conversion; when not given, all functions are converted.'
)
flags.DEFINE_bool(
    'raise_exception', False,
    'Raise exception on unsuccessful conversion, by default simply exits.')
flags.DEFINE_list('dslx_path', [], 'Additional paths to search for modules.')
FLAGS = flags.FLAGS


def main(argv):
  binary = os.path.basename(argv[0])
  if len(argv) < 2:
    raise app.UsageError('Wrong number of command-line arguments; '
                         'expect %s <input-file>' % binary)

  init_xls.init_xls(sys.argv)

  path = argv[1]
  with open(path, 'r') as f:
    text = f.read()

  name = os.path.basename(path)
  name, _ = os.path.splitext(name)
  module = parser_helpers.parse_text(
      text, name, print_on_error=True, filename=path)

  importer = import_helpers.Importer(tuple(FLAGS.dslx_path))
  type_info = None

  try:
    type_info = cpp_typecheck.check_module(module, importer.cache,
                                           importer.additional_search_paths)
    if FLAGS.entry:
      print(
          ir_converter.convert_one_function(module, FLAGS.entry, type_info,
                                            importer.cache))
    else:
      print(ir_converter.convert_module(module, type_info, importer.cache))
  except (PositionalError, cpp_parser.CppParseError) as e:
    parser_helpers.pprint_positional_error(e)
    if FLAGS.raise_exception:
      raise
    else:
      sys.exit(1)
  finally:
    if type_info is not None:
      type_info.clear_type_info_refs_for_gc()


if __name__ == '__main__':
  app.run(main)
