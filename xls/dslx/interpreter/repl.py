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

"""Minimal read-eval-print-loop (REPL) for DSL input, just for expressions."""

import readline  # pylint: disable=unused-import
import sys

from absl import app
from absl import flags
from pyfakefs import fake_filesystem

from xls.common.python import init_xls
from xls.dslx import bit_helpers
from xls.dslx import import_helpers
from xls.dslx import parser_helpers
from xls.dslx import span
from xls.dslx.python import cpp_concrete_type as concrete_type_mod
from xls.dslx.python import cpp_parser as parser
from xls.dslx.python import cpp_pos
from xls.dslx.python import cpp_typecheck
from xls.dslx.python import interp_value as value_mod
from xls.dslx.python import interpreter as interpreter_mod
from xls.dslx.python import scanner
from xls.dslx.python.cpp_deduce import XlsTypeError

FLAGS = flags.FLAGS
FILENAME = '/fake/repl.x'
FAKE_POS = cpp_pos.Pos(FILENAME, 0, 0)
FAKE_SPAN = cpp_pos.Span(FAKE_POS, FAKE_POS)
UN_KEYWORD = scanner.Token(FAKE_SPAN, scanner.Keyword.UN)
SN_KEYWORD = scanner.Token(FAKE_SPAN, scanner.Keyword.SN)


def handle_line(line: str, stmt_index: int):
  """Runs a single user-provided line as a REPL input."""
  fn_name = f'repl_{stmt_index}'
  module_text = f"""
  import std
  fn {fn_name}() -> () {{
    {line}
  }}
  """

  # For error reporting we use a helper that puts this into a fake filesystem
  # location.
  def make_fakefs_open():
    fs = fake_filesystem.FakeFilesystem()
    fs.CreateFile(FILENAME, module_text)
    return fake_filesystem.FakeFileOpen(fs)

  importer = import_helpers.Importer()

  while True:
    try:
      fake_module = parser.Parser(
          scanner.Scanner(FILENAME, module_text), fn_name).parse_module()
    except span.PositionalError as e:
      parser_helpers.pprint_positional_error(e, fs_open=make_fakefs_open())
      return

    # First attempt at type checking, we expect this may fail the first time
    # around and we'll substitute the real return type we observe.
    try:
      type_info = cpp_typecheck.check_module(fake_module, importer.cache,
                                             importer.additional_search_paths)
    except XlsTypeError as e:
      # We use nil as a placeholder, and swap it with the type that was expected
      # and retry once we determine what that should be.
      if e.rhs_type == concrete_type_mod.ConcreteType.NIL:
        module_text = module_text.replace(' -> ()', ' -> ' + str(e.lhs_type))
        continue
      # Any other errors are likely real type errors in the code and we should
      # report them.
      parser_helpers.pprint_positional_error(e, fs_open=make_fakefs_open())
      return

    # It type checked ok, and we can proceed.
    break

  # Interpret the line and print the result.
  # TODO(leary): 2020-06-20 No let bindings for the moment, just useful for
  # evaluating expressions -- could put them into the module scope as consts.
  interpreter = interpreter_mod.Interpreter(
      fake_module,
      type_info,
      importer.typecheck,
      import_cache=importer.cache,
      additional_search_paths=(),
      trace_all=False)
  result = interpreter.run_function(fn_name, args=(), symbolic_bindings=None)
  print(result)
  type_info.clear_type_info_refs_for_gc()
  return result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  init_xls.init_xls(sys.argv)

  stmt_index = 0

  last_result = None

  while True:
    prompt = f'dslx[{stmt_index}]> ' if sys.stdin.isatty() else ''
    try:
      # TODO(leary): Update this to support multi-line input.
      line = input(prompt)
    except EOFError:
      if sys.stdin.isatty():
        print('\r', end='')
      break

    # Some helper 'magic' commands for printing out values in different ways.
    if line == '%int':
      if last_result is None:
        print('No last result for magic command.')
        continue
      assert isinstance(last_result, value_mod.Value), last_result
      print(last_result.get_bit_value_int64())
      continue
    if line == '%bin':
      if last_result is None:
        print('No last result for magic command.')
      assert isinstance(last_result, value_mod.Value), last_result
      assert last_result is not None
      print(bit_helpers.to_bits_string(last_result.get_bit_value_uint64()))
      continue

    result = handle_line(line, stmt_index)
    if result is None:
      continue
    last_result = result
    stmt_index += 1


if __name__ == '__main__':
  app.run(main)
