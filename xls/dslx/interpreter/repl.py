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

"""Read-eval-print-loop (REPL) for DSLX input."""

import readline  # pylint: disable=unused-import
import sys

from absl import app
from absl import flags
from xls.dslx import ast
from xls.dslx import bindings as bindings_mod
from xls.dslx import concrete_type as concrete_type_mod
from xls.dslx import deduce
from xls.dslx import fakefs_util
from xls.dslx import parser
from xls.dslx import parser_helpers
from xls.dslx import scanner
from xls.dslx import span
from xls.dslx import typecheck
from xls.dslx.interpreter import interpreter as interpreter_mod
from xls.dslx.interpreter import interpreter_helpers

FLAGS = flags.FLAGS
FILENAME = '/fake/repl.x'
FAKE_POS = span.Pos(FILENAME, 0, 0)
FAKE_SPAN = span.Span(FAKE_POS, FAKE_POS)
UN_KEYWORD = scanner.Token(scanner.TokenKind.KEYWORD, FAKE_SPAN,
                           scanner.Keyword.UN)
SN_KEYWORD = scanner.Token(scanner.TokenKind.KEYWORD, FAKE_SPAN,
                           scanner.Keyword.SN)


def concrete_type_to_annotation(
    concrete_type: concrete_type_mod.ConcreteType) -> ast.TypeAnnotation:
  if concrete_type.is_bits():
    assert concrete_type.is_ubits() or concrete_type.is_sbits()
    keyword = UN_KEYWORD if concrete_type.is_ubits() else SN_KEYWORD
    num_tok = scanner.Token(scanner.TokenKind.NUMBER, FAKE_SPAN,
                            concrete_type.get_total_bit_count())
    return ast.TypeAnnotation(FAKE_SPAN, keyword, dims=(ast.Number(num_tok),))

  raise NotImplementedError(concrete_type)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  bindings = bindings_mod.Bindings()
  fake_module = ast.Module(name='repl', top=())
  interp_callback = interpreter_helpers.interpret_expr
  typecheck_callback = typecheck.check_function_or_test_in_module

  prompt = 'dslx> ' if sys.stdin.isatty() else ''

  stmt_index = 0
  while True:
    try:
      # TODO(leary): Update this to support multi-line input.
      line = input(prompt)
    except EOFError:
      print('\r', end='')
      break
    try:
      expr = parser.Parser(scanner.Scanner(FILENAME,
                                           line)).parse_expression(bindings)
    except span.PositionalError as e:
      with fakefs_util.scoped_fakefs(FILENAME, line):
        parser_helpers.pprint_positional_error(e)
      continue

    try:
      node_to_type = deduce.NodeToType()
      ctx = deduce.DeduceCtx(node_to_type, fake_module, interp_callback,
                             typecheck_callback)
      result_type = deduce.deduce(expr, ctx)
    except span.PositionalError as e:
      with fakefs_util.scoped_fakefs(FILENAME, line):
        parser_helpers.pprint_positional_error(e)
      continue

    stmt_name = f'repl_{stmt_index}'
    name_tok = scanner.Token(
        scanner.TokenKind.IDENTIFIER, span=FAKE_SPAN, value=stmt_name)
    fn = ast.Function(
        FAKE_SPAN,
        ast.NameDef(name_tok), (), (),
        concrete_type_to_annotation(result_type),
        expr,
        public=False)
    fake_module.top = fake_module.top + (fn,)
    try:
      node_to_type = typecheck.check_module(fake_module, f_import=None)
    except span.PositionalError as e:
      with fakefs_util.scoped_fakefs(FILENAME, line):
        parser_helpers.pprint_positional_error(e)
      continue
    stmt_index += 1
    interpreter = interpreter_mod.Interpreter(
        fake_module, node_to_type, f_import=None, trace_all=False)
    result = interpreter.run_function(stmt_name, args=())
    print(result)


if __name__ == '__main__':
  app.run(main)
