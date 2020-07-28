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
#
# Lint as: python3

"""Helpers that parse-then-interpret some text with error handler."""

import functools
import os
import sys
from typing import Text, Optional

from xls.dslx import import_routines
from xls.dslx import ir_converter
from xls.dslx import parser_helpers
from xls.dslx import typecheck
from xls.dslx.interpreter.interpreter import Interpreter
from xls.dslx.parser import Parser
from xls.dslx.scanner import Scanner
from xls.dslx.span import PositionalError


def _matches(test_name: Text, test_filter: Optional[Text]) -> bool:
  if test_filter is None:
    return True
  # TODO(leary): 2019-08-28 Implement wildcards.
  return test_name == test_filter


def parse_and_test(program: Text,
                   name: Text,
                   *,
                   filename: Text,
                   raise_on_error: bool = True,
                   test_filter: Optional[Text] = None,
                   trace_all: bool = False,
                   compare_jit: bool = True) -> bool:
  """Parses program and run all tests contained inside.

  Args:
    program: The program text to parse.
    name: Name for the module.
    filename: The filename from which "program" text originates.
    raise_on_error: When true, raises exceptions that happen in tests;
      otherwise, simply returns a boolean to the caller when all test have run.
    test_filter: Test filter specification (e.g. as passed from bazel test
      environment).
    trace_all: Whether or not to trace all expressions.
    compare_jit: Whether or not to assert equality between interpreted and
      JIT'd function return values.

  Returns:
    Whether or not an error occurred during parsing/testing.

  Raises:
    ScanError, ParseError: In case of front-end errors.
    TypeInferenceError, TypeError: In case of type errors (when do_typecheck).
    EvaluateError: In case of a runtime failure.
  """
  did_fail = False
  test_name = None
  import_cache = {}
  f_import = functools.partial(import_routines.do_import, cache=import_cache)
  try:
    module = Parser(Scanner(filename, program)).parse_module(name)
    node_to_type = None
    node_to_type = typecheck.check_module(module, f_import)

    ir_package = (ir_converter.convert_module_to_package(module, node_to_type,
                                                         traverse_tests=True)
                  if compare_jit else None)

    interpreter = Interpreter(
        module, node_to_type, f_import, trace_all=trace_all,
        ir_package=ir_package)
    for test_name in module.get_test_names():
      if not _matches(test_name, test_filter):
        continue
      print('[ RUN      ]', test_name, file=sys.stderr)
      interpreter.run_test(test_name)
      print('[       OK ]', test_name, file=sys.stderr)
  except PositionalError as e:
    did_fail = True
    parser_helpers.pprint_positional_error(e)
    if test_name:
      print('[   FAILED ]', test_name, e.__class__.__name__, file=sys.stderr)
    if raise_on_error:
      raise
  return did_fail


def parse_and_test_path(path: Text,
                        raise_on_error: bool = True,
                        test_filter: Optional[Text] = None,
                        trace_all: bool = False,
                        compare_jit: bool = True) -> bool:
  """Wrapper around parse_and_test that reads the file contents at "path"."""
  with open(path) as f:
    text = f.read()

  name = os.path.basename(path)
  name, _ = os.path.splitext(name)

  return parse_and_test(
      text,
      name,
      filename=path,
      raise_on_error=raise_on_error,
      test_filter=test_filter,
      trace_all=trace_all,
      compare_jit=compare_jit
  )
