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

"""Helpers that parse-then-interpret some text with error handler."""

import io
import os
import sys
import time
from typing import Text, Optional, cast, Tuple

from xls.dslx import import_helpers
from xls.dslx import ir_converter
from xls.dslx import parser_helpers
from xls.dslx.interpreter import quickcheck_helpers
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.cpp_deduce import TypeInferenceError
from xls.dslx.python.cpp_deduce import XlsTypeError
from xls.dslx.python.cpp_parser import CppParseError
from xls.dslx.python.cpp_parser import Parser
from xls.dslx.python.cpp_scanner import ScanError
from xls.dslx.python.cpp_scanner import Scanner
from xls.dslx.python.interpreter import FailureError
from xls.dslx.python.interpreter import Interpreter
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
                   additional_search_paths: Tuple[str, ...] = (),
                   raise_on_error: bool = True,
                   test_filter: Optional[Text] = None,
                   trace_all: bool = False,
                   compare_jit: bool = True,
                   seed: Optional[int] = None) -> bool:
  """Parses program and run all tests contained inside.

  Args:
    program: The program text to parse.
    name: Name for the module.
    filename: The filename from which "program" text originates.
    additional_search_paths: Additional paths at which we search for imported
      module files.
    raise_on_error: When true, raises exceptions that happen in tests;
      otherwise, simply returns a boolean to the caller when all test have run.
    test_filter: Test filter specification (e.g. as passed from bazel test
      environment).
    trace_all: Whether or not to trace all expressions.
    compare_jit: Whether or not to assert equality between interpreted and
      JIT'd function return values.
    seed: Seed for QuickCheck random input stimulus.

  Returns:
    Whether or not an error occurred during parsing/testing.

  Raises:
    ScanError, ParseError: In case of front-end errors.
    TypeInferenceError, TypeError: In case of type errors.
    EvaluateError: In case of a runtime failure.
  """
  did_fail = False
  test_name = None
  type_info = None

  importer = import_helpers.Importer(additional_search_paths)
  ran = 0

  try:
    module = Parser(Scanner(filename, program), name).parse_module()
    type_info = cpp_typecheck.check_module(module, importer.cache,
                                           importer.additional_search_paths)

    ir_package = (
        ir_converter.convert_module_to_package(
            module, type_info, traverse_tests=True) if compare_jit else None)

    interpreter = Interpreter(
        module,
        type_info,
        importer.typecheck,
        importer.additional_search_paths,
        importer.cache,
        trace_all=trace_all,
        ir_package=ir_package)
    for test_name in module.get_test_names():
      if not _matches(test_name, test_filter):
        continue
      ran += 1
      print('[ RUN UNITTEST     ]', test_name, file=sys.stderr)
      interpreter.run_test(test_name)
      print('[               OK ]', test_name, file=sys.stderr)

    if ir_package and module.get_quickchecks():
      if seed is None:
        # We want to guarantee non-determinism by default. See
        # https://abseil.io/docs/cpp/guides/random#stability-of-generated-sequences
        # for rationale.
        seed = int(os.getpid() * time.time())
      print(f'[ SEED: {seed} ]')
      for quickcheck in module.get_quickchecks():
        test_name = quickcheck.f.name.identifier
        print('[ RUN QUICKCHECK   ]', test_name, file=sys.stderr)
        quickcheck_helpers.run_quickcheck(
            interpreter, ir_package, quickcheck, seed=seed)
        print('[               OK ]', test_name, file=sys.stderr)

  except (PositionalError, FailureError, CppParseError, ScanError,
          TypeInferenceError, XlsTypeError) as e:
    did_fail = True
    parser_helpers.pprint_positional_error(
        e, output=cast(io.IOBase, sys.stderr))
    if test_name:
      print(
          '[           FAILED ]',
          test_name,
          e.__class__.__name__,
          file=sys.stderr)
    if raise_on_error:
      raise
  finally:
    if type_info is not None:
      type_info.clear_type_info_refs_for_gc()

  print('[==================]', ran, 'test(s) ran.', file=sys.stderr)
  return did_fail


def parse_and_test_path(path: Text,
                        raise_on_error: bool = True,
                        test_filter: Optional[Text] = None,
                        trace_all: bool = False,
                        compare_jit: bool = True,
                        seed: Optional[int] = None) -> bool:
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
      compare_jit=compare_jit,
      seed=seed)
