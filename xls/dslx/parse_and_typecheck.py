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

"""Helpers for parsing-and-typechecking."""

import io
from typing import Text, Tuple, Callable

from absl import logging
from pyfakefs import fake_filesystem as fakefs

from xls.dslx import fakefs_util
from xls.dslx import parser_helpers
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.cpp_deduce import TypeInferenceError
from xls.dslx.python.cpp_deduce import XlsTypeError
from xls.dslx.python.import_routines import ImportCache


def parse_text(
    text: Text,
    name: Text,
    print_on_error: bool,
    *,
    import_cache: ImportCache,
    additional_search_paths: Tuple[str, ...],
    filename: Text,
    fs_open: Callable[[Text], io.IOBase] = None,
) -> Tuple[ast.Module, type_info_mod.TypeInfo]:
  """Parses text into a module with name "name" and typechecks it."""
  logging.vlog(1, 'Parsing text; name: %r', name)
  module = parser_helpers.parse_text(
      text,
      name,
      print_on_error=print_on_error,
      filename=filename,
      fs_open=fs_open)
  logging.vlog(1, 'Parsed text; name: %r', name)

  try:
    type_info = cpp_typecheck.check_module(module, import_cache,
                                           additional_search_paths)
  except (XlsTypeError, TypeInferenceError) as e:
    if print_on_error:
      # If the typecheck fails, pretty-print the error.
      parser_helpers.pprint_positional_error(e, fs_open=fs_open)
    raise

  return module, type_info


def parse_text_fakefs(
    text: Text, name: Text, print_on_error: bool, *, import_cache: ImportCache,
    additional_search_paths: Tuple[str, ...],
    filename: Text) -> Tuple[ast.Module, type_info_mod.TypeInfo]:
  """Wraps parse_text with a *fake filesystem* holding "text" in "filename".

  This primarily exists for use from testing infrastructure! For binaries and
  libraries one would expect things to be in runfiles instead of text.

  Args:
    text: Text to put in the fake file.
    name: Name to use for the module.
    print_on_error: Whether to print to stderr if an error is encountered
      parsing/typechecking the DSLX text.
    import_cache: Import cache to use for any module dependencies.
    additional_search_paths: Additional search paths to use on import.
    filename: Path to use in the fake filesystem for the contexts of the fake
      file (with DSLX text).

  Returns:
    The DSLX module and the type information.
  """
  fs = fakefs.FakeFilesystem()
  fakefs_util.create_file(fs, filename, contents=text)
  fake_open = fakefs.FakeFileOpen(fs)
  return parse_text(
      text,
      name,
      print_on_error=print_on_error,
      import_cache=import_cache,
      additional_search_paths=additional_search_paths,
      filename=filename,
      fs_open=fake_open)
