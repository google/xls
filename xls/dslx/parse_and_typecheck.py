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

"""Helpers for parsing-and-typechecking."""

import io
from typing import Text, Optional, Tuple, Callable

from pyfakefs import fake_filesystem as fakefs

from xls.dslx import ast
from xls.dslx import fakefs_util
from xls.dslx import parser_helpers
from xls.dslx import type_info as type_info_mod
from xls.dslx import typecheck
from xls.dslx.import_fn import ImportFn
from xls.dslx.xls_type_error import XlsTypeError


def _run_typecheck(
    module: ast.Module, print_on_error: bool, f_import: ImportFn,
    fs_open: Callable[[Text], io.IOBase]) -> type_info_mod.TypeInfo:
  try:
    return typecheck.check_module(module, f_import)
  except XlsTypeError as e:
    if print_on_error:
      # If the typecheck fails, pretty-print the error.
      parser_helpers.pprint_positional_error(e, fs_open=fs_open)
    raise


def parse_text(
    text: Text,
    name: Text,
    print_on_error: bool,
    *,
    f_import: Optional[Callable],
    filename: Text,
    fs_open: Callable[[Text], io.IOBase] = None,
) -> Tuple[ast.Module, type_info_mod.TypeInfo]:
  """Parses text into a module with name "name" and typechecks it."""
  module = parser_helpers.parse_text(
      text,
      name,
      print_on_error=print_on_error,
      filename=filename,
      fs_open=fs_open)

  type_info = _run_typecheck(module, print_on_error, f_import, fs_open=fs_open)

  return module, type_info


def parse_text_fakefs(
    text: Text, name: Text, print_on_error: bool, *,
    f_import: Optional[Callable],
    filename: Text) -> Tuple[ast.Module, type_info_mod.TypeInfo]:
  """Wraps parse_text with a *fake filesystem* holding "text" in "filename".

  This primarily exists for use from testing infrastructure! For binaries and
  libraries one would expect things to be in runfiles instead of text.

  Args:
    text: Text to put in the fake file.
    name: Name to use for the module.
    print_on_error: Whether to print to stderr if an error is encountered
      parsing/typechecking the DSLX text.
    f_import: Hook used when a module needs to import another module it depends
      upon.
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
      f_import=f_import,
      filename=filename,
      fs_open=fake_open)
