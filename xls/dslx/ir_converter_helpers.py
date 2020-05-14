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

"""Helper routines for parsing DSLX text and converting it to IR."""

from typing import Text, Optional

from xls.dslx import parse_and_typecheck
from xls.dslx.import_fn import ImportFn
from xls.dslx.ir_converter import convert_module_to_package
from xls.ir.python import package as ir_package


def parse_dslx_and_convert(name: Text, text: Text, print_on_error: bool,
                           f_import: Optional[ImportFn],
                           filename: Text) -> ir_package.Package:
  """Returns the parsed IR package for the given DSLX module text.

  Args:
    name: Name of the package being created.
    text: DSLX text being parsed.
    print_on_error: Whether to print (to stderr) if there is an error in the
      DSLX text.
    f_import: Function used to import dependency modules.
    filename: Filename that the DSLX text originates from.
  """
  module, node_to_type = parse_and_typecheck.parse_text(
      text,
      name,
      print_on_error=print_on_error,
      filename=filename,
      f_import=f_import)
  return convert_module_to_package(module, node_to_type)


def parse_dslx_and_convert_fakefs(name: Text, text: Text, print_on_error: bool,
                                  f_import: Optional[ImportFn],
                                  filename: Text) -> ir_package.Package:
  module, node_to_type = parse_and_typecheck.parse_text_fakefs(
      text,
      name,
      print_on_error=print_on_error,
      filename=filename,
      f_import=f_import)
  return convert_module_to_package(module, node_to_type)
