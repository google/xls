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

# Lint as: python3
"""Helpers that convert various XLS text/IR forms as a tools library."""

from typing import Text, Optional

from xls.dslx import ir_converter
from xls.dslx import parse_and_typecheck
from xls.ir.python import package as ir_package_mod
from xls.passes.python import standard_pipeline as standard_pipeline_mod


def convert_dslx_to_package(
    text: Text,
    name: Text,
    f_import,
    print_on_error: bool = True,
    filename: Optional[Text] = None) -> ir_package_mod.Package:
  """Converts the given DSLX text to an IR package.

  Args:
    text: DSLX text.
    name: Name of the DSLX module / resulting package.
    f_import: Function for resolving import dependencies.
    print_on_error: Whether to print (to stderr) when an error occurs with the
      DSLX text.
    filename: Filename to use when displaying error messages, a fake filesystem
      will be created to hold the DSLX text under this filename.

  Returns:
    The parsed and typechecked DSLX code, converted to an XLS IR package.
  """
  fparse_text = parse_and_typecheck.parse_text_fakefs if filename is None else parse_and_typecheck.parse_text
  filename = filename or '/fake/conversion.x'
  m, node_to_type = fparse_text(
      text,
      name,
      print_on_error=print_on_error,
      f_import=f_import,
      filename=filename)
  p = ir_converter.convert_module_to_package(m, node_to_type)
  return p


def optimize_and_dump(p: ir_package_mod.Package) -> Text:
  standard_pipeline_mod.run_standard_pass_pipeline(p)
  return p.dump_ir()
