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

"""Module for converting AST to IR text dumps."""

import pprint
from typing import Text, List

from absl import logging

from xls.dslx import extract_conversion_order
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ir_converter
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.import_routines import ImportCache
from xls.ir.python import package as ir_package
from xls.ir.python import verifier as verifier_mod


def convert_module_to_package(
    module: ast.Module,
    type_info: type_info_mod.TypeInfo,
    import_cache: ImportCache,
    emit_positions: bool = True,
    traverse_tests: bool = False) -> ir_package.Package:
  """Converts the contents of a module to IR form.

  Args:
    module: Module to convert.
    type_info: Concrete type information used in conversion.
    import_cache: Cache of modules potentially referenced by "module" above.
    emit_positions: Whether to emit positional metadata into the output IR.
    traverse_tests: Whether to convert functions called in DSLX test constructs.
      Note that this does NOT convert the test constructs themselves.

  Returns:
    The IR package that corresponds to this module.
  """
  emitted = []  # type: List[Text]
  package = ir_package.Package(module.name)
  order = extract_conversion_order.get_order(module, type_info,
                                             dict(type_info.get_imports()),
                                             traverse_tests)
  logging.vlog(3, 'Convert order: %s', pprint.pformat(order))
  for record in order:
    logging.vlog(1, 'Converting to IR: %r', record)
    emitted.append(
        cpp_ir_converter.convert_one_function(
            package,
            record.m,
            record.f,
            record.type_info,
            import_cache,
            symbolic_bindings=record.bindings,
            emit_positions=emit_positions))

  verifier_mod.verify_package(package)
  return package


def convert_module(module: ast.Module,
                   type_info: type_info_mod.TypeInfo,
                   import_cache: ImportCache,
                   emit_positions: bool = True) -> Text:
  """Same as convert_module_to_package, but converts to IR text."""
  return convert_module_to_package(module, type_info, import_cache,
                                   emit_positions).dump_ir()


def convert_one_function(module: ast.Module,
                         entry_function_name: Text,
                         type_info: type_info_mod.TypeInfo,
                         import_cache: ImportCache,
                         emit_positions: bool = True) -> Text:
  """Returns function named entry_function_name in module as IR text."""
  logging.vlog(1, 'IR-converting entry function: %r', entry_function_name)
  package = ir_package.Package(module.name)
  cpp_ir_converter.convert_one_function(
      package,
      module,
      module.get_function(entry_function_name),
      type_info,
      import_cache,
      symbolic_bindings=None,
      emit_positions=emit_positions)
  return package.dump_ir()
