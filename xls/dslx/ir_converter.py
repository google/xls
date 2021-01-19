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
from typing import Text, List, Optional

from absl import logging

from xls.common.xls_error import XlsError
from xls.dslx import extract_conversion_order
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ir_converter
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.python.import_routines import ImportCache
from xls.dslx.span import PositionalError
from xls.ir.python import package as ir_package
from xls.ir.python import verifier as verifier_mod


class ParametricConversionError(XlsError):
  """Raised when we attempt to IR convert a parametric function."""


class ConversionError(PositionalError):
  """Raised on an issue converting to IR at a particular file position."""


def _convert_one_function(package: ir_package.Package,
                          module: ast.Module,
                          function: ast.Function,
                          type_info: type_info_mod.TypeInfo,
                          import_cache: ImportCache,
                          symbolic_bindings: Optional[SymbolicBindings] = None,
                          emit_positions: bool = True) -> Text:
  """Converts a single function into its emitted text form.

  Args:
    package: IR package we're converting the function into.
    module: Module we're converting a function within.
    function: Function we're converting.
    type_info: Type information about module from the typechecking phase.
    import_cache: Cache of modules potentially referenced by "module" above.
    symbolic_bindings: Parametric bindings to use during conversion, if this
      function is parametric.
    emit_positions: Whether to emit position information into the IR based on
      the AST's source positions.

  Returns:
    The converted IR function text.
  """
  function_by_name = module.get_function_by_name()
  constant_by_name = module.get_constant_by_name()
  type_definition_by_name = module.get_type_definition_by_name()
  import_by_name = module.get_import_by_name()
  converter = cpp_ir_converter.IrConverter(package, module, type_info,
                                           import_cache, emit_positions)

  freevars = function.body.get_free_variables(
      function.span.start).get_name_def_tups(module)
  logging.vlog(2, 'Unfiltered free variables for function %s: %s',
               function.identifier, freevars)
  logging.vlog(3, 'Type definition by name: %r', type_definition_by_name)
  for identifier, name_def in freevars:
    if (identifier in function_by_name or
        identifier in type_definition_by_name or identifier in import_by_name or
        isinstance(name_def, ast.BuiltinNameDef)):
      pass
    elif identifier in constant_by_name:
      converter.add_constant_dep(constant_by_name[identifier])
    else:
      raise NotImplementedError(
          f'Cannot convert free variable: {identifier}; not a function nor constant'
      )

  symbolic_binding_keys = set(b.identifier for b in symbolic_bindings or ())
  f_parametric_keys = function.get_free_parametric_keys()
  if f_parametric_keys > symbolic_binding_keys:
    raise ValueError(
        'Not enough symbolic bindings to convert function {!r}; need {!r} got {!r}'
        .format(function.name.identifier, f_parametric_keys,
                symbolic_binding_keys))

  logging.vlog(3, 'Converting function: %s; symbolic bindings: %s', function,
               symbolic_bindings)
  f = converter.visit_function(function, symbolic_bindings)
  return f.dump_ir(recursive=False)


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
        _convert_one_function(
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
  _convert_one_function(
      package,
      module,
      module.get_function(entry_function_name),
      type_info,
      import_cache,
      emit_positions=emit_positions)
  return package.dump_ir()
