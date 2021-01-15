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
from xls.dslx.python import cpp_ast_visitor
from xls.dslx.python import cpp_ir_converter
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_ast_visitor import visit
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.python.import_routines import ImportCache
from xls.dslx.span import PositionalError
from xls.ir.python import function as ir_function
from xls.ir.python import package as ir_package
from xls.ir.python import verifier as verifier_mod


class ParametricConversionError(XlsError):
  """Raised when we attempt to IR convert a parametric function."""


class ConversionError(PositionalError):
  """Raised on an issue converting to IR at a particular file position."""


class _IrConverterFb(cpp_ast_visitor.AstVisitor):
  """An AST visitor that converts AST nodes into IR.

  Note that ASTs can only be converted to IR once they have been fully
  concretized; there is no parametric-function support in the IR text!

  Attributes:
    module: Module that we're converting IR for.
    node_to_ir: Mapping from AstNode to the IR that was used to emit it (as a
      FunctionBuilder BValue).
    symbolic_bindings: Mapping from parametric binding name (e.g. "N" in
      `fn [N: u32] id(x: bits[N]) -> bits[N]`) to its value in this conversion.
    type_info: Type information for the AstNodes determined during the type
      checking phase (that must precede IR conversion).
    constant_deps: Externally-noted constant dependencies that this function
      has (as free variables); noted via add_constant_dep().
    emit_positions: Whether or not we should emit position data based on the AST
      node source positions.
  """

  @classmethod
  def make(cls, package: ir_package.Package, module: ast.Module,
           type_info: type_info_mod.TypeInfo, import_cache: ImportCache,
           emit_positions: bool):
    return cls.from_state(
        cpp_ir_converter.IrConverter(package, module, type_info, import_cache,
                                     emit_positions))

  @classmethod
  def from_state(cls, state: cpp_ir_converter.IrConverter):
    return cls(state)

  def __init__(self, state: cpp_ir_converter.IrConverter):
    self.state = state

  def add_constant_dep(self, constant: ast.Constant) -> None:
    self.state.add_constant_dep(constant)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TypeRef(self, node: ast.TypeRef) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TypeRefTypeAnnotation(self,
                                  node: ast.TypeRefTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_ArrayTypeAnnotation(self, node: ast.ArrayTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_BuiltinTypeAnnotation(self,
                                  node: ast.BuiltinTypeAnnotation) -> None:
    pass

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_TupleTypeAnnotation(self, node: ast.TupleTypeAnnotation) -> None:
    pass

  def visit_Ternary(self, node: ast.Ternary):
    self.state.handle_ternary(node)

  def visit_Binop(self, node: ast.Binop):
    self.state.handle_binop(node)

  def _visit(self, node: ast.AstNode) -> None:
    visit(node, self)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Match(self, node: ast.Match):
    self.state.handle_match(node, self._visit)

  def visit_Unop(self, node: ast.Unop):
    self.state.handle_unop(node)

  def visit_Attr(self, node: ast.Attr) -> None:
    self.state.handle_attr(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Index(self, node: ast.Index) -> None:
    self.state.handle_index(node, self._visit)

  def visit_Number(self, node: ast.Number):
    self.state.handle_number(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Constant(self, node: ast.Constant) -> None:
    self.state.handle_constant_def(node, self._visit)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Array(self, node: ast.Array) -> None:
    self.state.handle_array(node, self._visit)

  def visit_ConstantArray(self, node: ast.ConstantArray) -> None:
    self.state.handle_constant_array(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Cast(self, node: ast.Cast) -> None:
    self.state.handle_cast(node, self._visit)

  def visit_XlsTuple(self, node: ast.XlsTuple) -> None:
    self.state.handle_xls_tuple(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_SplatStructInstance(self, node: ast.SplatStructInstance) -> None:
    self.state.handle_splat_struct_instance(node, self._visit)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_StructInstance(self, node: ast.StructInstance) -> None:
    self.state.handle_struct_instance(node, self._visit)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_For(self, node: ast.For) -> None:

    def visit_converter(state: cpp_ir_converter.IrConverter, node: ast.Expr):
      assert isinstance(state, cpp_ir_converter.IrConverter), state
      assert isinstance(node, ast.Expr), node
      converter = _IrConverterFb.from_state(state)
      converter._visit(node)  # pylint: disable=protected-access

    self.state.handle_for(node, self._visit, visit_converter)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Invocation(self, node: ast.Invocation):
    self.state.handle_invocation(node, self._visit)

  def visit_ConstRef(self, node: ast.ConstRef) -> None:
    self.state.handle_const_ref(node)

  def visit_NameRef(self, node: ast.NameRef) -> None:
    self.state.handle_name_ref(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_ColonRef(self, node: ast.ColonRef) -> None:
    self.state.handle_colon_ref(node, self._visit)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Let(self, node: ast.Let):
    self.state.handle_let(node, self._visit)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Param(self, node: ast.Param):
    self.state.handle_param(node)

  @cpp_ast_visitor.AstVisitor.no_auto_traverse
  def visit_Function(
      self,
      node: ast.Function,
      symbolic_bindings: Optional[SymbolicBindings] = None
  ) -> ir_function.Function:
    return self.state.handle_function(node, symbolic_bindings, self._visit)

  def get_text(self) -> Text:
    return self.state.package.dump_ir()


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
  converter = _IrConverterFb.make(
      package, module, type_info, import_cache, emit_positions=emit_positions)

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
  f = converter.visit_Function(function, symbolic_bindings)
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
