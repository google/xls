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

# pylint: disable=invalid-name

"""Type system deduction rules for AST nodes."""

import typing
from typing import Text, Dict, Union, Callable, Type, Tuple, List, Set, Optional

from absl import logging
import dataclasses

from xls.dslx import ast
from xls.dslx import ast_helpers
from xls.dslx import bit_helpers
from xls.dslx import dslx_builtins
from xls.dslx import parametric_instantiator
from xls.dslx import scanner
from xls.dslx import span
from xls.dslx.concrete_type import ArrayType
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.concrete_type import EnumType
from xls.dslx.concrete_type import FunctionType
from xls.dslx.concrete_type import TupleType
from xls.dslx.parametric_expression import ParametricAdd
from xls.dslx.parametric_expression import ParametricExpression
from xls.dslx.parametric_expression import ParametricSymbol
from xls.dslx.xls_type_error import TypeInferenceError
from xls.dslx.xls_type_error import XlsTypeError


# Dictionary used as registry for rule dispatch based on AST node class.
RULES = {}


SymbolicBindings = parametric_instantiator.SymbolicBindings
RuleFunction = Callable[[ast.AstNode, 'DeduceCtx'], ConcreteType]


def _rule(cls: Type[ast.AstNode]):
  """Decorator for a type inference rule that pertains to class 'cls'."""

  def register(f):
    # Register the checked function as the rule.
    RULES[cls] = f
    return f

  return register


class TypeMissingError(span.PositionalError):
  """Raised when there is no binding from an AST node to its corresponding type.

  This is useful to raise in order to flag free variables that are dependencies
  for type inference; e.g. functions within a module that invoke other top-level
  functions. The type inference system can catch the error, infer the
  dependency, and re-attempt the deduction of the dependent function.
  """

  def __init__(self, node: ast.AstNode, suffix: Text = ''):
    assert isinstance(node, ast.AstNode), repr(node)
    message = 'Missing type for AST node: {node}{suffix}'.format(
        node=node, suffix=' :: ' + suffix if suffix else '')
    # We don't know the real span of the user, we rely on the appropriate caller
    # to catch the error and populate this field properly.
    fake_span = span.Span(span.Pos('<fake>', 0, 0), span.Pos('<fake>', 0, 0))
    super(TypeMissingError, self).__init__(message, fake_span)
    self.node = node
    self.suffix = suffix
    self.user = None


ImportedInfo = Tuple[ast.Module, 'NodeToType']


class NodeToType(object):
  """Helper type that checks the types of {AstNode: ConcreteType} mappings.

  Easily "chains" onto an existing mapping of node types when entering a scope
  with parametric bindings; e.g. a new node_to_type mapping is created for
  a parametric function's body after parametric instantiation.

  Also raises a TypeMissingError instead of a KeyError when we encounter a node
  that does not have a type known, so that it can be handled in a more specific
  way versus a KeyError.
  """

  def __init__(self, parent: Optional['NodeToType'] = None):
    self._dict = {}  # type: Dict[ast.AstNode, ConcreteType]
    self._imports = {}  # type: Dict[ast.Import, ImportedInfo]
    self._name_to_const = {}  # type: Dict[ast.NameDef, ast.Constant]
    self._parent = parent  # type: NodeToType

  @property
  def parent(self) -> 'NodeToType':
    return self._parent

  def update(self, other: 'NodeToType') -> None:
    self._dict.update(other._dict)  # pylint: disable=protected-access
    self._imports.update(other._imports)  # pylint: disable=protected-access

  def add_import(self, import_node: ast.Import, info: ImportedInfo) -> None:
    assert import_node not in self._imports, import_node
    self._imports[import_node] = info
    self.update(info[1])

  def note_constant(self, name_def: ast.NameDef, constant: ast.Constant):
    self._name_to_const[name_def] = constant

  def get_const_int(self, name_def: ast.NameDef, user_span: span.Span) -> int:
    if name_def not in self._name_to_const and self.parent:
      constant = self.parent._name_to_const[name_def]  # pylint: disable=protected-access
    else:
      constant = self._name_to_const[name_def]
    value = constant.value
    if isinstance(value, ast.Number):
      return value.get_value_as_int()
    raise TypeInferenceError(
        span=user_span,
        type_=None,
        suffix='Expected to find a constant integral value with the name {};'
        'got: {}'.format(name_def, constant.value))

  def get_imports(self) -> Dict[ast.Import, ImportedInfo]:
    return self._imports if not self.parent else {
        **self._imports,  # pylint: disable=protected-access
        **self.parent._imports  # pylint: disable=protected-access
    }

  def get_imported(self, import_node: ast.Import) -> ImportedInfo:
    if self.parent:
      if import_node in self._imports:
        return self._imports[import_node]

      return self.parent._imports[import_node]  # pylint: disable=protected-access

    return self._imports[import_node]

  def __setitem__(self, k: ast.AstNode, v: ConcreteType) -> None:
    self._dict[k] = v

  def __getitem__(self, k: ast.AstNode) -> ConcreteType:
    """Attempts to resolve AST node 'k' in the node-to-type dictionary.

    Args:
      k: The AST node to resolve to a concrete type.

    Raises:
      TypeMissingError: When the node is not found.

    Returns:
      The previously-determined type of the AST node 'k'.
    """
    assert isinstance(k, ast.AstNode), repr(k)
    try:
      if k in self._dict:
        return self._dict[k]
      if self.parent:
        return self.parent.__getitem__(k)
    except KeyError:
      span_suffix = ' @ {}'.format(k.span) if hasattr(k, 'span') else ''
      raise TypeMissingError(
          k, suffix='resolving type of node{}'.format(span_suffix))
    else:
      span_suffix = ' @ {}'.format(k.span) if hasattr(k, 'span') else ''
      raise TypeMissingError(
          k,
          suffix='resolving type of {} node{}'.format(k.__class__.__name__,
                                                      span_suffix))

  def __contains__(self, k: ast.AstNode) -> bool:
    return (k in self._dict or self.parent.__contains__(k)
            if self.parent else k in self._dict)


# Type signature for the import function callback:
# (import_tokens) -> (module, node_to_type)
ImportFn = Callable[[Tuple[Text, ...]], Tuple[ast.Module, NodeToType]]

# Type signature for interpreter function callback:
# (module, node_to_type, env, bit_widths, expr, f_import, fn_ctx) ->
#   value_as_int
#
# This is an abstract interface to break circular dependencies; see
# interpreter_helpers.py
InterpCallbackType = Callable[[
    ast.Module, NodeToType, Dict[Text, int], Dict[Text, int], ast.Expr, ImportFn
], int]

# Type for stack of functions deduction is running on.
# [(name, symbolic_bindings), ...]
FnStack = List[Tuple[Text, Dict[Text, int]]]


@dataclasses.dataclass
class DeduceCtx:
  """A wrapper over useful objects for typechecking.

  Attributes:
    node_to_type: Maps an AST node to its deduced type.
    module: The (entry point) module we are typechecking.
    interpret_expr: An Interpreter wrapper that parametric_instantiator uses to
      evaluate bindings with complex expressions (eg. function calls).
    check_function_in_module: A callback to typecheck parametric functions that
      are not in this module.
    fn_stack: Keeps track of the function we're currently typechecking and the
      symbolic bindings we should be using.
  """
  node_to_type: NodeToType
  module: ast.Module
  interpret_expr: InterpCallbackType
  check_function_in_module: Callable[[ast.Function, 'DeduceCtx'], None]
  fn_stack: FnStack = dataclasses.field(default_factory=list)


def resolve(type_: ConcreteType, ctx: DeduceCtx) -> ConcreteType:
  """Resolves "type_" via provided symbolic bindings.

  Uses the symbolic bindings of the function we're currently inside of to
  resolve parametric types.

  Args:
    type_: Type to resolve any contained dims for.
    ctx: Deduction context to use in resolving the dims.

  Returns:
    "type_" with dimensions resolved according to bindings in "ctx".
  """
  _, fn_symbolic_bindings = ctx.fn_stack[-1]

  def resolver(dim):
    if isinstance(dim, ParametricExpression):
      return dim.evaluate(fn_symbolic_bindings)
    return dim

  return type_.map_size(resolver)


@_rule(ast.Param)
def _deduce_Param(self: ast.Param, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.type_, ctx)


@_rule(ast.Constant)
def _deduce_Constant(self: ast.Constant, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  result = ctx.node_to_type[self.name] = deduce(self.value, ctx)
  ctx.node_to_type.note_constant(self.name, self)
  return result


@_rule(ast.ConstantArray)
def _deduce_ConstantArray(
    self: ast.ConstantArray, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a ConstantArray AST node."""
  # We permit constant arrays to drop annotations for numbers as a convenience
  # (before we have unifying type inference) by allowing constant arrays to have
  # a leading type annotation. If they don't have a leading type annotation,
  # just fall back to normal array type inference, if we encounter a number
  # without a type annotation we'll flag an error per usual.
  if self.type_ is None:
    return _deduce_Array(self, ctx)

  # Determine the element type that corresponds to the annotation and go mark
  # any un-typed numbers in the constant array as having that type.
  concrete_type = deduce(self.type_, ctx)
  if not isinstance(concrete_type, ArrayType):
    raise TypeInferenceError(
        self.type_.span, concrete_type,
        f'Annotated type for array literal must be an array type; got {concrete_type.get_debug_type_name()} {self.type_}'
    )
  element_type = concrete_type.get_element_type()
  for member in self.members:
    assert ast.Constant.is_constant(member)
    if isinstance(member, ast.Number) and not member.type_:
      ctx.node_to_type[member] = element_type
      member.check_bitwidth(element_type)
  # Use the base class to check all members are compatible.
  _deduce_Array(self, ctx)
  return concrete_type


def _create_element_invocation(owner: ast.AstNodeOwner, span_: span.Span,
                               callee: Union[ast.NameRef, ast.ModRef],
                               arg_array: ast.Expr) -> ast.Invocation:
  """Creates a function invocation on the first element of the given array.

  We need to create a fake invocation to deduce the type of a function
  in the case where map is called with a builtin as the map function. Normally,
  map functions (including parametric ones) have their types deduced when their
  ast.Function nodes are encountered (where a similar fake ast.Invocation node
  is created).

  Builtins don't have ast.Function nodes, so that inference can't occur, so we
  essentually perform that synthesis and deduction here.

  Args:
    owner: AST node owner.
    span_: The location in the code where analysis is occurring.
    callee: The function to be invoked.
    arg_array: The array of arguments (at least one) to the function.

  Returns:
    An invocation node for the given function when called with an element in the
    argument array.
  """
  annotation = ast_helpers.make_builtin_type_annotation(
      owner, span_,
      scanner.Token(scanner.TokenKind.KEYWORD, span_, scanner.Keyword.U32), ())
  index_number = ast.Number(owner, span_, '32', ast.NumberKind.OTHER,
                            annotation)
  index = ast.Index(owner, span_, arg_array, index_number)
  return ast.Invocation(owner, span_, callee, (index,))


def _check_parametric_invocation(parametric_fn: ast.Function,
                                 invocation: ast.Invocation,
                                 symbolic_bindings: SymbolicBindings,
                                 ctx: DeduceCtx):
  """Checks the parametric fn body using the invocation's symbolic bindings."""
  if isinstance(invocation.callee, ast.ModRef):
    # We need to typecheck this function with respect to its own module.
    # Let's use typecheck._check_function_or_test_in_module() to do this
    # in case we run into more dependencies in that module.
    if symbolic_bindings in invocation.types_mappings:
      # We've already typechecked this imported parametric function using
      # these symbolic bindings.
      return

    imported_module, imported_node_to_type = ctx.node_to_type.get_imported(
        invocation.callee.mod)
    invocation_imported_node_to_type = NodeToType(parent=imported_node_to_type)
    imported_ctx = DeduceCtx(invocation_imported_node_to_type, imported_module,
                             ctx.interpret_expr, ctx.check_function_in_module)
    imported_ctx.fn_stack.append(
        (parametric_fn.name.identifier, dict(symbolic_bindings)))
    ctx.check_function_in_module(parametric_fn, imported_ctx)

    invocation.types_mappings[
        symbolic_bindings] = invocation_imported_node_to_type
  else:
    assert isinstance(invocation.callee, ast.NameRef), invocation.callee
    # We need to typecheck this function with respect to its own module
    # Let's take advantage of the existing try-catch mechanism in
    # typecheck._check_function_or_test_in_module().

    try:
      # See if the body is present in the node_to_type mapping (we do this just
      # to observe if it raises an exception).
      ctx.node_to_type[parametric_fn.body]
    except TypeMissingError as e:
      # If we've already typechecked the parametric function with the
      # current symbolic bindings, no need to do it again.
      if symbolic_bindings not in invocation.types_mappings:
        # Let's typecheck this parametric function using the symbolic bindings
        # we just derived to make sure they check out ok.
        e.node = invocation.callee.name_def
        ctx.fn_stack.append(
            (parametric_fn.name.identifier, dict(symbolic_bindings)))
        ctx.node_to_type = NodeToType(parent=ctx.node_to_type)
        raise

    if symbolic_bindings not in invocation.types_mappings:
      # If we haven't yet stored a node_to_type for these symbolic bindings
      # and we're at this point, it means that we just finished typechecking
      # the parametric function. Let's store the results.
      invocation.types_mappings[symbolic_bindings] = ctx.node_to_type
      ctx.node_to_type = ctx.node_to_type.parent


@_rule(ast.Invocation)
def _deduce_Invocation(self: ast.Invocation, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Invocation AST node."""
  logging.vlog(5, 'Deducing type for invocation: %s', self)
  arg_types = []
  _, fn_symbolic_bindings = ctx.fn_stack[-1]
  for arg in self.args:
    try:
      arg_types.append(resolve(deduce(arg, ctx), ctx))
    except TypeMissingError as e:
      # These nodes could be ModRefs or NameRefs.
      callee_is_map = isinstance(
          self.callee, ast.NameRef) and self.callee.name_def.identifier == 'map'
      arg_is_builtin = isinstance(
          arg, ast.NameRef
      ) and arg.name_def.identifier in dslx_builtins.PARAMETRIC_BUILTIN_NAMES
      if callee_is_map and arg_is_builtin:
        invocation = _create_element_invocation(ctx.module, self.span, arg,
                                                self.args[0])
        arg_types.append(resolve(deduce(invocation, ctx), ctx))
      else:
        raise

  try:
    # This will get us the type signature of the function.
    # If the function is parametric, we won't check its body
    # until after we have symbolic bindings for it
    callee_type = deduce(self.callee, ctx)
  except TypeMissingError as e:
    e.span = self.span
    e.user = self
    raise

  if not isinstance(callee_type, FunctionType):
    raise XlsTypeError(self.callee.span, callee_type, None,
                       'Callee does not have a function type.')

  if isinstance(self.callee, ast.ModRef):
    imported_module, _ = ctx.node_to_type.get_imported(self.callee.mod)
    callee_name = self.callee.value_tok.value
    callee_fn = imported_module.get_function(callee_name)
  else:
    assert isinstance(self.callee, ast.NameRef), self.callee
    callee_name = self.callee.identifier
    callee_fn = ctx.module.get_function(callee_name)

  self_type, callee_sym_bindings = parametric_instantiator.instantiate_function(
      self.span, callee_type, tuple(arg_types), ctx,
      callee_fn.parametric_bindings)

  # Within the context of (mod_name, fn_name, fn_sym_bindings),
  # this invocation of callee will have bindings with values specified by
  # callee_sym_bindings
  self.symbolic_bindings[tuple(
      fn_symbolic_bindings.items())] = callee_sym_bindings

  if callee_fn.is_parametric():
    # Now that we have callee_sym_bindings, let's use them to typecheck the body
    # of callee_fn to make sure these values actually work
    _check_parametric_invocation(callee_fn, self, callee_sym_bindings, ctx)

  return self_type


def _deduce_slice_type(self: ast.Index, ctx: DeduceCtx,
                       lhs_type: ConcreteType) -> ConcreteType:
  """Deduces the concrete type of an Index AST node with a slice spec."""
  index_slice = self.index
  assert isinstance(index_slice, (ast.Slice, ast.WidthSlice)), index_slice

  # TODO(leary): 2019-10-28 Only slicing bits types for now, and only with
  # number ast nodes, generalize to arrays and constant expressions.
  if not isinstance(lhs_type, BitsType):
    raise XlsTypeError(self.span, lhs_type, None,
                       'Value to slice is not of "bits" type.')

  bit_count = lhs_type.get_total_bit_count()

  if isinstance(index_slice, ast.WidthSlice):
    start = index_slice.start
    if isinstance(start, ast.Number) and start.type_ is None:
      start_type = lhs_type.to_ubits()
      resolved_start_type = resolve(start_type, ctx)
      if not bit_helpers.fits_in_bits(
          start.get_value_as_int(), resolved_start_type.get_total_bit_count()):
        raise TypeInferenceError(
            start.span, resolved_start_type,
            'Cannot fit {} in {} bits (inferred from bits to slice).'.format(
                start.get_value_as_int(),
                resolved_start_type.get_total_bit_count()))
      ctx.node_to_type[start] = start_type
    else:
      start_type = deduce(start, ctx)

    # Check the start is unsigned.
    if start_type.signed:
      raise TypeInferenceError(
          start.span,
          type_=start_type,
          suffix='Start index for width-based slice must be unsigned.')

    width_type = deduce(index_slice.width, ctx)
    if isinstance(width_type.get_total_bit_count(), int) and isinstance(
        lhs_type.get_total_bit_count(), int
    ) and width_type.get_total_bit_count() > lhs_type.get_total_bit_count():
      raise XlsTypeError(
          start.span, lhs_type, width_type,
          'Slice type must have <= original number of bits; attempted slice from {} to {} bits.'
          .format(lhs_type.get_total_bit_count(),
                  width_type.get_total_bit_count()))

    # Check the width type is bits-based (no enums, since value could be out
    # of range of the enum values).
    if not isinstance(width_type, BitsType):
      raise TypeInferenceError(
          self.span,
          type_=width_type,
          suffix='Require a bits-based type for width-based slice.')

    # The width type is the thing returned from the width-slice.
    return width_type

  assert isinstance(index_slice, ast.Slice), index_slice
  limit = index_slice.limit.get_value_as_int() if index_slice.limit else None
  # PyType has trouble figuring out that start is definitely an Number at this
  # point.
  start = index_slice.start
  assert isinstance(start, (ast.Number, type(None)))
  start = start.get_value_as_int() if start else None

  _, fn_symbolic_bindings = ctx.fn_stack[-1]
  if isinstance(bit_count, ParametricExpression):
    bit_count = bit_count.evaluate(fn_symbolic_bindings)
  start, width = bit_helpers.resolve_bit_slice_indices(bit_count, start, limit)
  key = tuple(fn_symbolic_bindings.items())
  index_slice.bindings_to_start_width[key] = (start, width)
  return BitsType(signed=False, size=width)


@_rule(ast.Index)
def _deduce_Index(self: ast.Index, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Index AST node."""
  lhs_type = deduce(self.lhs, ctx)

  # Check whether this is a slice-based indexing operations.
  if isinstance(self.index, (ast.Slice, ast.WidthSlice)):
    return _deduce_slice_type(self, ctx, lhs_type)

  index_type = deduce(self.index, ctx)
  if isinstance(lhs_type, TupleType):
    if not isinstance(self.index, ast.Number):
      raise XlsTypeError(self.index.span, index_type, None,
                         'Tuple index is not a literal number.')
    index_value = self.index.get_value_as_int()
    if index_value >= lhs_type.get_tuple_length():
      raise XlsTypeError(
          self.index.span, lhs_type, None,
          'Tuple index {} is out of range for this tuple type.'.format(
              index_value))
    return lhs_type.get_unnamed_members()[index_value]

  if not isinstance(lhs_type, ArrayType):
    raise TypeInferenceError(self.lhs.span, lhs_type,
                             'Value to index is not an array.')

  index_ok = isinstance(index_type,
                        BitsType) and not isinstance(index_type, ArrayType)
  if not index_ok:
    raise XlsTypeError(self.index.span, index_type, None,
                       'Index type is not scalar bits.')
  return lhs_type.get_element_type()


@_rule(ast.XlsTuple)
def _deduce_XlsTuple(self: ast.XlsTuple, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  members = tuple(deduce(m, ctx) for m in self.members)
  return TupleType(members)


def _bind_names(name_def_tree: ast.NameDefTree, type_: ConcreteType,
                ctx: DeduceCtx) -> None:
  """Binds names in name_def_tree to corresponding type given in type_."""
  if name_def_tree.is_leaf():
    name_def = name_def_tree.get_leaf()
    ctx.node_to_type[name_def] = type_
    return

  if not isinstance(type_, TupleType):
    raise XlsTypeError(
        name_def_tree.span,
        type_,
        rhs_type=None,
        suffix='Expected a tuple type for these names, but got {}.'.format(
            type_))

  if len(name_def_tree.tree) != type_.get_tuple_length():
    raise TypeInferenceError(
        name_def_tree.span, type_,
        'Could not bind names, names are mismatched in number vs type; at '
        'this level of the tuple: {} names, {} types.'.format(
            len(name_def_tree.tree), type_.get_tuple_length()))

  for subtree, subtype in zip(name_def_tree.tree, type_.get_unnamed_members()):
    ctx.node_to_type[subtree] = subtype
    _bind_names(subtree, subtype, ctx)


@_rule(ast.Let)
def _deduce_Let(self: ast.Let, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Let AST node."""

  rhs_type = deduce(self.rhs, ctx)
  resolved_rhs_type = resolve(rhs_type, ctx)

  if self.type_ is not None:
    concrete_type = deduce(self.type_, ctx)
    resolved_concrete_type = resolve(concrete_type, ctx)

    if resolved_rhs_type != resolved_concrete_type:
      raise XlsTypeError(
          self.rhs.span, resolved_concrete_type, resolved_rhs_type,
          'Annotated type did not match inferred type of right hand side.')

  _bind_names(self.name_def_tree, resolved_rhs_type, ctx)

  if self.const:
    deduce(self.const, ctx)

  return deduce(self.body, ctx)


def _unify_WildcardPattern(_self: ast.WildcardPattern, _type: ConcreteType,
                           _ctx: DeduceCtx) -> None:
  pass  # Wildcard matches any type.


def _unify_NameDefTree(self: ast.NameDefTree, type_: ConcreteType,
                       ctx: DeduceCtx) -> None:
  """Unifies the NameDefTree AST node with the observed RHS type type_."""
  resolved_rhs_type = resolve(type_, ctx)
  if self.is_leaf():
    leaf = self.get_leaf()
    if isinstance(leaf, ast.NameDef):
      ctx.node_to_type[leaf] = resolved_rhs_type
    elif isinstance(leaf, ast.WildcardPattern):
      pass
    elif isinstance(leaf, (ast.Number, ast.EnumRef)):
      resolved_leaf_type = resolve(deduce(leaf, ctx), ctx)
      if resolved_leaf_type != resolved_rhs_type:
        raise TypeInferenceError(
            span=self.span,
            type_=resolved_rhs_type,
            suffix='Conflicting types; pattern expects {} but got {} from value'
            .format(resolved_rhs_type, resolved_leaf_type))
    else:
      assert isinstance(leaf, ast.NameRef), repr(leaf)
      ref_type = ctx.node_to_type[leaf.name_def]
      resolved_ref_type = resolve(ref_type, ctx)
      if resolved_ref_type != resolved_rhs_type:
        raise TypeInferenceError(
            span=self.span,
            type_=resolved_rhs_type,
            suffix='Conflicting types; pattern expects {} but got {} from reference'
            .format(resolved_rhs_type, resolved_ref_type))
  else:
    assert isinstance(self.tree, tuple)
    if isinstance(type_, TupleType) and type_.get_tuple_length() == len(
        self.tree):
      for subtype, subtree in zip(type_.get_unnamed_members(), self.tree):
        _unify(subtree, subtype, ctx)


def _unify(n: ast.AstNode, other: ConcreteType, ctx: DeduceCtx) -> None:
  f = globals()['_unify_{}'.format(n.__class__.__name__)]
  f(n, other, ctx)


@_rule(ast.Match)
def _deduce_Match(self: ast.Match, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Match AST node."""
  matched = deduce(self.matched, ctx)

  for arm in self.arms:
    for pattern in arm.patterns:
      _unify(pattern, matched, ctx)

  arm_types = tuple(deduce(arm, ctx) for arm in self.arms)

  resolved_arm0_type = resolve(arm_types[0], ctx)

  for i, arm_type in enumerate(arm_types[1:], 1):
    resolved_arm_type = resolve(arm_type, ctx)
    if resolved_arm_type != resolved_arm0_type:
      raise XlsTypeError(
          self.arms[i].span, resolved_arm_type, resolved_arm0_type,
          'This match arm did not have the same type as preceding match arms.')
  return resolved_arm0_type


@_rule(ast.MatchArm)
def _deduce_MatchArm(self: ast.MatchArm, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.expr, ctx)


@_rule(ast.For)
def _deduce_For(self: ast.For, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a For AST node."""
  init_type = deduce(self.init, ctx)
  annotated_type = deduce(self.type_, ctx)
  _bind_names(self.names, annotated_type, ctx)
  body_type = deduce(self.body, ctx)
  deduce(self.iterable, ctx)

  resolved_init_type = resolve(init_type, ctx)
  resolved_body_type = resolve(body_type, ctx)

  if resolved_init_type != resolved_body_type:
    raise XlsTypeError(
        self.span, resolved_init_type, resolved_body_type,
        "For-loop init value type did not match for-loop body's result type.")
  # TODO(leary): 2019-02-19 Type check annotated_type (the bound names each
  # iteration) against init_type/body_type -- this requires us to understand
  # how iterables turn into induction values.
  return resolved_init_type


@_rule(ast.While)
def _deduce_While(self: ast.While, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a While AST node."""
  init_type = deduce(self.init, ctx)
  test_type = deduce(self.test, ctx)

  resolved_init_type = resolve(init_type, ctx)
  resolved_test_type = resolve(test_type, ctx)

  if resolved_test_type != ConcreteType.U1:
    raise XlsTypeError(self.test.span, test_type, ConcreteType.U1,
                       'Expect while-loop test to be a bool value.')

  body_type = deduce(self.body, ctx)
  resolved_body_type = resolve(body_type, ctx)

  if resolved_init_type != resolved_body_type:
    raise XlsTypeError(
        self.span, init_type, body_type,
        "While-loop init value type did not match while-loop body's "
        'result type.')
  return resolved_init_type


@_rule(ast.Carry)
def _deduce_Carry(self: ast.Carry, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.loop.init, ctx)


def _is_acceptable_cast(from_: ConcreteType, to: ConcreteType) -> bool:
  if {type(from_), type(to)} == {ArrayType, BitsType}:
    return from_.get_total_bit_count() == to.get_total_bit_count()
  return True


@_rule(ast.Cast)
def _deduce_Cast(self: ast.Cast, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Cast AST node."""
  type_result = deduce(self.type_, ctx)
  expr_type = deduce(self.expr, ctx)

  resolved_type_result = resolve(type_result, ctx)
  resolved_expr_type = resolve(expr_type, ctx)

  if not _is_acceptable_cast(from_=resolved_type_result, to=resolved_expr_type):
    raise XlsTypeError(
        self.span, expr_type, type_result,
        'Cannot cast from expression type {} to {}.'.format(
            resolved_expr_type, resolved_type_result))
  return resolved_type_result


@_rule(ast.Unop)
def _deduce_Unop(self: ast.Unop, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.operand, ctx)


@_rule(ast.Array)
def _deduce_Array(self: ast.Array, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Array AST node."""
  member_types = [deduce(m, ctx) for m in self.members]
  resolved_type0 = resolve(member_types[0], ctx)
  for i, x in enumerate(member_types[1:], 1):
    resolved_x = resolve(x, ctx)
    logging.vlog(5, 'array member type %d: %s', i, resolved_x)
    if resolved_x != resolved_type0:
      raise XlsTypeError(
          self.members[i].span, resolved_type0, resolved_x,
          'Array member did not have same type as other members.')

  inferred = ArrayType(resolved_type0, len(member_types))

  if not self.type_:
    return inferred

  annotated = deduce(self.type_, ctx)
  if not isinstance(annotated, ArrayType):
    raise XlsTypeError(self.span, annotated, None,
                       'Array was not annotated with an array type.')
  resolved_element_type = resolve(annotated.get_element_type(), ctx)
  if resolved_element_type != resolved_type0:
    raise XlsTypeError(
        self.span, resolved_element_type, resolved_type0,
        'Annotated element type did not match inferred element type.')

  if self.has_ellipsis:
    # Since there are ellipsis, we determine the size from the annotated type.
    # We've already checked the element types lined up.
    return annotated
  else:
    if annotated.size != len(member_types):
      raise XlsTypeError(
          self.span, annotated, inferred,
          'Annotated array size {!r} does not match inferred array size {!r}.'
          .format(annotated.size, len(member_types)))
    return inferred


@_rule(ast.TypeRef)
def _deduce_TypeRef(self: ast.TypeRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  return deduce(self.type_def, ctx)


@_rule(ast.ConstRef)
@_rule(ast.NameRef)
def _deduce_NameRef(self: ast.NameRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a NameDef AST node."""
  try:
    result = ctx.node_to_type[self.name_def]
  except TypeMissingError as e:
    logging.vlog(3, 'Could not resolve name def: %s', self.name_def)
    e.span = self.span
    e.user = self
    raise
  return result


@_rule(ast.EnumRef)
def _deduce_EnumRef(self: ast.EnumRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an EnumRef AST node."""
  try:
    result = ctx.node_to_type[self.enum]
  except TypeMissingError as e:
    logging.vlog(3, 'Could not resolve enum to type: %s', self.enum)
    e.span = self.span
    e.user = self
    raise

  # Check the name we're accessing is actually defined on the enum.
  assert isinstance(result, EnumType), result
  enum = result.nominal_type
  assert isinstance(enum, ast.Enum), enum
  name = self.value_tok.value
  if not enum.has_value(name):
    raise TypeInferenceError(
        span=self.span,
        type_=None,
        suffix='Name {!r} is not defined by the enum {}'.format(
            name, enum.identifier))
  return result


@_rule(ast.Number)
def _deduce_Number(self: ast.Number, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Number AST node."""
  if not self.type_:
    if self.kind == ast.NumberKind.BOOL:
      return ConcreteType.U1
    if self.kind == ast.NumberKind.CHARACTER:
      return ConcreteType.U8
    raise TypeInferenceError(
        span=self.span,
        type_=None,
        suffix='Could not infer a type for this number, please annotate a type.'
    )
  concrete_type = resolve(deduce(self.type_, ctx), ctx)
  self.check_bitwidth(concrete_type)
  return concrete_type


@_rule(ast.TypeDef)
def _deduce_TypeDef(self: ast.TypeDef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  concrete_type = deduce(self.type_, ctx)
  ctx.node_to_type[self.name] = concrete_type
  return concrete_type


def _dim_to_parametric(self: ast.TypeAnnotation,
                       expr: ast.Expr) -> ParametricExpression:
  """Converts a dimension expression to a 'parametric' AST node."""
  assert not isinstance(expr, ast.ConstRef), expr
  if isinstance(expr, ast.NameRef):
    return ParametricSymbol(expr.name_def.identifier, expr.span)
  if isinstance(expr, ast.Binop):
    if expr.operator.kind == scanner.TokenKind.PLUS:
      return ParametricAdd(
          _dim_to_parametric(self, expr.lhs),
          _dim_to_parametric(self, expr.rhs))
  msg = 'Could not concretize type with dimension: {}.'.format(expr)
  raise TypeInferenceError(self.span, self, suffix=msg)


def _dim_to_parametric_or_int(
    self: ast.TypeAnnotation, expr: ast.Expr,
    ctx: DeduceCtx) -> Union[int, ParametricExpression]:
  if isinstance(expr, ast.Number):
    ctx.node_to_type[expr] = ConcreteType.U32
    return expr.get_value_as_int()
  if isinstance(expr, ast.ConstRef):
    return ctx.node_to_type.get_const_int(expr.name_def, expr.span)
  return _dim_to_parametric(self, expr)


def get_signedness_and_bits(
    type_annotation: ast.BuiltinTypeAnnotation) -> Tuple[bool, int]:
  return scanner.TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS[type_annotation.tok.value]


@_rule(ast.TypeRefTypeAnnotation)
def _deduce_TypeRefTypeAnnotation(self: ast.TypeRefTypeAnnotation,
                                  ctx: DeduceCtx) -> ConcreteType:
  """Dedeuces the concrete type of a TypeRef type annotation."""
  base_type = deduce(self.type_ref, ctx)
  maybe_struct = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
      self.type_ref.type_def, _get_imported_module_via_node_to_type,
      ctx.node_to_type)
  if (isinstance(maybe_struct, ast.Struct) and maybe_struct.is_parametric() and
      self.parametrics):
    base_type = _concretize_struct_annotation(self, maybe_struct, base_type)
  return base_type


@_rule(ast.BuiltinTypeAnnotation)
def _deduce_BuiltinTypeAnnotation(
    self: ast.BuiltinTypeAnnotation,
    ctx: DeduceCtx,  # pylint: disable=unused-argument
) -> ConcreteType:
  signedness, primitive_bits = get_signedness_and_bits(self)
  return BitsType(signedness, primitive_bits)


@_rule(ast.TupleTypeAnnotation)
def _deduce_TupleTypeAnnotation(self: ast.TupleTypeAnnotation,
                                ctx: DeduceCtx) -> ConcreteType:
  members = []
  for member in self.members:
    members.append(deduce(member, ctx))
  return TupleType(tuple(members))


@_rule(ast.ArrayTypeAnnotation)
def _deduce_ArrayTypeAnnotation(self: ast.ArrayTypeAnnotation,
                                ctx: DeduceCtx) -> ConcreteType:
  """Deduces the concrete type of an Array type annotation."""
  dim = _dim_to_parametric_or_int(self, self.dim, ctx)
  if isinstance(
      self.element_type,
      ast.BuiltinTypeAnnotation) and self.element_type.tok.is_keyword_in(
          (scanner.Keyword.BITS, scanner.Keyword.UN, scanner.Keyword.SN)):
    signedness = self.element_type.tok.is_keyword(scanner.Keyword.SN)
    return BitsType(signedness, dim)
  element_type = deduce(self.element_type, ctx)
  result = ArrayType(element_type, dim)
  logging.vlog(4, 'array type annotation: %s => %s', self, result)
  return result


@_rule(ast.ModRef)
def _deduce_ModRef(self: ast.ModRef, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of an entity referenced via module reference."""
  imported_module, imported_node_to_type = ctx.node_to_type.get_imported(
      self.mod)
  leaf_name = self.value_tok.value

  # May be a type definition reference.
  if leaf_name in imported_module.get_typedef_by_name():
    td = imported_module.get_typedef_by_name()[leaf_name]
    if not td.public:
      raise TypeInferenceError(
          self.span,
          type_=None,
          suffix='Attempted to refer to module type that is not public.')
    return imported_node_to_type[td.name]

  # May be a function reference.
  try:
    f = imported_module.get_function(leaf_name)
  except KeyError:
    raise TypeInferenceError(
        self.span,
        type_=None,
        suffix='Module {!r} function {!r} does not exist.'.format(
            imported_module.name, leaf_name))

  if not f.public:
    raise TypeInferenceError(
        self.span,
        type_=None,
        suffix='Attempted to refer to module {!r} function {!r} that is not public.'
        .format(imported_module.name, f.name))
  if f.name not in imported_node_to_type:
    logging.vlog(
        2, 'Function name not in imported_node_to_type; must be parametric: %r',
        f.name)
    assert f.is_parametric()
    # We don't type check parametric functions until invocations.
    # Let's typecheck this imported parametric function with respect to its
    # module (this will only get the type signature, body gets typechecked
    # after parametric instantiation).
    imported_ctx = DeduceCtx(imported_node_to_type, imported_module,
                             ctx.interpret_expr, ctx.check_function_in_module)
    imported_ctx.fn_stack.append(ctx.fn_stack[-1])
    ctx.check_function_in_module(f, imported_ctx)
    ctx.node_to_type.update(imported_ctx.node_to_type)
    imported_node_to_type = imported_ctx.node_to_type
  return imported_node_to_type[f.name]


@_rule(ast.Enum)
def _deduce_Enum(self: ast.Enum, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Enum AST node."""
  resolved_type = resolve(deduce(self.type_, ctx), ctx)
  if not isinstance(resolved_type, BitsType):
    raise XlsTypeError(self.span, resolved_type, None,
                       'Underlying type for an enum must be a bits type.')
  # Grab the bit count of the Enum's underlying type.
  bit_count = resolved_type.get_total_bit_count()
  self.set_signedness(resolved_type.get_signedness())
  result = EnumType(self, bit_count)
  for name, value in self.values:
    # Note: the parser places the type_ from the enum on the value when it is
    # a number, so this deduction flags inappropriate numbers.
    deduce(value, ctx)
    ctx.node_to_type[name] = ctx.node_to_type[value] = result
  ctx.node_to_type[self.name] = ctx.node_to_type[self] = result
  return result


@_rule(ast.Ternary)
def _deduce_Ternary(self: ast.Ternary, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Ternary AST node."""

  test_type = deduce(self.test, ctx)
  resolved_test_type = resolve(test_type, ctx)
  if resolved_test_type != ConcreteType.U1:
    raise XlsTypeError(self.span, resolved_test_type, ConcreteType.U1,
                       'Test type for conditional expression is not "bool"')
  cons_type = deduce(self.consequent, ctx)
  resolved_cons_type = resolve(cons_type, ctx)
  alt_type = deduce(self.alternate, ctx)
  resolved_alt_type = resolve(alt_type, ctx)
  if resolved_cons_type != resolved_alt_type:
    raise XlsTypeError(
        self.span, resolved_cons_type, resolved_alt_type,
        'Ternary consequent type (in the "then" clause) did not match '
        'alternate type (in the "else" clause)')
  return resolved_cons_type


def _deduce_Concat(self: ast.Binop, ctx: DeduceCtx) -> ConcreteType:
  """Deduces the concrete type of a concatenate Binop AST node."""
  lhs_type = deduce(self.lhs, ctx)
  resolved_lhs_type = resolve(lhs_type, ctx)
  rhs_type = deduce(self.rhs, ctx)
  resolved_rhs_type = resolve(rhs_type, ctx)

  # Array-ness must be the same on both sides.
  if (isinstance(resolved_lhs_type, ArrayType) != isinstance(
      resolved_rhs_type, ArrayType)):
    raise XlsTypeError(
        self.span, resolved_lhs_type, resolved_rhs_type,
        'Attempting to concatenate array/non-array values together.')

  if (isinstance(resolved_lhs_type, ArrayType) and
      resolved_lhs_type.get_element_type() !=
      resolved_rhs_type.get_element_type()):
    raise XlsTypeError(
        self.span, resolved_lhs_type, resolved_rhs_type,
        'Array concatenation requires element types to be the same.')

  new_size = resolved_lhs_type.size + resolved_rhs_type.size  # pytype: disable=attribute-error
  if isinstance(resolved_lhs_type, ArrayType):
    return ArrayType(resolved_lhs_type.get_element_type(), new_size)

  return BitsType(signed=False, size=new_size)


@_rule(ast.Binop)
def _deduce_Binop(self: ast.Binop, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of a Binop AST node."""
  # Concatenation is handled differently from other binary operations.
  if self.operator.kind == scanner.TokenKind.DOUBLE_PLUS:
    return _deduce_Concat(self, ctx)

  lhs_type = deduce(self.lhs, ctx)
  rhs_type = deduce(self.rhs, ctx)

  resolved_lhs_type = resolve(lhs_type, ctx)
  resolved_rhs_type = resolve(rhs_type, ctx)

  if resolved_lhs_type != resolved_rhs_type:
    raise XlsTypeError(
        self.span, resolved_lhs_type, resolved_rhs_type,
        'Could not deduce type for binary operation {0} ({0!r}).'.format(
            self.operator))

  # Enums only support a more limited set of binary operations.
  if isinstance(lhs_type,
                EnumType) and self.operator.kind not in self.ENUM_OK_KINDS:
    raise XlsTypeError(
        self.span, resolved_lhs_type, None,
        "Cannot use '{}' on values with enum type {}".format(
            self.operator.kind.value, lhs_type.nominal_type.identifier))

  if self.operator.kind in self.COMPARISON_KINDS:
    return ConcreteType.U1

  return resolved_lhs_type


@_rule(ast.Struct)
def _deduce_Struct(self: ast.Struct, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Returns the concrete type for a (potentially parametric) struct."""
  for parametric in self.parametric_bindings:
    parametric_binding_type = deduce(parametric.type_, ctx)
    assert isinstance(parametric_binding_type, ConcreteType)
    if parametric.expr:
      expr_type = deduce(parametric.expr, ctx)
      if expr_type != parametric_binding_type:
        raise XlsTypeError(
            parametric.span,
            parametric_binding_type,
            expr_type,
            suffix='Annotated type of derived parametric '
            'value did not match inferred type.')
    ctx.node_to_type[parametric.name] = parametric_binding_type

  members = tuple(
      (k.identifier, resolve(deduce(m, ctx), ctx)) for k, m in self.members)
  result = ctx.node_to_type[self.name] = TupleType(members, self)
  logging.vlog(5, 'Deduced type for struct %s => %s; node_to_type: %r', self,
               result, ctx.node_to_type)
  return result


def _validate_struct_members_subset(
    members: ast.StructInstanceMembers, struct_type: ConcreteType,
    struct_text: str, ctx: DeduceCtx
) -> Tuple[Set[str], Tuple[ConcreteType], Tuple[ConcreteType]]:
  """Validates a struct instantiation is a subset of members with no dups.

  Args:
    members: Sequence of members used in instantiation. Note this may be a
      subset; e.g. in the case of splat instantiation.
    struct_type: The deduced type for the struct (instantiation).
    struct_text: Display name to use for the struct in case of an error.
    ctx: Wrapper containing node to type mapping context.

  Returns:
    A tuple containing the set of struct member names that were instantiated,
    the ConcreteTypes of the provided arguments, and the ConcreteTypes of the
    corresponding struct member definition.
  """
  seen_names = set()
  arg_types = []
  member_types = []
  for k, v in members:
    if k in seen_names:
      raise TypeInferenceError(
          v.span,
          type_=None,
          suffix='Duplicate value seen for {!r} in this {!r} struct instance.'
          .format(k, struct_text))
    seen_names.add(k)
    expr_type = resolve(deduce(v, ctx), ctx)
    arg_types.append(expr_type)
    try:
      member_type = struct_type.get_member_type_by_name(k)  # pytype: disable=attribute-error
      member_types.append(member_type)
    except KeyError:
      raise TypeInferenceError(
          v.span,
          None,
          suffix='Struct {!r} has no member {!r}, but it was provided by this instance.'
          .format(struct_text, k))

  return seen_names, tuple(arg_types), tuple(member_types)


@_rule(ast.StructInstance)
def _deduce_StructInstance(
    self: ast.StructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  logging.vlog(5, 'Deducing type for struct instance: %s', self)
  struct_type = deduce(self.struct, ctx)
  expected_names = set(struct_type.tuple_names)  # pytype: disable=attribute-error
  seen_names, arg_types, member_types = _validate_struct_members_subset(
      self.unordered_members, struct_type, self.struct_text, ctx)
  if seen_names != expected_names:
    missing = ', '.join(
        repr(s) for s in sorted(list(expected_names - seen_names)))
    raise TypeInferenceError(
        self.span,
        None,
        suffix='Struct instance is missing member(s): {}'.format(missing))

  struct_def = self.struct
  if not isinstance(struct_def, ast.Struct):
    # Traverse TypeDefs and ModRefs until we get the struct AST node.
    struct_def = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        struct_def, _get_imported_module_via_node_to_type, ctx.node_to_type)
  assert isinstance(struct_def, ast.Struct), struct_def

  resolved_struct_type, _ = parametric_instantiator.instantiate_struct(
      self.span, struct_type, arg_types, member_types, ctx,
      struct_def.parametric_bindings)

  return resolved_struct_type


def _concretize_struct_annotation(type_annotation: ast.TypeRefTypeAnnotation,
                                  struct: ast.Struct,
                                  base_type: ConcreteType) -> ConcreteType:
  """Returns concretized struct type using the provided bindings.

  For example, if we have a struct defined as `struct [N: u32, M: u32] Foo`,
  the default TupleType will be (N, M). If a type annotation provides bindings,
  (e.g. Foo[A, 16]), we will replace N, M with those values. In the case above,
  we will return (A, 16) instead.

  Args:
    type_annotation: The provided type annotation for this parametric struct.
    struct: The corresponding struct AST node.
    base_type: The TupleType of the struct, based only on the struct definition.
  """
  assert len(struct.parametric_bindings) == len(type_annotation.parametrics)
  defined_to_annotated = {}
  for defined_parametric, annotated_parametric in zip(
      struct.parametric_bindings, type_annotation.parametrics):
    assert isinstance(defined_parametric,
                      ast.ParametricBinding), defined_parametric
    if isinstance(annotated_parametric, ast.Number):
      defined_to_annotated[defined_parametric.name.identifier] = \
          int(annotated_parametric.value)
    else:
      assert isinstance(annotated_parametric,
                        ast.NameRef), repr(annotated_parametric)
      defined_to_annotated[defined_parametric.name.identifier] = \
          ParametricSymbol(annotated_parametric.identifier,
                           annotated_parametric.span)

  def resolver(dim):
    if isinstance(dim, ParametricExpression):
      return dim.evaluate(defined_to_annotated)
    return dim

  return base_type.map_size(resolver)


def _get_imported_module_via_node_to_type(
    import_: ast.Import,
    node_to_type: NodeToType) -> Tuple[ast.Module, NodeToType]:
  """Uses node_to_type to retrieve the corresponding module of a ModRef."""
  return node_to_type.get_imported(import_)


@_rule(ast.SplatStructInstance)
def _deduce_SplatStructInstance(
    self: ast.SplatStructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  struct_type = deduce(self.struct, ctx)
  splatted_type = deduce(self.splatted, ctx)

  assert isinstance(struct_type, TupleType), struct_type
  assert isinstance(splatted_type, TupleType), splatted_type

  # We will make sure this splat typechecks during instantiation. Let's just
  # ensure the same number of elements for now.
  assert len(struct_type.tuple_names) == len(splatted_type.tuple_names)

  (seen_names, seen_arg_types,
   seen_member_types) = _validate_struct_members_subset(self.members,
                                                        struct_type,
                                                        self.struct_text, ctx)

  arg_types = list(seen_arg_types)
  member_types = list(seen_member_types)
  for m in struct_type.tuple_names:
    if m not in seen_names:
      splatted_member_type = splatted_type.get_member_type_by_name(m)
      struct_member_type = struct_type.get_member_type_by_name(m)

      arg_types.append(splatted_member_type)
      member_types.append(struct_member_type)

  # At this point, we should have the same number of args compared to the
  # number of members defined in the struct.
  assert len(arg_types) == len(member_types)

  struct_def = self.struct
  if not isinstance(struct_def, ast.Struct):
    # Traverse TypeDefs and ModRefs until we get the struct AST node.
    struct_def = ast_helpers.evaluate_to_struct_or_enum_or_annotation(
        struct_def, _get_imported_module_via_node_to_type, ctx.node_to_type)

  assert isinstance(struct_def, ast.Struct), struct_def

  resolved_struct_type, _ = parametric_instantiator.instantiate_struct(
      self.span, struct_type, tuple(arg_types), tuple(member_types), ctx,
      struct_def.parametric_bindings)

  return resolved_struct_type


@_rule(ast.Attr)
def _deduce_Attr(self: ast.Attr, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of a struct attribute access expression."""
  struct = deduce(self.lhs, ctx)
  if not struct.has_named_member(self.attr.identifier):  # pytype: disable=attribute-error
    raise TypeInferenceError(
        span=self.span,
        type_=None,
        suffix='Struct does not have a member with name {!r}.'.format(
            self.attr))

  return struct.get_named_member_type(self.attr.identifier)  # pytype: disable=attribute-error


def _deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  f = RULES[n.__class__]
  f = typing.cast(Callable[[ast.AstNode, DeduceCtx], ConcreteType], f)
  result = f(n, ctx)
  ctx.node_to_type[n] = result
  return result


def deduce(n: ast.AstNode, ctx: DeduceCtx) -> ConcreteType:
  """Deduces and returns the type of value produced by this expr.

  Also adds n to ctx.node_to_type memoization dictionary.

  Args:
    n: The AST node to deduce the type for.
    ctx: Wraps a node_to_type, a dictionary mapping nodes to their types.

  Returns:
    The type of this expression.

  As a side effect the node_to_type mapping is filled with all the deductions
  that were necessary to determine (deduce) the resulting type of n.
  """
  assert isinstance(n, ast.AstNode), n
  if n in ctx.node_to_type:
    result = ctx.node_to_type[n]
    assert isinstance(result, ConcreteType), result
  else:
    result = ctx.node_to_type[n] = _deduce(n, ctx)
    logging.vlog(5, 'Deduced type of %s => %s', n, result)
    assert isinstance(result, ConcreteType), \
        '_deduce did not return a ConcreteType; got: {!r}'.format(result)
  return result
