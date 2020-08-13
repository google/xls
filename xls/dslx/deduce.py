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
from typing import Text, Dict, Union, Callable, Type, Tuple, List, Set

from absl import logging
import dataclasses

from xls.dslx import ast
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

  Also raises a TypeMissingError instead of a KeyError when we encounter a node
  that does not have a type known, so that it can be handled in a more specific
  way versus a KeyError.
  """

  def __init__(self):
    self._dict = {}  # type: Dict[ast.AstNode, ConcreteType]
    self._imports = {}  # type: Dict[ast.Import, ImportedInfo]
    self._name_to_const = {}  # type: Dict[ast.NameDef, ast.Constant]
    self.parametric_fn_cache = {}

  def update(self, other: 'NodeToType') -> None:
    self._dict.update(other._dict)  # pylint: disable=protected-access
    self._imports.update(other._imports)  # pylint: disable=protected-access
    self.parametric_fn_cache.update(other.parametric_fn_cache)

  def add_import(self, import_node: ast.Import, info: ImportedInfo) -> None:
    assert import_node not in self._imports, import_node
    self._imports[import_node] = info
    self.update(info[1])

  def note_constant(self, name_def: ast.NameDef, constant: ast.Constant):
    self._name_to_const[name_def] = constant

  def get_const_int(self, name_def: ast.NameDef, user_span: span.Span) -> int:
    constant = self._name_to_const[name_def]
    if isinstance(constant.value, ast.Number):
      return constant.value.get_value_as_int()
    raise TypeInferenceError(
        span=user_span,
        type_=None,
        suffix='Expected to find a constant integral value with the name {};'
        'got: {}'.format(name_def, constant.value))

  def get_imports(self) -> Dict[ast.Import, ImportedInfo]:
    return self._imports

  def get_imported(self, import_node: ast.Import) -> ImportedInfo:
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
      return self._dict[k]
    except KeyError:
      span_suffix = ' @ {}'.format(k.span) if hasattr(k, 'span') else ''
      raise TypeMissingError(
          k, suffix='resolving type of node{}'.format(span_suffix))

  def __contains__(self, k: ast.AstNode) -> bool:
    return k in self._dict


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

# Maps (module_name, parametric function node) to (node -> type)
ParametricFnCache = Dict[ast.Function, Dict[ast.AstNode, ConcreteType]]

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
    parametric_fn_cache: Maps a parametric_fn to all the nodes that it is
      dependent on. Used in typecheck.py to trick the typechecker into checking
      the body of a parametric fn again (per instantiation).
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


def _create_element_invocation(span_: span.Span, callee: Union[ast.NameRef,
                                                               ast.ModRef],
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
    span_: The location in the code where analysis is occurring.
    callee: The function to be invoked.
    arg_array: The array of arguments (at least one) to the function.

  Returns:
    An invocation node for the given function when called with an element in the
    argument array.
  """
  annotation = ast.TypeAnnotation(
      span_, scanner.Token(scanner.TokenKind.KEYWORD, span_,
                           scanner.Keyword.U32), ())
  index_number = ast.Number(
      scanner.Token(scanner.TokenKind.KEYWORD, span_, '32'), annotation)
  index = ast.Index(span_, arg_array, index_number)
  return ast.Invocation(span_, callee, (index,))


def _check_parametric_invocation(parametric_fn: ast.Function,
                                 invocation: ast.Invocation,
                                 symbolic_bindings: SymbolicBindings,
                                 ctx: DeduceCtx):
  """Checks the parametric fn body using the invocation's symbolic bindings."""
  if isinstance(invocation.callee, ast.ModRef):
    # We need to typecheck this function with respect to its own module.
    # Let's use typecheck._check_function_or_test_in_module() to do this
    # in case we run into more dependencies in that module
    imported_module, imported_node_to_type = ctx.node_to_type.get_imported(
        invocation.callee.mod)
    imported_ctx = DeduceCtx(imported_node_to_type, imported_module,
                             ctx.interpret_expr, ctx.check_function_in_module)
    imported_ctx.fn_stack.append(
        (parametric_fn.name.identifier, dict(symbolic_bindings)))
    ctx.check_function_in_module(parametric_fn, imported_ctx)
    ctx.node_to_type.update(imported_ctx.node_to_type)
    ctx.node_to_type.parametric_fn_cache.update(
        imported_ctx.node_to_type.parametric_fn_cache)
  else:
    assert isinstance(invocation.callee, ast.NameRef), invocation.callee
    # We need to typecheck this function with respect to its own module
    # Let's take advantage of the existing try-catch mechanism in
    # typecheck._check_function_or_test_in_module()
    ctx.fn_stack.append(
        (parametric_fn.name.identifier, dict(symbolic_bindings)))

    # If the body of this function hasn't been typechecked, let's
    # tell typecheck.py's handler to check it.
    try:
      body_return_type = ctx.node_to_type[parametric_fn.body]
    except TypeMissingError as e:
      e.node = invocation.callee.name_def
      raise

    ctx.fn_stack.pop()

    # TODO(hjmontero): 2020-07-13 HACK: We remove the type of the body to so
    # that we re-typecheck it if we see this invocation again.
    ctx.node_to_type._dict.pop(parametric_fn.body)  # pylint: disable=protected-access


@_rule(ast.Invocation)
def _deduce_Invocation(self: ast.Invocation, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the concrete type of an Invocation AST node."""
  logging.vlog(5, 'Deducing type for invocation: %s', self)
  arg_types = []
  fn_name, fn_symbolic_bindings = ctx.fn_stack[-1]
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
        invocation = _create_element_invocation(self.span, arg, self.args[0])
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
    callee_name = self.callee.tok.value
    callee_fn = ctx.module.get_function(callee_name)

  self_type, callee_sym_bindings = parametric_instantiator.instantiate(
      self.span, callee_type, tuple(arg_types), ctx,
      callee_fn.parametric_bindings)

  # Within the context of (mod_name, fn_name, fn_sym_bindings),
  # this invocation of callee will have bindings with values specified by
  # callee_sym_bindings
  self.symbolic_bindings[(
      ctx.module.name, fn_name,
      tuple(fn_symbolic_bindings.items()))] = callee_sym_bindings

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
  start, width = bit_helpers.resolve_bit_slice_indices(bit_count, start, limit,
                                                       fn_symbolic_bindings)
  index_slice.computed_start = start
  index_slice.computed_width = width
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

  if self.type_ is not None:
    concrete_type = deduce(self.type_, ctx)

    resolved_rhs_type = resolve(rhs_type, ctx)
    resolved_concrete_type = resolve(concrete_type, ctx)

    if resolved_rhs_type != resolved_concrete_type:
      raise XlsTypeError(
          self.rhs.span, resolved_concrete_type, resolved_rhs_type,
          'Annotated type did not match inferred type of right hand side.')

  _bind_names(self.name_def_tree, rhs_type, ctx)

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
      ctx.node_to_type[leaf] = type_
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
  return arm_types[0]


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
  return init_type


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
  return init_type


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
  return type_result


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

  inferred = ArrayType(member_types[0], len(member_types))

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
    if self.tok.is_keyword_in((scanner.Keyword.TRUE, scanner.Keyword.FALSE)):
      return ConcreteType.U1
    if self.tok.kind == scanner.TokenKind.CHARACTER:
      return ConcreteType.U8
    raise TypeInferenceError(
        span=self.span,
        type_=None,
        suffix='Could not infer a type for this number, please annotate a type.'
    )
  concrete_type = deduce(self.type_, ctx)
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


def _concretize_TypeAnnotation(
    self: ast.TypeAnnotation,
    ctx: DeduceCtx) -> ConcreteType[Union[int, ParametricExpression]]:
  """Converts this AST-level type definition into a concrete type."""
  if self.is_tuple():
    return TupleType(
        tuple(_concretize_TypeAnnotation(e, ctx) for e in self.tuple_members))

  def resolve_dim(i: int) -> Union[int, ParametricExpression]:
    """Resolves dims in 'self' to concrete integers or parametric AST nodes."""
    dim = self.dims[i]
    if isinstance(dim, ast.Number):
      return dim.get_value_as_int()
    else:  # It's not a number, so convert it to parametric AST nodes.
      if isinstance(dim, ast.ConstRef):
        return ctx.node_to_type.get_const_int(dim.name_def, dim.span)
      if isinstance(dim, ast.NameRef):
        ctx.node_to_type[dim] = ctx.node_to_type[dim.name_def]
      return _dim_to_parametric(self, dim)

  if self.is_typeref():
    base_type = deduce(self.get_typeref(), ctx)
    logging.vlog(5, 'base type for typeref: %s', base_type)
    if not self.has_dims():
      return base_type

    for i in reversed(range(len(self.dims))):
      base_type = ArrayType(base_type, resolve_dim(i))
    return base_type

  # Append the datatype bit count to the dims as a minormost dimension.
  if isinstance(self.primitive, scanner.Token) and self.primitive.is_keyword_in(
      (scanner.Keyword.BITS, scanner.Keyword.UN, scanner.Keyword.SN)):
    signedness = self.primitive.is_keyword(scanner.Keyword.SN)
    t = BitsType(signedness, resolve_dim(-1))
    for i, _ in reversed(list(enumerate(self.dims[:-1]))):
      t = ArrayType(t, resolve_dim(i))
    return t

  signedness, primitive_bits = self.primitive_to_signedness_and_bits()
  t = BitsType(signedness, primitive_bits)
  for i, _ in reversed(list(enumerate(self.dims))):
    t = ArrayType(t, resolve_dim(i))
  return t


@_rule(ast.TypeAnnotation)
def _deduce_TypeAnnotation(
    self: ast.TypeAnnotation,  # pytype: disable=wrong-arg-types
    ctx: DeduceCtx) -> ConcreteType:
  return _concretize_TypeAnnotation(self, ctx)


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
  deduce(self.type_, ctx)
  # Grab the bit count of the Enum's underlying type.
  bit_count = ctx.node_to_type[self.type_].get_total_bit_count()
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
  return cons_type


def _deduce_Concat(self: ast.Binop, ctx: DeduceCtx) -> ConcreteType:
  """Deduces the concrete type of a concatenate Binop AST node."""
  lhs_type = deduce(self.lhs, ctx)
  rhs_type = deduce(self.rhs, ctx)

  # Array-ness must be the same on both sides.
  if isinstance(lhs_type, ArrayType) != isinstance(rhs_type, ArrayType):
    raise XlsTypeError(
        self.span, lhs_type, rhs_type,
        'Attempting to concatenate array/non-array values together.')

  if (isinstance(lhs_type, ArrayType) and
      lhs_type.get_element_type() != rhs_type.get_element_type()):
    raise XlsTypeError(
        self.span, lhs_type, rhs_type,
        'Array concatenation requires element types to be the same.')

  new_size = lhs_type.size + rhs_type.size  # pytype: disable=attribute-error
  if isinstance(lhs_type, ArrayType):
    return ArrayType(lhs_type.get_element_type(), new_size)

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
        self.span, lhs_type, None,
        "Cannot use '{}' on values with enum type {}".format(
            self.operator.kind.value, lhs_type.nominal_type.identifier))

  if self.operator.kind in self.COMPARISON_KINDS:
    return ConcreteType.U1

  return lhs_type


@_rule(ast.Struct)
def _deduce_Struct(self: ast.Struct, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  members = tuple((k.identifier, deduce(m, ctx)) for k, m in self.members)
  result = ctx.node_to_type[self.name] = TupleType(members, self)
  logging.vlog(5, 'Deduced type for struct %s => %s; node_to_type: %r', self,
               result, ctx.node_to_type)
  return result


def _typecheck_struct_members_subset(members: ast.StructInstanceMembers,
                                     struct_type: ConcreteType,
                                     struct_text: str,
                                     ctx: DeduceCtx) -> Set[str]:
  """Validates a struct instantiation is a subset of members with no dups.

  Args:
    members: Sequence of members used in instantiation. Note this may be a
      subset; e.g. in the case of splat instantiation.
    struct_type: The deduced type for the struct (instantiation).
    struct_text: Display name to use for the struct in case of an error.
    ctx: Wrapper containing node to type mapping context.

  Returns:
    The set of struct member names that were instantiated.
  """
  seen_names = set()
  for k, v in members:
    if k in seen_names:
      raise TypeInferenceError(
          v.span,
          type_=None,
          suffix='Duplicate value seen for {!r} in this {!r} struct instance.'
          .format(k, struct_text))
    seen_names.add(k)
    expr_type = deduce(v, ctx)
    try:
      member_type = struct_type.get_member_type_by_name(k)  # pytype: disable=attribute-error
    except KeyError:
      raise TypeInferenceError(
          v.span,
          None,
          suffix='Struct {!r} has no member {!r}, but it was provided by this instance.'
          .format(struct_text, k))
    if member_type != expr_type:
      raise XlsTypeError(
          v.span, member_type, expr_type,
          'Member type for {!r} ({}) does not match expression type {}.'.format(
              k, member_type, expr_type))
  return seen_names


@_rule(ast.StructInstance)
def _deduce_StructInstance(
    self: ast.StructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  logging.vlog(5, 'Deducing type for struct instance: %s', self)
  struct_type = deduce(self.struct, ctx)
  expected_names = set(struct_type.tuple_names)  # pytype: disable=attribute-error
  seen_names = _typecheck_struct_members_subset(self.unordered_members,
                                                struct_type, self.struct_text,
                                                ctx)
  if seen_names != expected_names:
    missing = ', '.join(
        repr(s) for s in sorted(list(expected_names - seen_names)))
    raise TypeInferenceError(
        self.span,
        None,
        suffix='Struct instance is missing member(s): {}'.format(missing))
  return struct_type


@_rule(ast.SplatStructInstance)
def _deduce_SplatStructInstance(
    self: ast.SplatStructInstance, ctx: DeduceCtx) -> ConcreteType:  # pytype: disable=wrong-arg-types
  """Deduces the type of the struct instantiation expression and its members."""
  struct_type = deduce(self.struct, ctx)
  splatted_type = deduce(self.splatted, ctx)
  if splatted_type != struct_type:
    raise XlsTypeError(
        self.splatted.span, struct_type, splatted_type,
        'Splatted expression must have the same type as the struct being instantiated.'
    )
  _typecheck_struct_members_subset(self.members, struct_type, self.struct_text,
                                   ctx)
  return struct_type


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
