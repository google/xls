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

"""Contains the algorithm for instantiating parametric invocation nodes."""

from typing import Any, Text, Dict, Tuple, Optional

from absl import logging

from xls.dslx.concrete_type_helpers import map_size
from xls.dslx.python import cpp_parametric_expression as parametric_expression
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import ConcreteTypeDim
from xls.dslx.python.cpp_concrete_type import EnumType
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_pos import Span
from xls.dslx.symbolic_bindings import SymbolicBindings
from xls.dslx.xls_type_error import ArgCountMismatchError
from xls.dslx.xls_type_error import XlsTypeError

Invocation = Any  # pylint: disable=invalid-name
ParametricBinding = Any
ParametricBindings = Tuple[ParametricBinding, ...]
DeduceCtx = Any
Expr = Any


class _ParametricInstantiator:
  """Abstract base class for parametric instantiation of fns and structs."""

  def __init__(self, span: Span, arg_types: Tuple[ConcreteType,
                                                  ...], ctx: DeduceCtx,
               parametric_constraints: Optional[ParametricBindings]):
    self.span = span
    self.arg_types = arg_types
    self.ctx = ctx
    self.symbolic_bindings = {}  # type: Dict[Text, int]
    self.constraints = {}  # type: Dict[Text, Optional[Expr]]
    self.bit_widths = {}  # type: Dict[Text, int]

    if parametric_constraints:
      for b in parametric_constraints:
        type_ = b.type_
        bit_count = type_.bits
        self.bit_widths[b.name.identifier] = bit_count
        self.constraints[b.name.identifier] = b.expr

  def _verify_constraints(self) -> None:
    """Verifies that all parametrics adhere to signature constraints.

    Take the following function signature for example:
      fn [X: u32, Y: u32 = X + X] f(x: bits[X], y: bits[Y]) -> bits[Y]

    The parametric Y has two constraints based only off the signature:
    it must match the bitwidth of the argument y and it must be equal to
    X + X. This function is responsible for computing any derived parametrics
    and asserting that their values are consistent with other constraints
    (arg types).
    """
    for binding, constraint in self.constraints.items():
      if constraint is None:
        # e.g. [X: u32]
        continue
      try:
        fn_name, fn_symbolic_bindings = self.ctx.fn_stack[-1]
        fn_ctx = (self.ctx.module.name, fn_name,
                  tuple(fn_symbolic_bindings.items()))
        result = self.ctx.interpret_expr(
            self.ctx.module,
            self.ctx.type_info,
            self.symbolic_bindings,
            self.bit_widths,
            constraint,
            fn_ctx=fn_ctx)
      except KeyError as e:
        # We haven't seen enough bindings to evaluate this constraint.
        continue

      if binding in self.symbolic_bindings.keys():
        if result != self.symbolic_bindings[binding]:
          raise XlsTypeError(
              self.span,
              BitsType(signed=False, size=self.symbolic_bindings[binding]),
              BitsType(signed=False, size=result),
              suffix=f'Parametric constraint violated, saw {binding} = {constraint} = {result}; '
              f'then {binding} = {self.symbolic_bindings[binding]}')
      else:
        self.symbolic_bindings[binding] = result

  def _symbolic_bind_dims(self, param_type: ConcreteType,
                          arg_type: ConcreteType) -> None:
    """Binds parametric symbols in param_type according to arg_type."""
    # Create bindings for symbolic parameter dimensions based on argument
    # values passed.
    param_dim = param_type.size.value
    arg_dim = arg_type.size.value
    if not isinstance(param_dim, parametric_expression.ParametricSymbol):
      return

    pdim_name = param_dim.identifier
    if (pdim_name in self.symbolic_bindings and
        self.symbolic_bindings[pdim_name] != arg_dim):
      if self.constraints[pdim_name]:
        # Error on violated constraint.
        raise XlsTypeError(
            self.span,
            BitsType(signed=False, size=self.symbolic_bindings[pdim_name]),
            arg_type,
            suffix=f'Parametric constraint violated, saw {pdim_name} '
            f'= {self.constraints[pdim_name]} '
            f'= {self.symbolic_bindings[pdim_name]}; '
            f'then {pdim_name} = {arg_dim}')
      else:
        # Error on conflicting argument types.
        raise XlsTypeError(
            self.span,
            param_type,
            arg_type,
            suffix='Parametric value {} was bound to different values at '
            'different places in invocation; saw: {!r}; then: {!r}'.format(
                pdim_name, self.symbolic_bindings[pdim_name], arg_dim))

    logging.vlog(2, 'Binding %r to %s', pdim_name, arg_dim)
    self.symbolic_bindings[pdim_name] = arg_dim

  def _symbolic_bind_bits(self, param_type: ConcreteType,
                          arg_type: ConcreteType) -> None:
    """Binds any parametric symbols in the "bits" param_type."""
    assert isinstance(param_type, ConcreteType), repr(param_type)
    assert isinstance(arg_type, ConcreteType), repr(arg_type)
    assert (type(param_type) == type(arg_type)  # pylint: disable=unidiomatic-typecheck
            and isinstance(param_type, (BitsType, EnumType)))

    if isinstance(param_type, EnumType):
      return  # Enums have no size.

    self._symbolic_bind_dims(param_type, arg_type)

  def _symbolic_bind_tuple(self, param_type: ConcreteType,
                           arg_type: ConcreteType) -> None:
    """Binds any parametric symbols in the "tuple" param_type."""
    assert isinstance(param_type, TupleType) and isinstance(arg_type, TupleType)
    for param_member, arg_member in zip(param_type.get_unnamed_members(),
                                        arg_type.get_unnamed_members()):
      self._symbolic_bind(param_member, arg_member)

  def _symbolic_bind_array(self, param_type: ConcreteType,
                           arg_type: ConcreteType) -> None:
    """Binds any parametric symbols in the "array" param_type."""
    assert isinstance(param_type, ArrayType) and isinstance(arg_type, ArrayType)
    self._symbolic_bind(param_type.get_element_type(),
                        arg_type.get_element_type())
    self._symbolic_bind_dims(param_type, arg_type)

  def _symbolic_bind_function(self, param_type: ConcreteType,
                              arg_type: ConcreteType) -> None:
    """Binds any parametric symbols in the "function" param_type."""
    assert isinstance(param_type, FunctionType) and isinstance(
        arg_type, FunctionType)
    for param_param, arg_param in zip(param_type.params, arg_type.params):
      self._symbolic_bind(param_param, arg_param)
    self._symbolic_bind(param_type.return_type, arg_type.return_type)

  def _symbolic_bind(self, param_type: ConcreteType,
                     arg_type: ConcreteType) -> None:
    """Binds symbols present in param_type according to value of arg_type."""
    assert isinstance(param_type, ConcreteType), repr(param_type)
    assert isinstance(arg_type, ConcreteType), repr(arg_type)
    if isinstance(param_type, BitsType):
      self._symbolic_bind_bits(param_type, arg_type)
    elif isinstance(param_type, EnumType):
      assert param_type.get_nominal_type(
          self.ctx.module) == arg_type.get_nominal_type(self.ctx.module)
      # If the enums are the same, we do the same thing as we do with bits
      # (ignore the primitive and symbolic bind the dims).
      self._symbolic_bind_bits(param_type, arg_type)
    elif isinstance(param_type, TupleType):
      param_nominal = param_type.get_nominal_type(self.ctx.module)
      arg_nominal = arg_type.get_nominal_type(self.ctx.module)
      logging.vlog(3, 'param nominal %s arg nominal %s', param_nominal,
                   arg_nominal)
      if param_nominal != arg_nominal:
        raise XlsTypeError(
            self.span,
            param_type,
            arg_type,
            suffix='parameter type name: {}; argument type name: {}.'.format(
                repr(param_nominal.identifier) if param_nominal else '<none>',
                repr(arg_nominal.identifier) if arg_nominal else '<none>'))
      self._symbolic_bind_tuple(param_type, arg_type)
    elif isinstance(param_type, ArrayType):
      self._symbolic_bind_array(param_type, arg_type)
    elif isinstance(param_type, FunctionType):
      self._symbolic_bind_function(param_type, arg_type)
    else:
      raise NotImplementedError('Bind symbols in parameter type {} @ {}'.format(
          param_type, self.span))

  def _instantiate_one_arg(self, i: int, param_type: ConcreteType,
                           arg_type: ConcreteType) -> ConcreteType:
    """Binds param_type via arg_type, updating symbolic bindings."""
    assert isinstance(param_type, ConcreteType), repr(param_type)
    assert isinstance(arg_type, ConcreteType), repr(arg_type)
    # Check parameter and arg types are the same kind.
    if type(param_type) != type(arg_type):  # pylint: disable=unidiomatic-typecheck
      raise XlsTypeError(
          self.span,
          param_type,
          arg_type,
          suffix='Parameter {} and argument types are different kinds '
          '({} vs {}).'.format(i, param_type.get_debug_type_name(),
                               arg_type.get_debug_type_name()))
    logging.vlog(3, 'Symbolically binding param_type %d %s against arg_type %s',
                 i, param_type, arg_type)
    self._symbolic_bind(param_type, arg_type)
    resolved = self._resolve(param_type)
    logging.vlog(3, 'Resolved param_type: %s', resolved)
    return resolved

  def _resolve(self, annotated: ConcreteType) -> ConcreteType:
    """Resolves a parametric type via symbolic_bindings."""

    if self.constraints:
      self._verify_constraints()

    def resolver(dim: ConcreteTypeDim) -> ConcreteTypeDim:
      if isinstance(dim.value, parametric_expression.ParametricExpression):
        before = dim.value
        after = before.evaluate(self.symbolic_bindings)
        logging.vlog(3, 'Resolved parametric expression via %s: %s => %s',
                     self.symbolic_bindings, before, after)
        return ConcreteTypeDim(after)
      return dim

    return map_size(annotated, self.ctx.module, resolver)

  def instantiate(self) -> Tuple[ConcreteType, SymbolicBindings]:
    raise NotImplementedError


class _FunctionInstantiator(_ParametricInstantiator):
  """Instantiates a parametric function invocation.

  Attributes:
    span: Span for the instantiation; e.g. of the invocation AST node being
      instantiated.
    function_type: (Parametric) function type being instantiated.
    arg_types: Argument types presented to the parametric function type.
    symbolic_bindings: Mapping from name to bound value as encountered in the
      instantiation process; e.g. instantiating `fn [N: u32] id(bits[N]) ->
      bits[N]` with a u32 would lead to `{'N': 32}` as the symbolic bindings.
    constraints: Mapping from parametric to its expression.
      e.g. For [X: u32, Y: u32 = X + X], we'd have X -> (32, None) and Y -> (32,
        (X + X)).
    bit_widths: Mapping from parametric to its bit count e.g. From above, X ->
      32 and Y -> 32
    ctx: Wrapper over useful typechecking objects (see deduce.DeduceCtx).
  """

  def __init__(self, span: Span, function_type: ConcreteType,
               arg_types: Tuple[ConcreteType, ...], ctx: DeduceCtx,
               parametric_constraints: Optional[ParametricBindings]):
    super().__init__(span, arg_types, ctx, parametric_constraints)
    self.function_type = function_type
    param_types = self.function_type.params
    if len(self.arg_types) != len(param_types):
      raise ArgCountMismatchError(self.span, arg_types, len(param_types),
                                  param_types,
                                  'Invocation of parametric function.')

  def instantiate(self) -> Tuple[ConcreteType, SymbolicBindings]:
    """Updates symbolic bindings for the parameter types according to arg_types.

    Instantiates the parameters of function_type according to the presented
    arg_types; e.g. when a bits[3,4] argument is passed to a bits[N,M]
    parameter, we note that N=3 and M=4 for resolution in the return type.

    Returns:
      The return type of the function_type, with parametric types instantiated
      in accordance with the presented argument types.
    """
    # Walk through all the params/args to collect symbolic bindings.
    for i, (param_type, arg_type) in enumerate(
        zip(self.function_type.params, self.arg_types)):
      param_type = self._instantiate_one_arg(i, param_type, arg_type)
      logging.vlog(
          3, 'Post-instantiation; paramno: %d; param_type: %s; arg_type: %s', i,
          param_type, arg_type)
      if param_type != arg_type:
        message = 'Mismatch between parameter and argument types.'
        if str(param_type) == str(arg_type):
          message += ' {!r} vs {!r}'.format(param_type, arg_type)
        raise XlsTypeError(self.span, param_type, arg_type, suffix=message)

    # Resolve the return type according to the bindings we collected.
    orig = self.function_type.return_type
    resolved = self._resolve(orig)
    logging.vlog(2, 'Resolved return type from %s to %s', orig, resolved)
    return resolved, tuple(sorted(self.symbolic_bindings.items()))


class _StructInstantiator(_ParametricInstantiator):
  """Instantiates a parametric struct.

  Attributes:
    span: Span for the instantiation; e.g. of the invocation AST node being
      instantiated.
    struct_type: (Parametric) struct type being instantiated.
    arg_types: Argument types being used to instantiate the struct.
    member_types: Member types defined in the struct.
    symbolic_bindings: Mapping from name to bound value as encountered in the
      instantiation process; e.g. instantiating `fn [N: u32] id(bits[N]) ->
      bits[N]` with a u32 would lead to `{'N': 32}` as the symbolic bindings.
    constraints: Mapping from parametric to its expression.
      e.g. For [X: u32, Y: u32 = X + X], we'd have X -> (32, None) and Y -> (32,
        (X + X)).
    bit_widths: Mapping from parametric to its bit count e.g. From above, X ->
      32 and Y -> 32
    ctx: Wrapper over useful typechecking objects (see deduce.DeduceCtx).
  """

  def __init__(self, span: Span, struct_type: ConcreteType,
               arg_types: Tuple[ConcreteType,
                                ...], member_types: Tuple[ConcreteType,
                                                          ...], ctx: DeduceCtx,
               parametric_constraints: Optional[ParametricBindings]):
    super().__init__(span, arg_types, ctx, parametric_constraints)
    self.struct_type = struct_type
    self.member_types = member_types

    if len(self.arg_types) != len(self.member_types):
      raise ArgCountMismatchError(self.span, arg_types, len(member_types),
                                  member_types,
                                  'Invocation of parametric struct.')

  def instantiate(self) -> Tuple[ConcreteType, SymbolicBindings]:
    """Updates symbolic bindings for the member types according to arg_types.

    Instantiates the parameters of struct_type according to the presented
    arg_types; e.g. when a bits[3,4] argument is passed to a bits[N,M]
    parameter, we note that N=3 and M=4 for resolution in the return type.

    Returns:
      The return type of the struct_type, with parametric types instantiated
      in accordance with the presented argument types.
    """
    # Walk through all the members/args to collect symbolic bindings.
    for i, (member_type,
            arg_type) in enumerate(zip(self.member_types, self.arg_types)):
      member_type = self._instantiate_one_arg(i, member_type, arg_type)
      logging.vlog(
          3, 'Post-instantiation; memno: %d; member_type: %s; struct_type: %s',
          i, member_type, arg_type)
      if member_type != arg_type:
        message = 'Mismatch between member and argument types.'
        if str(member_type) == str(arg_type):
          message += ' {!r} vs {!r}'.format(member_type, arg_type)
        raise XlsTypeError(self.span, member_type, arg_type, suffix=message)

    # Resolve the struct type according to the bindings we collected.
    resolved = self._resolve(self.struct_type)
    logging.vlog(3, 'Resolved struct type from %s to %s', self.struct_type,
                 resolved)
    return resolved, tuple(sorted(self.symbolic_bindings.items()))


def instantiate_function(
    span: Span, callee_type: ConcreteType, arg_types: Tuple[ConcreteType, ...],
    ctx: DeduceCtx, parametric_bindings: Optional[ParametricBindings]
) -> Tuple[ConcreteType, SymbolicBindings]:
  """Instantiates a fn invocation using the bindings derived from arg_types."""
  return _FunctionInstantiator(span, callee_type, arg_types, ctx,
                               parametric_bindings).instantiate()


def instantiate_struct(
    span: Span, callee_type: ConcreteType, arg_types: Tuple[ConcreteType, ...],
    member_types: Tuple[ConcreteType, ...], ctx: DeduceCtx,
    parametric_bindings: Optional[ParametricBindings]
) -> Tuple[ConcreteType, SymbolicBindings]:
  """Instantiates a struct using the bindings derived from arg_types."""
  return _StructInstantiator(span, callee_type, arg_types, member_types, ctx,
                             parametric_bindings).instantiate()
