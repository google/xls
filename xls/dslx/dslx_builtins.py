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

# pylint: disable=function-redefined
# pylint: disable=unused-argument
"""Describes builtin functions and how to typecheck the builtins."""

from typing import Text, Tuple, Callable, Union, Optional, Any
from absl import logging

from xls.dslx import parametric_instantiator
from xls.dslx import xls_type_error
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import ConcreteTypeDim
from xls.dslx.python.cpp_concrete_type import FunctionType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_pos import Span

ParametricBinding = Any
ParametricBindings = Tuple[ParametricBinding, ...]
DeduceCtx = Any

# Built-in functions that have a parametric type signature.
_PARAMETRIC_BUILTINS = (
    ('add_with_carry', '(uN[T], uN[T]) -> (u1, uN[T])'),
    ('assert_eq', '(T, T) -> ()'),
    ('assert_lt', '(T, T) -> ()'),
    ('bit_slice', '(uN[T], uN[U], uN[V]) -> uN[V]'),
    ('clz', '(uN[N]) -> uN[N]'),
    ('ctz', '(uN[N]) -> uN[N]'),
    ('concat', '(uN[M], uN[N]) -> uN[M+N]'),
    ('fail!', '(T) -> T'),
    ('map', '(T[N], (T) -> U) -> U[N]'),
    ('one_hot', '(uN[N], u1) -> uN[N+1]'),
    ('one_hot_sel', '(xN[N], xN[M][N]) -> xN[M]'),
    ('rev', '(uN[N]) -> uN[N]'),
    ('select', '(u1, T, T) -> T'),

    # Bitwise reduction ops.
    ('and_reduce', '(uN[N]) -> u1'),
    ('or_reduce', '(uN[N]) -> u1'),
    ('xor_reduce', '(uN[N]) -> u1'),

    # Signed comparisons.
    ('sge', '(xN[N], xN[N]) -> u1'),
    ('sgt', '(xN[N], xN[N]) -> u1'),
    ('sle', '(xN[N], xN[N]) -> u1'),
    ('slt', '(xN[N], xN[N]) -> u1'),

    # Use a dummy value to determine size.
    ('signex', '(xN[M], xN[N]) -> xN[N]'),
    ('slice', '(T[M], uN[N], T[P]) -> T[P]'),
    ('trace', '(T) -> T'),
    ('update', '(T[N], uN[M], T) -> T[N]'),
    ('enumerate', '(T[N]) -> (u32, T)[N]'),

    # Require-const-argument.
    #
    # Note this is messed up and should be replaced with known-statically-sized
    # iota syntax.
    ('range', '(uN[N], uN[N]) -> ()'),
)

PARAMETRIC_BUILTIN_NAMES = frozenset({t[0] for t in _PARAMETRIC_BUILTINS})
_PARAMETRIC_NAME_TO_SIGNATURE = {t[0]: t[1] for t in _PARAMETRIC_BUILTINS}

# Set of unary builtins appropriate as functions - that transform values.
# TODO(b/144724970): Add enumerate here (and maybe move to ir_converter.py).
UNARY_BUILTIN_NAMES = frozenset(('clz', 'ctz'))

ArgTypes = Tuple[ConcreteType, ...]


class _Checker(object):
  """Fluent API for checking argument type properties (and raising errors)."""

  def __init__(self, arg_types: ArgTypes, name: Text, span: Span):
    self.arg_types = arg_types
    self.name = name
    self.span = span

  def len(self, target: int) -> '_Checker':
    if len(self.arg_types) != target:
      raise xls_type_error.ArgCountMismatchError(
          self.span, self.arg_types, target, None,
          'Invalid number of arguments passed to {!r}'.format(self.name))
    return self

  def eq(self, lhs: ConcreteType, rhs: ConcreteType, fmt: Text) -> '_Checker':
    if lhs != rhs:
      raise xls_type_error.XlsTypeError(self.span, lhs, rhs,
                                        fmt.format(lhs, rhs))
    return self

  def is_fn(self, argno: int, argc: int) -> '_Checker':
    """Checks arg argno is a function with argc parameters."""
    t = self.arg_types[argno]
    if not isinstance(t, FunctionType):
      raise xls_type_error.XlsTypeError(
          self.span, t, None,
          'Want argument {} to be a function; got {}'.format(argno, t))
    if len(t.params) != argc:
      raise xls_type_error.XlsTypeError(
          self.span, t, None,
          'Want argument {} to be a function with {} parameters; got {}'.format(
              argno, argc, t))
    return self

  def is_array(self, argno: int) -> '_Checker':
    t = self.arg_types[argno]
    if not isinstance(t, ArrayType):
      raise xls_type_error.XlsTypeError(
          self.span, t, None,
          'Want argument {} to be an array; got {}'.format(argno, t))
    return self

  def is_bits(self, argno: Union[int, Tuple[int, ...]]) -> '_Checker':
    """Checks all args identified by argno are bits types."""
    if isinstance(argno, tuple):
      for a in argno:
        self.is_bits(a)
      return self

    t = self.arg_types[argno]
    if not isinstance(t, BitsType):
      raise xls_type_error.XlsTypeError(
          self.span, t, None,
          'Want argument {} to be bits; got {}'.format(argno, t))
    return self

  def is_uN(self, argno: Union[int, Tuple[int, ...]]) -> '_Checker':  # pylint: disable=invalid-name
    """Checks all args identified by argno are uN types."""
    if isinstance(argno, tuple):
      for a in argno:
        self.is_uN(a)
      return self

    assert isinstance(argno, int), argno
    t = self.arg_types[argno]
    if not isinstance(t, BitsType) or t.signed:
      raise xls_type_error.XlsTypeError(
          self.span, t, None,
          'Want argument {} to be unsigned bits; got {}'.format(argno, t))
    return self

  def check_is_bits(self, t: ConcreteType, fmt: Text) -> '_Checker':
    if not isinstance(t, BitsType):
      raise xls_type_error.XlsTypeError(self.span, t, None, fmt.format(t))
    return self

  def check_is_len(self, t: ArrayType, target: int, fmt: str) -> '_Checker':
    if t.size != target:
      raise xls_type_error.XlsTypeError(self.span, t, None,
                                        fmt.format(t=t, target=target))
    return self

  def check_is_same(self, t: ConcreteType, u: ConcreteType,
                    fmt: Text) -> '_Checker':
    if t != u:
      raise xls_type_error.XlsTypeError(self.span, t, u, fmt.format(t, u))
    return self

  def is_u1(self, argno: int) -> '_Checker':
    return self.eq(self.arg_types[argno], ConcreteType.U1,
                   'Expected argument %d to be a u1; got {0}' % argno)

  def is_same(self, argno0: int, argno1: int) -> '_Checker':
    t, u = self.arg_types[argno0], self.arg_types[argno1]
    return self.check_is_same(
        t, u, 'Want arguments %d and %d to be of the same type; got {} vs {}' %
        (argno0, argno1))


# Here we register functions that convert the type signatures of builtin
# functions (as described in the table above) into their ConcreteType form.
#
# Note that some of these may also create symbolic bindings that are returned
# along with the concrete type; e.g. map which determines the symbolic bindings
# for the mapped function.
_FSIGNATURE_REGISTRY = {}


def register_fsignature(signature: Text):
  """Registers conversion function for the given signature."""

  def do_register(f):  # Places f in the registry associated with signature.
    _FSIGNATURE_REGISTRY[signature] = f
    return f

  return do_register


# TODO(leary): 2019-12-12 These *could* be automatically made by interpreting
# the signature string, but just typing in the limited set we use is easier for
# now.


@register_fsignature('(uN[T], uN[T]) -> (u1, uN[T])')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_uN(0).is_same(0, 1)
  return_type = TupleType((ConcreteType.U1, arg_types[0]))
  return FunctionType(arg_types, return_type)


@register_fsignature('(u1, T, T) -> T')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(3).is_u1(0).is_same(1, 2)
  return FunctionType(arg_types, arg_types[1])


@register_fsignature('(T) -> T')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(1)
  return FunctionType(arg_types, arg_types[0])


@register_fsignature('(T, T) -> ()')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_same(0, 1)
  return FunctionType(arg_types, ConcreteType.NIL)


@register_fsignature('(uN[N]) -> u1')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(1).is_uN(0)
  return FunctionType(arg_types, ConcreteType.U1)


@register_fsignature('(uN[N]) -> uN[N]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(1).is_uN(0)
  return FunctionType(arg_types, arg_types[0])


@register_fsignature('(uN[N], uN[N]) -> ()')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_uN((0, 1)).is_same(0, 1)
  return FunctionType(arg_types, ConcreteType.NIL)


@register_fsignature('(xN[N], xN[M][N]) -> xN[M]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  """Checks the type signature shown above and returns deduced output type."""
  checker = _Checker(arg_types, name, span).len(2).is_bits(0).is_array(1)

  arg0 = arg_types[0]
  arg1 = arg_types[1]
  assert isinstance(arg1, ArrayType), arg1
  assert isinstance(arg1.size.value, int), arg1
  return_type = arg1.element_type
  checker.check_is_bits(return_type,
                        'Want arg 1 element type to be bits; got {0}')
  checker.check_is_len(arg1, arg0.size,
                       'bit width {target} must match {t} array size {t.size}')
  return FunctionType(arg_types, return_type)


@register_fsignature('(T[N], uN[M], T) -> T[N]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  checker = _Checker(arg_types, name, span).len(3).is_array(0).is_uN(1)
  checker.check_is_same(
      arg_types[0].get_element_type(), arg_types[2],  # pytype: disable=attribute-error
      'Want argument 0 element type {0} to match argument 2 type {1}')
  return FunctionType(arg_types, arg_types[0])


@register_fsignature('(T[N]) -> (u32, T)[N]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(1).is_array(0)
  t = arg_types[0].get_element_type()  # pytype: disable=attribute-error
  e = TupleType((ConcreteType.U32, t))
  return_type = ArrayType(e, arg_types[0].size)  # pytype: disable=attribute-error
  return FunctionType(arg_types, return_type)


@register_fsignature('(xN[N], xN[N]) -> u1')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_bits(0).is_same(0, 1)
  return FunctionType(arg_types, ConcreteType.U1)


@register_fsignature('(xN[M], xN[N]) -> xN[N]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_bits((0, 1))
  return FunctionType(arg_types, arg_types[1])


@register_fsignature('(T[M], uN[N], T[P]) -> T[P]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  checker = _Checker(arg_types, name,
                     span).len(3).is_array(0).is_uN(1).is_array(2)
  checker.eq(
      arg_types[0].get_element_type(), arg_types[2].get_element_type(),  # pytype: disable=attribute-error
      'Element type of argument 0 {0} should match element type of argument 2 {1}'
  )
  return FunctionType(arg_types, arg_types[2])


@register_fsignature('(uN[N], u1) -> uN[N+1]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_uN(0).is_u1(1)
  return_type = BitsType(
      signed=False,
      size=arg_types[0].get_total_bit_count() + ConcreteTypeDim(1))
  return FunctionType(arg_types, return_type)


@register_fsignature('(uN[N], uN[N]) -> ()')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_uN((0, 1))
  return FunctionType(arg_types, ConcreteType.NIL)


@register_fsignature('(uN[M], uN[N]) -> uN[M+N]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(2).is_uN((0, 1))
  return_type = BitsType(
      signed=False,
      size=arg_types[0].get_total_bit_count() +
      arg_types[1].get_total_bit_count())
  return FunctionType(arg_types, return_type)


@register_fsignature('(uN[T], uN[U], uN[V]) -> uN[V]')
def fsig(arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
         _: Optional[ParametricBindings]) -> ConcreteType:
  _Checker(arg_types, name, span).len(3).is_uN((0, 1, 2))
  return FunctionType(arg_types, arg_types[2])


@register_fsignature('(T[N], (T) -> U) -> U[N]')
def fsig(
    arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
    parametric_bindings: Optional[ParametricBindings]
) -> Tuple[ConcreteType, parametric_instantiator.SymbolicBindings]:
  """Returns the inferred/checked return type for a map-style signature."""
  logging.vlog(5, 'Instantiating for builtin %r @ %s', name, span)
  _Checker(arg_types, name, span).len(2).is_array(0).is_fn(1, argc=1)
  t = arg_types[0].get_element_type()  # pytype: disable=attribute-error
  u, symbolic_bindings = parametric_instantiator.instantiate_function(
      span, arg_types[1], (t,), ctx, parametric_bindings)
  return_type = ArrayType(u, arg_types[0].size)  # pytype: disable=attribute-error
  return FunctionType(arg_types, return_type), symbolic_bindings


SignatureFn = Callable[
    [ArgTypes, Text, Span, DeduceCtx, Optional[ParametricBindings]],
    Tuple[ConcreteType, parametric_instantiator.SymbolicBindings]]


def get_fsignature(builtin_name: Text) -> SignatureFn:
  """Returns a function that produces the type of builtin_name.

  Many builtins are parametric, and so the function type is determined (or type
  errors are raised) based on the types that are presented to it as arguments.

  The returned function is then invoked as:

      fsignature = get_fsignature(builtin_name)
      fn_type, symbolic_bindings = fsignature(arg_types, builtin_name,
                                              invocation_span)

  Where the second line provides the argument types presented to the builtin.

  This is similar conceptually to type deduction, just the builtin functions
  have no definitions in the source code, and sometimes we do fancier rules than
  we currently have support for in the DSL. As parametric support grows,
  however, one day these may all be a special "builtin" module.

  Args:
    builtin_name: The name of the builtin that we want the function for.
  """
  signature = _PARAMETRIC_NAME_TO_SIGNATURE[builtin_name]
  f = _FSIGNATURE_REGISTRY[signature]

  # Since most of the functions don't need to provide symbolic bindings we make
  # a little wrapper that provides trivially empty ones to alleviate the typing
  # burden.
  def wrapper(
      arg_types: ArgTypes, name: Text, span: Span, ctx: DeduceCtx,
      parametric_bindings: Optional[ParametricBindings]
  ) -> Tuple[ConcreteType, parametric_instantiator.SymbolicBindings]:
    result = f(arg_types, name, span, ctx, parametric_bindings)
    if isinstance(result, tuple):
      return result
    assert isinstance(result, ConcreteType), result
    return result, ()

  return wrapper


# Sanity check we have all the converters we need for the builtins.
def _check_registry():
  for name, sig in _PARAMETRIC_BUILTINS:
    assert sig in _FSIGNATURE_REGISTRY, (name, sig)


_check_registry()
