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
"""AST generator for fuzzing."""

import collections
import math
import random

from typing import Tuple, TypeVar, Text, Dict, Callable, Sequence, Optional
from xls.common import memoize
from xls.dslx import ast_helpers
from xls.dslx import bit_helpers
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_scanner as scanner
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span

T = TypeVar('T')
U = TypeVar('U')
Random = random.Random
Env = Dict[Text, Tuple[Callable[[], ast.Expr], ast.TypeAnnotation, bool]]
del random  # Avoids accidentally calling module functions, use object methods.


class EmptyEnvError(Exception):
  """Raised when attempting to sample from an empty environment."""


@memoize.memoize
def make_bit_patterns(bit_count: int) -> Tuple[int, ...]:
  """Creates a sequence of interesting bit patterns at a given bit count."""
  if bit_count == 0:
    return (0,)
  all_ones = int('0b' + '1' * bit_count, 2)
  all_but_high_ones = int('0b0' + '1' * (bit_count - 1), 2)
  off_on = int(
      '0b' + '01' * (bit_count // 2) + ('0' if bit_count % 2 != 0 else ''), 2)
  on_off = int(
      '0b' + '10' * (bit_count // 2) + ('1' if bit_count % 2 != 0 else ''), 2)
  one_hots = tuple(
      int('0b{}1{}'.format('0' * pos, '0' * (bit_count - pos - 1)), 2)
      for pos in range(bit_count))
  zero = 0
  result = (all_ones, all_but_high_ones, off_on, on_off, zero) + one_hots
  assert all(x & bit_helpers.to_mask(bit_count) == x for x in result)
  return result


def builtin_type_to_bits(type_annotation: ast.TypeAnnotation) -> int:
  """Converts a type annotation (array of builtin or builtin) to a bit count."""
  if isinstance(type_annotation, ast.ArrayTypeAnnotation):
    assert isinstance(type_annotation.element_type, ast.BuiltinTypeAnnotation)
    dim = type_annotation.dim
    assert isinstance(dim, ast.Number), dim
    return builtin_type_to_bits(
        type_annotation.element_type) * ast_helpers.get_value_as_int(dim)

  assert isinstance(type_annotation,
                    ast.BuiltinTypeAnnotation), repr(type_annotation)
  return type_annotation.bits


class AstGeneratorOptions(object):
  """Options that are used to configure the AST generator."""

  def __init__(self,
               emit_signed_types: bool = True,
               disallow_divide: bool = False,
               max_width_bits_types: int = 64,
               max_width_aggregate_types: int = 1024,
               emit_loops: bool = True,
               binop_allowlist: Optional[Sequence[ast.BinopKind]] = None,
               short_samples: bool = False):
    self.disallow_divide = disallow_divide
    self.max_width_bits_types = max_width_bits_types
    self.max_width_aggregate_types = max_width_aggregate_types
    self.binop_allowlist = binop_allowlist
    self.emit_loops = emit_loops
    self.short_samples = short_samples
    self.emit_signed_types = emit_signed_types


class AstGenerator(object):
  """RNG-based AST fuzz generator."""

  def __init__(self,
               rng: Random,
               options: AstGeneratorOptions,
               codegen_ops_only: bool = True):
    self.options = options
    self.rng = rng
    # Should we only generate ops that can be codegenned?
    self._codegen_ops_only = codegen_ops_only
    self.fake_pos = Pos('<fake>', 0, 0)
    self.fake_span = Span(self.fake_pos, self.fake_pos)
    self.name_generator = self._name_generator()
    if options.binop_allowlist:
      assert all(
          binop in ast_helpers.BINOP_SAME_TYPE_KIND_LIST
          for binop in options.binop_allowlist
      ), 'Contains invalid TokenKinds for same-type binop allowlist: {}'.format(
          options.binop_allowlist)
      self._binops = options.binop_allowlist
    else:
      self._binops = list(ast_helpers.BINOP_SAME_TYPE_KIND_LIST)
      if options.disallow_divide:
        self._binops.remove(ast.BinopKind.DIV)

    type_kws = set(scanner.TYPE_KEYWORD_STRINGS) - set(['bits', 'uN', 'sN'])
    if not options.emit_signed_types:
      type_kws = {kw for kw in type_kws if not kw.startswith('s')}

    def kw_width(kw):
      if kw == 'bool':
        return 1
      # Keyword should be of uN or sN form.
      return int(kw[1:])

    type_kws = {
        kw for kw in type_kws
        if kw_width(kw) <= self.options.max_width_bits_types
    }
    self._kw_identifiers = sorted(list(type_kws))

    # Set of functions created during generation.
    self._functions = []

    # Set of types defined during module generation.
    self._type_defs = []

    # Widths of the aggregate types, indexed by str(TypeAnnotation).
    self._type_bit_counts = {}

  def reset(self, seed: int):
    self.name_generator = self._name_generator()
    self.rng.seed(seed)
    self._functions = []
    self._type_defs = []

  def _name_generator(self):
    i = 0
    while True:
      yield 'x{}'.format(i)
      i += 1

  def gensym(self) -> Text:
    return next(self.name_generator)

  def should_nest(self, level: int) -> bool:
    value = self.rng.gammavariate(1.0 if self.options.short_samples else 7.0,
                                  5.0)
    return value >= level

  def randrange(self, limit: int) -> int:
    return self.rng.randrange(limit)

  def _generate_type_primitive(self) -> scanner.Token:
    """Generates a primitive type token for use in building a type."""
    kw_identifier = self.rng.choice(self._kw_identifiers)
    return scanner.Token(self.fake_span,
                         scanner.KeywordFromString(kw_identifier))

  def _make_type_annotation(self, signed: bool,
                            width: int) -> ast.TypeAnnotation:
    assert width > 0, width
    if width <= 64:
      return ast.BuiltinTypeAnnotation(
          self.m, self.fake_span, ast_helpers.get_builtin_type(signed, width))
    raise NotImplementedError(signed, width)

  def _get_type_bit_count(self, type_: ast.TypeAnnotation) -> int:
    """Returns the bit count of the given type."""
    if isinstance(type_, ast.BuiltinTypeAnnotation):
      return type_.bits
    if isinstance(type_, ast.ArrayTypeAnnotation):
      return self._get_type_bit_count(type_.element_type)
    assert str(type_) in self._type_bit_counts, (str(type_),
                                                 self._type_bit_counts)
    return self._type_bit_counts[str(type_)]

  def _make_tuple_type(
      self, members: Tuple[ast.TypeAnnotation, ...]) -> ast.TypeAnnotation:
    """Creates a tuple type with the given `members`."""
    tuple_type = ast.TupleTypeAnnotation(self.m, self.fake_span, members)
    self._type_bit_counts[str(tuple_type)] = sum(
        self._get_type_bit_count(t) for t in members)
    return tuple_type

  def _make_array_type(self, element_type: ast.TypeAnnotation,
                       array_size: int) -> ast.TypeAnnotation:
    """Creates an array type with the given size and element type."""
    array_type = ast_helpers.make_type_ref_type_annotation(
        self.m, self.fake_span, self._create_type_ref(element_type),
        (self._make_number(array_size, None),))
    self._type_bit_counts[str(
        array_type)] = self._get_type_bit_count(element_type) * array_size
    return array_type

  def _get_array_size(self, array_type: ast.TypeAnnotation) -> int:
    """Returns the (constant) size of an array type.

    Since those are the only array types the generator currently produces, this
    can be used to determine the length of array types in the environment.

    Args:
      array_type: The type to extract the array size/length from.
    """
    assert isinstance(array_type, ast.ArrayTypeAnnotation), array_type
    dim = array_type.dim
    assert isinstance(dim, ast.Number), dim
    return ast_helpers.get_value_as_int(dim)

  def _generate_primitive_type(self) -> ast.TypeAnnotation:
    """Generates a random primitive-based type (no extra dims or tuples)."""
    primitive_token = self._generate_type_primitive()
    return ast_helpers.make_builtin_type_annotation(
        self.m, self.fake_span, primitive_token, dims=())

  def _generate_bits_type(self) -> ast.TypeAnnotation:
    """Generates a random bits type."""
    if self.options.max_width_bits_types <= 64 or self.rng.randint(1, 10) != 1:
      return self._generate_primitive_type()
    else:
      # Generate a type wider than 64-bits. With smallish probability choose a
      # *really* wide type if the max_width_bits_types supports it, otherwise
      # choose a width up to 128 bits.
      max_width = self.options.max_width_bits_types
      if max_width > 128 and self.rng.randint(1, 10) > 1:
        max_width = 128
      return self._make_type_annotation(
          self.rng.choice((True, False)),
          64 + self.rng.randint(1, max_width - 64))

  def _make_identifier_token(self, identifier: Text) -> scanner.Token:
    return scanner.Token(scanner.TokenKind.IDENTIFIER, self.fake_span,
                         identifier)

  def _make_name_ref(self, name_def: ast.NameDef) -> ast.NameRef:
    return ast.NameRef(self.m, self.fake_span, name_def.identifier, name_def)

  def _make_name_def(self, identifier: Text) -> ast.NameDef:
    return ast.NameDef(self.m, self.fake_span, identifier)

  def _generate_param(self) -> ast.Param:
    identifier = self.gensym()
    type_ = self._generate_bits_type()
    name_def = ast.NameDef(self.m, self.fake_span, identifier)
    return ast.Param(self.m, name_def, type_)

  def _generate_params(self, count: int) -> Tuple[ast.Param, ...]:
    return tuple(self._generate_param() for _ in range(count))

  def _builtin_name_ref(self, identifier: Text) -> ast.NameRef:
    return ast.NameRef(self.m, self.fake_span, identifier,
                       ast.BuiltinNameDef(self.m, identifier))

  def _make_ge(self, lhs: ast.Expr, rhs: ast.Expr) -> ast.Expr:
    return ast.Binop(self.m, self.fake_span, ast.BinopKind.GE, lhs, rhs)

  def _make_sel(self, test: ast.Expr, lhs: ast.Expr, rhs: ast.Expr) -> ast.Expr:
    return ast.Ternary(self.m, self.fake_span, test, lhs, rhs)

  def _generate_umin(self, arg: ast.Expr, arg_type: ast.TypeAnnotation,
                     other: int) -> ast.Expr:
    rhs = self._make_number(other, arg_type)
    test = self._make_ge(arg, rhs)
    return self._make_sel(test, rhs, arg)

  def _generate_binop_same_input_type(
      self, lhs: ast.Expr, rhs: ast.Expr,
      input_type: ast.TypeAnnotation) -> Tuple[ast.Binop, ast.TypeAnnotation]:
    """Generates a binary operator on lhs/rhs which have the same input type."""
    if self.rng.random() < 0.1:
      op = self.rng.choice(ast_helpers.BINOP_COMPARISON_KIND_LIST)
      output_type = self._make_type_annotation(False, 1)
    else:
      op = self.rng.choice(self._binops)
      if op in ast_helpers.BINOP_SHIFTS and self.rng.random() < 0.8:
        # Clamp the RHS to be in range most of the time.
        assert isinstance(input_type, ast.BuiltinTypeAnnotation), input_type
        bit_count = builtin_type_to_bits(input_type)
        new_upper = self.rng.randrange(bit_count)
        rhs = self._generate_umin(rhs, input_type, new_upper)
      output_type = input_type
    return ast.Binop(self.m, self.fake_span, op, lhs, rhs), output_type

  def _choose_env_value(
      self,
      env: Env,
      take: Callable[[ast.TypeAnnotation], bool] = lambda x: True
  ) -> Tuple[Callable[[], ast.Expr], ast.TypeAnnotation]:
    """Returns (make_expr, type_)."""
    choices = sorted([t for t in env.items() if take(t[1][1])])
    if not choices:
      raise EmptyEnvError
    _, (make_expr, type_, _) = self.rng.choice(choices)
    return make_expr, type_

  def _env_contains_tuple(self, env: Env) -> bool:
    return any(
        isinstance(type_, ast.TupleTypeAnnotation)
        for (_, type_, _) in env.values())

  def _not_array(self, t: ast.TypeAnnotation) -> bool:
    return not isinstance(t, ast.ArrayTypeAnnotation)

  def _not_tuple_or_array(self, t: ast.TypeAnnotation) -> bool:
    return not isinstance(t, (ast.TupleTypeAnnotation, ast.ArrayTypeAnnotation))

  def _is_tuple(self, t: ast.TypeAnnotation) -> bool:
    return isinstance(t, ast.TupleTypeAnnotation)

  def _is_builtin_unsigned(self, t: ast.TypeAnnotation) -> bool:
    return isinstance(t, ast.BuiltinTypeAnnotation) and not t.signedness

  def _is_builtin_bool(self, t: ast.TypeAnnotation) -> bool:
    return str(t) == 'bool'

  def _env_contains_array(self, env: Env) -> bool:
    return any(
        isinstance(type_, ast.ArrayTypeAnnotation)
        for (_, type_, _) in env.values())

  def _generate_logical_binop(self,
                              env: Env) -> Tuple[ast.Binop, ast.TypeAnnotation]:
    """Generates a logical binary operation (e.g. and, xor, or)."""
    make_lhs, lhs_type = self._choose_env_value(env, self._not_tuple_or_array)
    make_rhs, rhs_type = self._choose_env_value(env, self._not_tuple_or_array)
    # Convert into one-bit numbers by checking whether lhs and rhs values are 0.
    lhs = ast.Binop(self.m, self.fake_span, ast.BinopKind.NE, make_lhs(),
                    self._make_number(0, lhs_type))
    rhs = ast.Binop(self.m, self.fake_span, ast.BinopKind.NE, make_rhs(),
                    self._make_number(0, rhs_type))
    # Pick some operation to do.
    op = self.rng.choice([ast.BinopKind.LOGICAL_AND, ast.BinopKind.LOGICAL_OR])
    return ast.Binop(self.m, self.fake_span, op, lhs,
                     rhs), self._make_type_annotation(False, 1)

  def _generate_binop(self, env: Env) -> Tuple[ast.Binop, ast.TypeAnnotation]:
    """Generates a binary operation AST node."""
    if self.rng.random() < 0.1:
      return self._generate_logical_binop(env)

    make_lhs, lhs_type = self._choose_env_value(env, self._not_tuple_or_array)
    make_rhs, rhs_type = self._choose_env_value(env, self._not_tuple_or_array)
    if lhs_type == rhs_type:
      return self._generate_binop_same_input_type(make_lhs(), make_rhs(),
                                                  lhs_type)

    if self.rng.choice([True, False]):
      # Cast RHS to LHS type.
      lhs = make_lhs()
      rhs = ast.Cast(self.m, self.fake_span, lhs_type, make_rhs())
      result_type = lhs_type
    else:
      # Cast LHS to RHS type.
      lhs = ast.Cast(self.m, self.fake_span, rhs_type, make_lhs())
      rhs = make_rhs()
      result_type = rhs_type

    return self._generate_binop_same_input_type(lhs, rhs, result_type)

  def _create_type_ref(self, type_: ast.TypeAnnotation) -> ast.TypeRef:
    """Creates and returns a type ref for the given annotation.

    As part of this process, an ast.TypeDef is created and added to the
    set of currently active set.

    Args:
      type_: The type for which to create refs.

    Returns:
      The generated TypeRef.
    """
    type_name = self.gensym()
    name_def = self._make_name_def(type_name)
    type_def = ast.TypeDef(self.m, self.fake_span, name_def, type_, False)
    type_ref = ast.TypeRef(self.m, self.fake_span, type_name, type_def)
    self._type_defs.append(type_def)
    self._type_bit_counts[str(type_ref)] = self._get_type_bit_count(type_)
    return type_ref

  def _generate_map(self, level,
                    env: Env) -> Tuple[ast.Invocation, ast.TypeAnnotation]:
    """Generates an invocation of the map builtin."""

    map_fn_name = self.gensym()
    # generate_function, in turn, can call generate_map, so we need some way of
    # bounding the recursion. To limit explosion, we increase level by three
    # (chosen empirically) instead of just one.
    map_fn = self.generate_function(map_fn_name, level + 3, param_count=1)
    self._functions.append(map_fn)

    map_arg_type = map_fn.params[0].type_
    assert isinstance(map_arg_type, ast.BuiltinTypeAnnotation), map_arg_type
    map_arg_signedness, map_arg_bits = map_arg_type.signedness_and_bits

    array_size = self.rng.randrange(
        1, max(2, self.options.max_width_aggregate_types // map_arg_bits))
    return_type = self._make_array_type(map_fn.return_type, array_size)

    # Seems pretty unlikely that we'll have the exact array we need, so we'll
    # just create one.
    # TODO(b/144724970): Consider creating arrays from values in the env.
    def get_number():
      return self._generate_number(env, map_arg_bits, map_arg_signedness)

    args = self._make_constant_array(
        tuple(get_number()[0] for i in range(array_size)))
    args.type_ = ast.ArrayTypeAnnotation(self.m, self.fake_span,
                                         map_fn.params[0].type_,
                                         self._make_number(array_size, None))

    fn_ref = self._make_name_ref(self._make_name_def(map_fn_name))
    invocation = ast.Invocation(self.m, self.fake_span,
                                self._builtin_name_ref('map'), (args, fn_ref))
    return invocation, return_type

  def _make_constant_array(self, exprs: Tuple[ast.Expr, ...]) -> ast.Array:
    return ast.ConstantArray(
        self.m, members=exprs, has_ellipsis=False, span=self.fake_span)

  def _make_array(self, exprs: Tuple[ast.Expr, ...]) -> ast.Array:
    return ast.Array(self.m, self.fake_span, exprs, has_ellipsis=False)

  def _generate_one_hot_select_builtin(
      self, env: Env) -> Tuple[ast.Invocation, ast.TypeAnnotation]:
    """Generates an invocation of the one_hot_sel builtin."""

    # We need to choose a selector with a certain number of bits, then form an
    # array from that many values in the environment.
    def choose_value(t: ast.TypeAnnotation) -> bool:
      return isinstance(t, ast.BuiltinTypeAnnotation) and 0 < t.bits <= 8

    try:
      make_lhs, lhs_type = self._choose_env_value(env, choose_value)
    except EmptyEnvError:
      # If there's no natural environment value to use as the LHS, make up a
      # number and number of bits.
      bits = self.rng.randrange(1, 8)
      make_lhs = lambda: self._generate_number(env, bits, signedness=False)[0]
      lhs_type = self._make_type_annotation(False, bits)

    make_rhs, rhs_type = self._choose_env_value(env, self._not_tuple_or_array)
    cases = [make_rhs]
    assert isinstance(lhs_type, ast.BuiltinTypeAnnotation), lhs_type
    total_operands = lhs_type.bits
    for _ in range(total_operands - 1):
      make_rhs, rhs_type = self._choose_env_value(env, lambda t: t == rhs_type)
      cases.append(make_rhs)

    invocation = ast.Invocation(
        self.m,
        self.fake_span,
        self._builtin_name_ref('one_hot_sel'),
        args=(make_lhs(),
              self._make_array(tuple(make_case() for make_case in cases))))
    return invocation, rhs_type

  def _generate_unop(self, env: Env) -> Tuple[ast.Unop, ast.TypeAnnotation]:
    make_arg, arg_type = self._choose_env_value(env, self._not_tuple_or_array)
    op = self.rng.choice(ast_helpers.UNOP_SAME_TYPE_KIND_LIST)
    return ast.Unop(self.m, self.fake_span, op, make_arg()), arg_type

  def _generate_unop_builtin(
      self, env: Env) -> Tuple[ast.Invocation, ast.TypeAnnotation]:
    """Generates a call to a unary builtin."""
    make_arg, arg_type = self._choose_env_value(env, self._is_builtin_unsigned)
    choices = ['clz']
    # Since one_hot adds a bit, only use it when we have head room beneath
    # max_width_bits_types to add another bit.
    one_hot_ok = builtin_type_to_bits(
        arg_type) < self.options.max_width_bits_types
    if one_hot_ok:
      choices.append('one_hot')
    to_invoke = self.rng.choice(choices)
    if to_invoke == 'clz':
      invocation = ast.Invocation(
          self.m,
          self.fake_span,
          self._builtin_name_ref(to_invoke),
          args=(make_arg(),))
      result_type = arg_type
    else:
      assert to_invoke == 'one_hot'
      lsb_or_msb = self.rng.choice((True, False))
      invocation = ast.Invocation(
          self.m,
          self.fake_span,
          self._builtin_name_ref(to_invoke),
          args=(make_arg(), self._make_bool(lsb_or_msb)))
      result_bits = builtin_type_to_bits(arg_type) + 1
      result_type = self._make_type_annotation(False, result_bits)
    return invocation, result_type

  def _generate_bit_slice(self,
                          env: Env) -> Tuple[ast.Index, ast.TypeAnnotation]:
    """Generates a bit slice AST node."""
    make_arg, arg_type = self._choose_env_value(env, self._not_tuple_or_array)
    bit_count = builtin_type_to_bits(arg_type)
    slice_type = self.rng.choice(['bit_slice', 'width_slice', 'dynamic_slice'])
    while True:
      start_low = 0 if slice_type == 'width_slice' else -bit_count - 1
      start = (
          self.rng.randrange(start_low, bit_count + 1) if self.rng.choice(
              (True, False)) else None)
      limit = (
          self.rng.randrange(-bit_count - 1, bit_count + 1) if self.rng.choice(
              (True, False)) else None)
      _, width = bit_helpers.resolve_bit_slice_indices(bit_count, start, limit)
      if width <= 0:  # Make sure we produce non-zero-width things.
        continue
      else:
        break
    if slice_type == 'width_slice':
      index_slice = ast.WidthSlice(self.m, self.fake_span,
                                   self._make_number(start or 0, None),
                                   self._make_type_annotation(False, width))
    elif slice_type == 'bit_slice':
      index_slice = ast.Slice(
          self.m, self.fake_span,
          None if start is None else self._make_number(start, None),
          None if limit is None else self._make_number(limit, None))
    else:
      start_arg, _ = self._choose_env_value(env, self._is_builtin_unsigned)
      index_slice = ast.WidthSlice(self.m, self.fake_span, start_arg(),
                                   self._make_type_annotation(False, width))
    type_ = self._make_type_annotation(False, width)
    return (ast.Index(self.m, self.fake_span, make_arg(), index_slice), type_)

  def _generate_bitwise_reduction(
      self, env: Env) -> Tuple[ast.Invocation, ast.TypeAnnotation]:
    """Generates one of the bitwise reductions as an Invocation node."""
    make_arg, _ = self._choose_env_value(env, self._is_builtin_unsigned)
    ops = ['and_reduce', 'or_reduce', 'xor_reduce']
    callee = self._builtin_name_ref(self.rng.choice(ops))
    type_ = self._make_type_annotation(False, 1)
    return (ast.Invocation(self.m, self.fake_span, callee,
                           (make_arg(),)), type_)

  def _generate_nary_operand_count(self, env: Env) -> int:
    count = int(math.ceil(self.rng.weibullvariate(1, 0.5) * 4))
    count = min(count, len(env))
    return count

  def _generate_cast_bits_to_array(
      self, env: Env) -> Tuple[ast.Cast, ast.TypeAnnotation]:
    """Generates a cast from bits to array type."""

    # Get a random bits-typed element from the environment.
    make_arg, arg_type = self._choose_env_value(env, self._is_builtin_unsigned)

    # Next, find factors of the bit count and select one pair.
    bit_count = builtin_type_to_bits(arg_type)
    factors = []
    for i in range(1, bit_count + 1):
      if bit_count % i == 0:
        factors.append((i, bit_count // i))

    element_size, array_size = self.rng.choice(factors)
    element_type = ast_helpers.make_builtin_type_annotation(
        self.m, self.fake_span,
        scanner.Token(
            value=scanner.Keyword.UN,
            span=self.fake_span), (self._make_number(element_size, None),))

    outer_array_type = self._make_array_type(element_type, array_size)

    return (ast.Cast(self.m, self.fake_span, outer_array_type,
                     make_arg()), outer_array_type)

  def _generate_array_concat(self,
                             env: Env) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Returns a binary concatenation of two arrays in env.

    The two arrays to concatenate in env will have the same element type.

    Args:
      env: Environment of values that can be selected from for array
        concatenation.
    Precondition: There must be an array value present in env.
    """
    make_lhs, lhs_type = self._choose_env_value(
        env, lambda t: isinstance(t, ast.ArrayTypeAnnotation))
    assert isinstance(lhs_type, ast.ArrayTypeAnnotation), lhs_type

    def array_same_elem_type(t: ast.TypeAnnotation) -> bool:
      return (isinstance(t, ast.ArrayTypeAnnotation) and
              t.element_type == lhs_type.element_type)

    make_rhs, rhs_type = self._choose_env_value(env, array_same_elem_type)
    result = ast.Binop(self.m, self.fake_span, ast.BinopKind.CONCAT, make_lhs(),
                       make_rhs())
    lhs_size = self._get_array_size(lhs_type)
    bits_per_elem = self._get_type_bit_count(lhs_type) // lhs_size
    result_size = lhs_size + self._get_array_size(rhs_type)
    dim = self._make_number(result_size, None)
    result_type = ast.ArrayTypeAnnotation(self.m, self.fake_span,
                                          lhs_type.element_type, dim)
    self._type_bit_counts[str(result_type)] = bits_per_elem * result_size
    return (result, result_type)

  def _generate_concat(self, env: Env) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Returns a (potentially vacuous) concatenate operation of values in `env`.

    Args:
      env: Environment of values that can be selected from for concatenation.
    Note: the concat operation will not exceed the maximum bit width so the
      concat may end up being a nop.
    """
    if self._env_contains_array(env) and self.rng.choice([True, False]):
      return self._generate_array_concat(env)

    count = self._generate_nary_operand_count(env) + 2
    operands = []
    operand_types = []
    for i in range(count):
      make_arg, arg_type = self._choose_env_value(env,
                                                  self._is_builtin_unsigned)
      operands.append(make_arg())
      operand_types.append(arg_type)
    result = operands[0]
    result_bits = builtin_type_to_bits(operand_types[0])
    for i in range(1, count):
      this_bits = builtin_type_to_bits(operand_types[i])
      if result_bits + this_bits > self.options.max_width_bits_types:
        break
      result = ast.Binop(self.m, self.fake_span, ast.BinopKind.CONCAT, result,
                         operands[i])
      result_bits += this_bits
    assert result_bits <= self.options.max_width_bits_types, result_bits
    return (result, self._make_type_annotation(False, result_bits))

  def _make_number(self, value: int,
                   type_: Optional[ast.TypeAnnotation]) -> ast.Number:
    """Creates a number AST node with value 'value' of type 'type_'."""
    if self._is_builtin_bool(type_):
      assert 0 <= value <= 1, value
      return ast.Number(self.m, self.fake_span, 'true' if value else 'false',
                        ast.NumberKind.BOOL, type_)
    return ast.Number(self.m, self.fake_span, hex(value), ast.NumberKind.OTHER,
                      type_)

  def _make_bool(self, value: bool) -> ast.Number:
    return self._make_number(int(value), self._make_type_annotation(False, 1))

  def _generate_number(
      self,
      env: Env,  # pylint: disable=unused-argument
      bits: Optional[int] = None,
      signedness: Optional[bool] = None
  ) -> Tuple[ast.Number, ast.TypeAnnotation]:
    """Generates a number AST node with its associated type."""
    if bits:
      assert signedness is not None
      type_ = self._make_type_annotation(signedness, bits)
    else:
      type_ = self._generate_primitive_type()
    bit_count = builtin_type_to_bits(type_)
    value = self.rng.choice(make_bit_patterns(bit_count))
    return (self._make_number(value, type_), type_)

  def _generate_retval(self, env: Env) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Generates a return-value positioned expression."""
    retval_count = self._generate_nary_operand_count(env)

    sorted_env = sorted(env.items())
    env_params = [(v[0], v[1]) for _, v in sorted_env if v[2]]
    env_non_params = [(v[0], v[1]) for _, v in sorted_env if not v[2]]

    make_exprs = []
    types = []
    total_bit_count = 0
    for _ in range(retval_count):
      p = self.rng.random()
      # Pick parameter values as retvals with low probability.
      if p < 0.1 or not env_non_params:
        make_expr, type_ = self.rng.choice(env_params)
      else:
        make_expr, type_ = self.rng.choice(env_non_params)
      assert isinstance(type_, ast.TypeAnnotation)
      if total_bit_count + self._get_type_bit_count(
          type_) > self.options.max_width_aggregate_types:
        continue
      make_exprs.append(make_expr)
      types.append(type_)
      total_bit_count += self._get_type_bit_count(type_)

    assert len(types) == len(make_exprs), (len(types), len(make_exprs))
    if len(types) == 1:
      return make_exprs[0](), types[0]
    else:
      assert len(types) != 1
      es = tuple(make_expr() for make_expr in make_exprs)
      return ast.XlsTuple(self.m, self.fake_span,
                          es), self._make_tuple_type(tuple(types))

  def _make_range(self, zero: ast.Expr, arg: ast.Expr):
    return ast.Invocation(
        self.m,
        self.fake_span,
        self._builtin_name_ref('range'),
        args=(zero, arg))

  def _generate_counted_for(self,
                            env: Env) -> Tuple[ast.For, ast.TypeAnnotation]:
    """Generates a counted for loop."""
    # Right now just generates the 'identity' for loop.
    ivar_type = self._make_type_annotation(False, 4)
    zero = self._make_number(0, type_=ivar_type)
    trips = self._make_number(self.randrange(8), type_=ivar_type)
    iterable = self._make_range(zero, trips)
    x_def = self._make_name_def('x')
    name_def_tree = ast.NameDefTree(
        self.m,
        self.fake_span,
        tree=(ast.NameDefTree(self.m, self.fake_span, self._make_name_def('i')),
              ast.NameDefTree(self.m, self.fake_span, x_def)))
    make_expr, type_ = self._choose_env_value(env, self._not_array)
    init = make_expr()
    body = self._make_name_ref(x_def)
    tree_type = self._make_tuple_type((ivar_type, type_))
    return ast.For(self.m, self.fake_span, name_def_tree, tree_type, iterable,
                   body, init), type_

  def _generate_tuple_or_index(self,
                               env: Env) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Generates either a tupling operation or an index-a-tuple operation."""
    p = self.rng.random()
    do_index = p <= 0.5 and self._env_contains_tuple(env)
    if do_index:
      make_tuple_expr, tuple_type = self._choose_env_value(env, self._is_tuple)
      # TODO(leary): 2019-08-07 Also make it possible to select a value from
      # the environment to use as an index.
      assert isinstance(tuple_type, ast.TupleTypeAnnotation), tuple_type
      i = self.randrange(len(tuple_type.members))
      index_expr = self._make_number(
          i, type_=self._make_type_annotation(False, 32))
      tuple_expr = make_tuple_expr()
      assert isinstance(tuple_type, ast.TupleTypeAnnotation), tuple_type
      return ast.Index(self.m, self.fake_span, tuple_expr,
                       index_expr), tuple_type.members[i]
    else:
      members = []
      types = []
      total_bit_count = 0
      for i in range(self._generate_nary_operand_count(env)):
        make_expr, type_ = self._choose_env_value(env, self._not_array)
        if total_bit_count + self._get_type_bit_count(
            type_) > self.options.max_width_aggregate_types:
          continue
        members.append(make_expr())
        types.append(type_)
        total_bit_count += self._get_type_bit_count(type_)

      return ast.XlsTuple(self.m, self.fake_span,
                          tuple(members)), self._make_tuple_type(tuple(types))

  def _generate_expr(self, env: Env,
                     level: int) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Generates an expression AST node and returns it."""
    if self.should_nest(level):
      identifier = self.gensym()
      name_def = ast.NameDef(self.m, self.fake_span, identifier)
      choices = collections.OrderedDict()
      if self.options.emit_loops:
        choices[self._generate_counted_for] = 1.0
      choices[self._generate_tuple_or_index] = 1.0
      choices[self._generate_concat] = 1.0
      choices[self._generate_binop] = 1.0
      choices[self._generate_unop] = 1.0
      choices[self._generate_unop_builtin] = 1.0
      choices[self._generate_one_hot_select_builtin] = 1.0
      choices[self._generate_number] = 1.0
      choices[self._generate_bit_slice] = 1.0
      if not self._codegen_ops_only:
        choices[self._generate_bitwise_reduction] = 1.0
      choices[self._generate_cast_bits_to_array] = 1.0
      # If maps recurse with equal probability, then the output will grow
      # exponentially with level, so we need to scale inversely.
      choices[lambda env: self._generate_map(level, env)] = 1.0 / (10**level)

      rng_choices = getattr(self.rng, 'choices')
      while True:
        expr_generator = rng_choices(
            population=list(choices.keys()),
            weights=list(choices.values()),
            k=1)[0]
        try:
          rhs, rhs_type = expr_generator(env)
        except EmptyEnvError:
          # Try a different generator that may be more accepting of env values.
          del choices[expr_generator]
          if not choices:  # If we ran out of things to try, bail.
            return self._generate_retval(env)
          continue
        else:
          break
      new_env = collections.OrderedDict(env)
      new_env[identifier] = (
          lambda: self._make_name_ref(name_def)), rhs_type, False
      body, body_type = self._generate_expr(new_env, level + 1)
      name_def_tree = ast.NameDefTree(self.m, self.fake_span, name_def)
      let = ast.Let(
          self.m,
          self.fake_span,
          name_def_tree,
          rhs_type,
          rhs,
          body,
          const=None)
      return let, body_type
    else:  # Should not nest any more -- select return values.
      return self._generate_retval(env)

  def _generate_body(
      self, level: int,
      params: Tuple[ast.Param, ...]) -> Tuple[ast.Expr, ast.TypeAnnotation]:
    """Generates the body of a function AST node."""

    def make_lambda(name: ast.NameDef) -> Callable[[], ast.NameRef]:
      return lambda: self._make_name_ref(name)

    env = collections.OrderedDict([
        (param.name.identifier, (make_lambda(param.name), param.type_, True))
        for param in params
    ])
    return self._generate_expr(env, level)

  def generate_function(self,
                        name: Text,
                        level: int = 0,
                        param_count: int = None) -> ast.Function:
    if param_count is None:
      param_count = int(math.ceil(self.rng.weibullvariate(1, 1.15) * 4))
    params = self._generate_params(param_count)
    body, body_type = self._generate_body(level, params)
    return ast.Function(
        self.m,
        self.fake_span,
        ast.NameDef(self.m, self.fake_span, name),
        parametric_bindings=(),
        params=params,
        return_type=body_type,
        body=body,
        public=False)

  def generate_function_in_module(
      self, fname: Text, mname: Text) -> Tuple[ast.Function, ast.Module]:
    """Generates a function named "fname" in a module named "mname"."""
    self.m = ast.Module(mname)
    f = self.generate_function(fname)
    top = tuple(self._type_defs) + tuple(self._functions) + (f,)
    for item in top:
      self.m.add_top(item)
    return f, self.m
