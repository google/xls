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
"""AST nodes that form circular references / are more 'core' to the grammar.

These are broken out largely to reduce pytype runtime on a monolithic AST file.
"""

import abc
from typing import Text, Optional, Union, Sequence, Tuple, List, cast

from absl import logging

from xls.dslx import bit_helpers
from xls.dslx.ast_node import AstNode
from xls.dslx.ast_node import AstVisitor
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.free_variables import FreeVariables
from xls.dslx.parametric_expression import ParametricAdd
from xls.dslx.parametric_expression import ParametricExpression
from xls.dslx.parametric_expression import ParametricSymbol
from xls.dslx.scanner import Keyword
from xls.dslx.scanner import Pos
from xls.dslx.scanner import Token
from xls.dslx.scanner import TokenKind
from xls.dslx.scanner import TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS
from xls.dslx.span import Span
from xls.dslx.xls_type_error import TypeInferenceError


class BuiltinNameDef(AstNode):
  """Represents the definition point of a built-in name.

  This node is for representation consistency; all references to names must have
  a corresponding definition where the name was bound. For primitive builtins
  there is no textual point, so we create positionless (in the text) definition
  points for them.
  """

  def __init__(self, identifier: Text):
    self.identifier = identifier

  def __repr__(self) -> Text:
    return 'BuiltinNameDef(identifier={!r})'.format(self.identifier)

  def __str__(self) -> Text:
    return self.identifier


########################
# Token-based AST nodes


class TokenAstNode(AstNode, metaclass=abc.ABCMeta):
  """AST node that simply wraps a token."""

  TOKEN_KINDS = ()  # type: Tuple[TokenKind, ...]

  def __init__(self, tok: Token):
    assert isinstance(self.TOKEN_KINDS, Sequence)
    assert isinstance(tok, Token), (self.__class__, tok)
    assert tok.kind in self.TOKEN_KINDS, (
        'Unexpected token kind for TokenAstNode subclass', self, tok)
    self.tok = tok

  def __repr__(self) -> Text:
    return '{}({!r})'.format(self.__class__.__name__, self.tok)

  @property
  def span(self) -> Span:
    return self.tok.span


class NameDef(TokenAstNode):
  """Represents the definition of a name (identifier)."""
  TOKEN_KINDS = (TokenKind.IDENTIFIER,)

  def __str__(self) -> Text:
    return self.identifier

  @property
  def identifier(self) -> Text:
    return self.tok.value

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return FreeVariables()


class WildcardPattern(TokenAstNode):
  """Represents a wildcard pattern in a 'match' construct."""

  TOKEN_KINDS = (TokenKind.IDENTIFIER,)

  def __str__(self) -> Text:
    return self.identifier

  @property
  def identifier(self) -> Text:
    return '_'

  def get_free_variables(
      self,
      pos: Pos  # pylint: disable=unused-argument
  ) -> FreeVariables:
    return FreeVariables()


##############
# Expressions


class Expr(AstNode, metaclass=abc.ABCMeta):
  """Represents an expression."""

  def __init__(self, span: Span):
    self._span = span

  @property
  def span(self) -> Span:
    return self._span

  @span.setter
  def span(self, value: Span) -> None:
    self._span = value

  @abc.abstractmethod
  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    """Returns the references to variables that are defined before 'pos'.

    This is useful for understanding what references to external data exist
    within a lexical scope.

    Args:
      start_pos: The starting position for the "free variable" query -- if a
        name reference is defined at a point previous to this, it is a free
        variable reference.

    Returns:
      The free variable references within this expression.
    """
    raise NotImplementedError(start_pos)


class Array(Expr):
  """Represents an array expression."""

  def __init__(self, members: Tuple[Expr, ...], has_ellipsis: bool, span: Span):
    super(Array, self).__init__(span)
    self.has_ellipsis = has_ellipsis
    self.members = members
    self._type = None  # type: Optional[TypeAnnotation]

  def _get_type_(self) -> Optional['TypeAnnotation']:
    return self._type

  def _set_type_(self, value: 'TypeAnnotation') -> None:
    assert isinstance(value, TypeAnnotation), value
    self._type = value

  type_ = property(_get_type_, _set_type_)

  def __str__(self) -> Text:
    if self._type:
      return '{}:[{}]'.format(self.type_,
                              ', '.join(str(m) for m in self.members))
    else:
      return '[{}]'.format(', '.join(str(m) for m in self.members))

  def _accept_children(self, visitor: AstVisitor) -> None:
    if self.type_:
      assert isinstance(self.type_, TypeAnnotation), self.type_
      self.type_.accept(visitor)
    for member in self.members:
      member.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = FreeVariables()
    for member in self.members:
      freevars = freevars.union(member.get_free_variables(start_pos))
    return freevars


class TypeRef(Expr):
  """Represents a name that refers to a defined type."""

  def __init__(self, span: Span, text: Text,
               type_def: Union['ModRef', 'TypeDef', 'Enum']):
    super(TypeRef, self).__init__(span)
    self._text = text
    self.type_def = type_def

  def __repr__(self) -> Text:
    return 'TypeRef(text={!r} type_def={!r})'.format(self.text, self.type_def)

  def __str__(self) -> Text:
    return self._text

  @property
  def text(self) -> Text:
    return self._text

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return FreeVariables()


class NameRef(TokenAstNode, Expr):
  """Represents a reference to a name (identifier)."""
  TOKEN_KINDS = (TokenKind.IDENTIFIER,)

  def __init__(self, tok: Token, name_def: Union['NameDef', 'BuiltinNameDef']):
    TokenAstNode.__init__(self, tok)
    Expr.__init__(self, tok.span)
    self.name_def = name_def

  def __repr__(self) -> Text:
    return '{}(tok={!r}, name_def={!r})'.format(self.__class__.__name__,
                                                self.tok, self.name_def)

  def __str__(self) -> Text:
    return self.identifier

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    if not hasattr(self.name_def, 'span') or self.name_def.span.start < pos:
      return FreeVariables({self.identifier: [self]})
    return FreeVariables()

  @property
  def identifier(self) -> Text:
    return self.tok.value


class ConstRef(NameRef):
  """Used to represent a named reference to a Constant name definition."""


class EnumRef(Expr):
  """Represents an enum-value reference (via ::, i.e. Foo::BAR)."""

  def __init__(self, span: Span, enum: Union['Enum', 'TypeDef'],
               value_tok: Token):
    super(EnumRef, self).__init__(span)
    self.enum = enum
    self.value_tok = value_tok

  def __str__(self) -> Text:
    return '{}::{}'.format(self.enum.identifier, self.value_tok.value)

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    return FreeVariables()


class Import(AstNode):
  """Represents an import statement; e.g.

  "import std as my_std"

  Attributes:
    span: Span of the import in the text.
    name: Name of the module being imported ("original" name before aliasing);
      e.g. "std". Only present if the import is aliased.
    name_def: The name definition we bind the import to.
    identifier: The identifier text we bind the import to.
  """

  def __init__(self, span: Span, name: Tuple[Text, ...], name_def: NameDef,
               alias: Optional[Text]):
    super(Import, self).__init__()
    self.span = span
    self.name = name
    self.name_def = name_def
    self.alias = alias

  def __repr__(self) -> Text:
    return 'Import(name={!r})'.format(self.name)

  def __str__(self) -> Text:
    if self.alias:
      return 'import {} as {}'.format('.'.join(self.name), self.alias)
    return 'import {}'.format('.'.join(self.name))

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.name_def.accept(visitor)

  @property
  def identifier(self) -> Text:
    return self.name_def.identifier


class ModRef(Expr):
  """Represents a module-value reference (via :: i.e. std::FOO)."""

  def __init__(self, span: Span, mod: Import, value_tok: Token):
    super(ModRef, self).__init__(span)
    self.mod = mod
    self.value_tok = value_tok

  def __str__(self) -> Text:
    return '{}::{}'.format(self.mod.identifier, self.value_tok.value)

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    return FreeVariables()


# Type used for leaves in a NameDefTree below.
LeafType = Union[NameDef, NameRef, EnumRef, ModRef, WildcardPattern, 'Number']


def _get_leaf_types() -> Tuple[type, ...]:  # pylint: disable=g-bare-generic
  """Returns a tuple of NameDefTree leaf types for isinstance assertions."""
  return (NameDef, WildcardPattern, Number, NameRef, EnumRef, ModRef)


class NameDefTree(AstNode):
  """Tree of name definition nodes; e.g.

  in LHS of let bindings.

  For example:

    let (a, (b, (c)), d) = ...

  Makes a:

    NameDefTree((NameDef('a'),
                 NameDefTree((
                   NameDef('b'),
                   NameDefTree((
                     NameDef('c'))))),
                 NameDef('d')))

  A "NameDef" is an AST node that signifies an identifier is being bound, so
  this is simply a tree of those (with the tree being constructed via tuples;
  leaves are NameDefs, interior nodes are tuples).

  Attributes:
    span: The span of the names at this level of the tree.
    tree: The subtree this represents (either a tuple of subtrees or a leaf).
  """

  def __init__(self, span: Span, tree: Union[LeafType, Tuple['NameDefTree',
                                                             ...]]):
    self.span = span
    self.tree = tree

  def __repr__(self) -> Text:
    return 'NameDefTree(span={}, tree={!r})'.format(self.span, self.tree)

  def __str__(self) -> Text:
    if isinstance(self.tree, (NameDef, NameRef, WildcardPattern)):
      return self.tree.identifier
    elif isinstance(self.tree, (Number, EnumRef)):
      return str(self.tree)
    else:
      return '({})'.format(', '.join(str(ndt) for ndt in self.tree))

  def is_leaf(self) -> bool:
    """Returns whether this is a leaf node."""
    return isinstance(self.tree, _get_leaf_types())

  def get_leaf(self) -> LeafType:
    """Retrieves the NameDef leaf for a terminal node in the NameDefTree."""
    assert self.is_leaf()
    return cast(LeafType, self.tree)

  def do_preorder(self, traversal, level: int = 1) -> None:
    """Performs a preorder traversal under this node in the NameDefTree.

    Args:
      traversal: Callback invoked as traversal(NameDefTree, level, branchno).
      level: Current level of the node.
    """
    if self.is_leaf():
      return

    for i, item in enumerate(self.tree):
      traversal(item, level, i)
      item.do_preorder(traversal, level + 1)

  def is_irrefutable(self) -> bool:
    return all(
        isinstance(x, (NameDef, WildcardPattern)) for x in self.flatten())

  def flatten1(self) -> List[Union[LeafType, 'NameDefTree']]:
    """Flattens tree a single level (*not* razing it all the way to leaves)."""
    if self.is_leaf():
      return [self.get_leaf()]
    return list(
        node.get_leaf() if node.is_leaf() else node for node in self.tree)

  def flatten(self) -> List[LeafType]:
    if self.is_leaf():
      return [self.get_leaf()]
    return sum((node.flatten() for node in self.tree), [])

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = FreeVariables()
    if isinstance(self.tree, tuple):
      for e in self.tree:
        freevars = freevars.union(e.get_free_variables(start_pos))
      return freevars
    return self.tree.get_free_variables(start_pos)


class Number(TokenAstNode, Expr):
  """Represents a number."""
  TOKEN_KINDS = (TokenKind.NUMBER, TokenKind.CHARACTER, TokenKind.KEYWORD)

  def __init__(self, tok: Token, type_: Optional['TypeAnnotation'] = None):
    TokenAstNode.__init__(self, tok)
    Expr.__init__(self, tok.span)
    self._type = type_
    if isinstance(self.value, str):
      assert not self.value.endswith(
          'L'), 'Number value had a Python-like "L" suffix: {!r}'.format(
              self.value)

  def _accept_children(self, visitor: AstVisitor) -> None:
    if self._type:
      self._type.accept(visitor)

  def _get_type_(self) -> Optional['TypeAnnotation']:
    return self._type

  def _set_type_(self, value: 'TypeAnnotation') -> None:
    assert isinstance(value, TypeAnnotation), value
    logging.vlog(1, 'setting type for number to: %s', value)
    self._type = value

  type_ = property(_get_type_, _set_type_)

  def __str__(self) -> Text:
    if self.type_:
      return '({}:{})'.format(self.type_, self.value)
    return str(self.value)

  def __repr__(self) -> Text:
    return 'Number(tok={!r}, type_={!r})'.format(self.tok, self.type_)

  @property
  def value(self) -> Text:
    return self.tok.value

  def get_value_as_int(self) -> int:
    """Returns the numerical value contained in the AST node as a Python int."""
    if self.tok.kind == TokenKind.CHARACTER:
      return ord(self.value)
    if self.tok.is_keyword(Keyword.TRUE):
      return 1
    if self.tok.is_keyword(Keyword.FALSE):
      return 0
    if isinstance(self.value, str):
      if self.value.startswith(('0x', '-0x')):
        return int(self.value.replace('_', ''), 16)
      if self.value.startswith(('0b', '-0b')):
        return int(self.value.replace('_', ''), 2)
      return int(self.value.replace('_', ''))
    return self.tok.value

  def get_free_variables(
      self,
      start_pos: Pos  # pylint: disable=unused-argument
  ) -> FreeVariables:
    return FreeVariables()

  def check_bitwidth(self, concrete_type: ConcreteType) -> None:
    # TODO(leary): 2019-04-19 Because we punt on typechecking symbolic concrete
    # types here we'll need to do another pass to check whether numerical values
    # fit once the parametric symbols are instantiated as integers.
    if (isinstance(concrete_type, BitsType) and
        isinstance(concrete_type.get_total_bit_count(), int) and
        not bit_helpers.fits_in_bits(self.get_value_as_int(),
                                     concrete_type.get_total_bit_count())):
      msg = 'value {!r} does not fit in the bitwidth of a {} ({})'.format(
          self.value, concrete_type, concrete_type.get_total_bit_count())
      raise TypeInferenceError(span=self.span, type_=concrete_type, suffix=msg)


class TypeAnnotation(AstNode):
  """Represents a type that can be annotated on an expression.

  This is notably *not* an expression, as types are not values.
  """

  @classmethod
  def make_array(cls, span: Span, primitive: Union[Token, TypeRef],
                 dims: Tuple[Expr, ...]) -> 'TypeAnnotation':
    assert dims is not None and dims
    return cls(span, primitive, dims)

  @classmethod
  def make_tuple(cls, span: Span, oparen: Token,
                 members: Tuple['TypeAnnotation', ...]) -> 'TypeAnnotation':
    if oparen.kind != TokenKind.OPAREN:
      raise ValueError(
          'Expect "(" token as primitive type for tuple; got {}'.format(oparen))
    return cls(span, oparen, tuple_members=members)

  def __init__(self,
               span: Span,
               primitive: Union[Token, TypeRef],
               dims: Optional[Tuple[Expr, ...]] = None,
               parametrics: Optional[Tuple[Expr, ...]] = None,
               tuple_members: Optional[Tuple['TypeAnnotation', ...]] = None):
    self.span = span
    self.primitive = primitive
    self._dims = dims
    self.parametrics = parametrics
    self._tuple_members = tuple_members
    self._check_invariants()

  def _check_invariants(self) -> None:
    assert isinstance(
        self.primitive,
        (Token,
         TypeRef)), 'Primitive should be Token or TypeRef; got {!r}'.format(
             self.primitive)
    if self.is_tuple():
      assert self._dims is None, 'Tuple should not have dims.'
      assert self._tuple_members is not None, 'Tuple should have members.'
    elif isinstance(self.primitive, TypeRef):
      assert self._tuple_members is None, ('TypeRef should not have tuple '
                                           'members.')
    else:
      assert self._dims is not None, 'Bits-based type should have dims.'
      assert self._tuple_members is None, (
          'Bits-based type should not have tuple members.')

  def _accept_children(self, visitor: AstVisitor) -> None:
    if self.is_tuple():
      for tuple_member in self._tuple_members:
        tuple_member.accept(visitor)
    elif isinstance(self.primitive, TypeRef):
      self.primitive.accept(visitor)
      for dim in self._dims or ():
        dim.accept(visitor)
    else:
      for dim in self._dims:
        dim.accept(visitor)

  def __repr__(self) -> Text:
    if isinstance(self.primitive, Token):
      primitive_str = self.primitive.to_error_str()
    else:
      primitive_str = repr(self.primitive)
    return ('TypeAnnotation(primitive={!r}, dims={!r}, tuple_members={!r}, '
            'span={})').format(primitive_str, self._dims, self._tuple_members,
                               self.span)

  def __str__(self) -> Text:
    """Returns string representation of this type.

    The value is suitable for printing DSLX-level errors to the user.
    """
    if self.is_tuple():
      assert not self._dims
      return '({}{})'.format(', '.join(str(e) for e in self.tuple_members),
                             ',' if len(self.tuple_members) == 1 else '')

    if isinstance(self.primitive, TypeRef):
      if self._dims:
        return '{}[{}]'.format(self.primitive, ','.join(map(str, self.dims)))
      else:
        return str(self.primitive)

    if not self.dims and not self.primitive.is_keyword(Keyword.BITS):
      return str(self.primitive)

    return '{}[{}]'.format(
        str(self.primitive), ','.join(str(d) for d in self.dims))

  def primitive_to_signedness_and_bits(self) -> Tuple[bool, int]:
    assert isinstance(
        self.primitive, Token
    ), 'Expected a token primitive for conversion to bit count; got {!r}'.format(
        self.primitive)
    assert self.primitive.kind == TokenKind.KEYWORD, self.primitive
    # The [su]N annotations contain their own dimensionness, so need to be
    # treated specially.
    if self.primitive.value == Keyword.UN:
      assert isinstance(self.dims[0], Number)
      return (False, self.dims[0].get_value_as_int())
    if self.primitive.value == Keyword.SN:
      assert isinstance(self.dims[0], Number)
      return (True, self.dims[0].get_value_as_int())
    return TYPE_KEYWORDS_TO_SIGNEDNESS_AND_BITS[self.primitive.value]

  def primitive_to_bits(self) -> int:
    return self.primitive_to_signedness_and_bits()[1]

  def primitive_to_signedness(self) -> bool:
    return self.primitive_to_signedness_and_bits()[0]

  def dim_to_parametric(self, expr: Expr) -> ParametricExpression:
    if isinstance(expr, NameRef):
      return ParametricSymbol(expr.name_def.identifier, expr.span)
    if isinstance(expr, Binop):
      if expr.operator.kind == TokenKind.PLUS:
        return ParametricAdd(
            self.dim_to_parametric(expr.lhs), self.dim_to_parametric(expr.rhs))
    raise TypeInferenceError(
        self.span,
        self,
        suffix='Could not concretize type with dimension: {}.'.format(expr))

  def _compatible_dims(self, dims: Tuple[Expr, ...],
                       other_dims: Tuple[Expr, ...]) -> bool:
    """Returns whether "dims" is type-compatible with "other_dims".

    For example:

      bits[32,N] is compatible with:
      bits[32,N] whether or not the AST nodes are identical.

    Args:
      dims: One sequence of dimension expressions.
      other_dims: Another sequence of dimension expressions.
    TODO(leary): 2019-01-22 The dimension expressions must be 'constexpr',
      evaluating to concrete constant values when parameters are presented for
      invocation at a given call site. Before evaluation, we can only check type
      expressions at the level that we can reason about expressions
      symbolically; e.g. "Is M+N=N?  Only if M=0 for integral types." We
      side-step the expression-equivalence problem in type checking for the
      moment, and only permit numbers to be equal and identical names to be
      equal, which limits expressiveness of type-checkable signatures but
      guarantees type checking correctness.
    """
    if len(dims) != len(other_dims):
      return False
    for d, o in zip(dims, other_dims):
      if isinstance(d, NameRef) and isinstance(o, NameRef):
        if d.name_def != o.name_def:
          return False
      elif isinstance(d, Number) and isinstance(o, Number):
        if d.value != o.value:
          return False
      else:
        raise NotImplementedError(d, o)
    return True

  def compatible(self, other: 'TypeAnnotation') -> bool:
    """Returns whether "self" and "other" are compatible types.

    Compatibility means that "self" may definitely be passed to a parameter
    accepting "other", and visa versa, without explicit coersion.

    Args:
      other: The type to test for type compatibility with.
    """
    assert other is not None
    if self.is_tuple():
      if not other.is_tuple():
        return False
      return self.get_tuple_length() == other.get_tuple_length() and all(
          e.compatible(f)
          for e, f in zip(self.tuple_members, other.tuple_members))
    if isinstance(self.primitive, Token) and isinstance(other.primitive, Token):
      return (self.primitive.kind == other.primitive.kind and
              self.primitive.value == other.primitive.value and
              self._compatible_dims(self.dims, other.dims))
    raise NotImplementedError(self, other)

  def is_array(self) -> bool:
    # Arrays can't be tuples (see _check_invariants), and it'd be confusing for
    # typeness to not be commutative, so we enforce that tuples can't be
    # arrays, either.
    if self.is_tuple():
      return False
    if not self.dims:
      return False
    if len(self.dims) > 1:
      return True
    # Both bits types and arrays have dims. If the primitive type is bits, un,
    # or sn then this is not an array as the dimensions field just gives the bit
    # width of the type.
    if isinstance(self.primitive,
                  Token) and (self.primitive.is_keyword(Keyword.BITS) or
                              self.primitive.is_keyword(Keyword.UN) or
                              self.primitive.is_keyword(Keyword.SN)):
      return False
    return True

  def is_tuple(self) -> bool:
    return isinstance(self.primitive,
                      Token) and self.primitive.kind == TokenKind.OPAREN

  def get_tuple_length(self) -> int:
    assert self.is_tuple(
    ), 'Can only retrieve length of tuple type; got {!r}'.format(self)
    return len(self.tuple_members)

  def is_nil(self) -> bool:
    """Returns whether this type is the nil (empty) tuple."""
    return self.is_tuple() and not self.tuple_members

  def is_typeref(self) -> bool:
    """Returns whether this AST type is a reference to a type definition."""
    return isinstance(self.primitive, TypeRef)

  def get_typeref(self) -> TypeRef:
    assert isinstance(self.primitive, TypeRef), self.primitive
    return self.primitive

  @property
  def typeref(self) -> TypeRef:
    """Retrieves the reference to the type definition."""
    assert isinstance(self.primitive, TypeRef)
    return self.primitive

  @property
  def tuple_members(self) -> Tuple['TypeAnnotation', ...]:
    assert self.is_tuple(
    ), 'Can only retrieve tuple members of tuple type; got: {!r}'.format(self)
    assert self._tuple_members is not None, (
        'Tuple must have non-None members: %r', self)
    return self._tuple_members

  def has_dims(self) -> bool:
    return bool(self._dims)

  @property
  def dims(self) -> Tuple[Expr, ...]:
    assert self._dims is not None, ('No dims on type AST node.', self)
    return self._dims


class TypeDef(AstNode):
  """Represents a user-defined-type definition; e.g.

    type Foo = (u32, u32);
    type Bar = (u32, Foo);
  """

  def __init__(self, public: bool, name: NameDef, type_: TypeAnnotation):
    self.public = public
    self.name = name
    self.type_ = type_

  @property
  def identifier(self) -> Text:
    return self.name.identifier

  def __repr__(self) -> Text:
    return 'TypeDef(name={!r} type_={!r})'.format(self.name, self.type_)

  def format(self) -> Text:
    """Returns a textual description of this AST node."""
    return 'type {} = {};'.format(self.name, self.type_)


class Enum(AstNode):
  """Represents a user-defined enum definition; e.g.

  enum Foo : u2 {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
  }
  """

  def __init__(self, span: Span, public: bool, name: NameDef,
               type_: TypeAnnotation,
               values: Tuple[Tuple[NameDef, Union[NameRef, Number]], ...]):
    self.span = span
    self.public = public
    self.name = name
    self.type_ = type_
    self.values = values

  def __repr__(self) -> Text:
    return 'Enum(span={}, name={}, type_={}, values={})'.format(
        self.span, self.name, self.type_, self.values)

  def has_value(self, name: Text) -> bool:
    """Returns whether this enum defn has a value with the name 'name'."""
    return any(k.identifier == name for k, v in self.values)

  def get_value(self, name: Text) -> Union[NameRef, Number]:
    return next(v for k, v in self.values if k.identifier == name)

  def get_signedness(self) -> bool:
    """Returns the signedness of the Enum's underlying type."""
    return self.type_.primitive_to_signedness()

  @property
  def identifier(self) -> Text:
    return self.name.identifier


class Ternary(Expr):
  """Represents the ternary expression.

  For example, in Pythonic style:

    consequent if test else alternate
  """

  def __init__(self, span: Span, test: Expr, consequent: Expr, alternate: Expr):
    super(Ternary, self).__init__(span)
    self.test = test
    self.consequent = consequent
    self.alternate = alternate

  def __str__(self) -> Text:
    return '({}) if ({}) else ({})'.format(self.consequent, self.test,
                                           self.alternate)

  def _accept_children(self, visitor) -> None:
    self.test.accept(visitor)
    self.consequent.accept(visitor)
    self.alternate.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.test.get_free_variables(start_pos).union(
        self.consequent.get_free_variables(start_pos).union(
            self.alternate.get_free_variables(start_pos)))


class Binop(Expr):
  """Represents a binary operation expression."""

  SHLL = TokenKind.DOUBLE_OANGLE  # <<
  SHRL = TokenKind.DOUBLE_CANGLE  # >>
  SHRA = TokenKind.TRIPLE_CANGLE  # >>>
  GE = TokenKind.CANGLE_EQUALS  # >=
  GT = TokenKind.CANGLE  # >
  LE = TokenKind.OANGLE_EQUALS  # <=
  LT = TokenKind.OANGLE  # <
  EQ = TokenKind.DOUBLE_EQUALS  # ==
  NE = TokenKind.BANG_EQUALS  # !=
  ADD = TokenKind.PLUS  # +
  SUB = TokenKind.MINUS  # -
  MUL = TokenKind.STAR  # *
  AND = TokenKind.AMPERSAND  # &
  OR = TokenKind.BAR  # |
  XOR = TokenKind.HAT  # ^
  DIV = TokenKind.SLASH  # /

  LOGICAL_AND = TokenKind.DOUBLE_AMPERSAND  # and
  LOGICAL_OR = TokenKind.DOUBLE_BAR  # or
  CONCAT = TokenKind.DOUBLE_PLUS  # ++

  # (T, T) -> T operators.
  SAME_TYPE_KIND_LIST = [
      AND,
      OR,
      SHLL,
      SHRL,
      SHRA,
      XOR,
      SUB,
      ADD,
      DIV,
      MUL,
  ]
  # (bits[M], bits[N]) -> bits[M+N] operators.
  CONCAT_TYPE_KIND_LIST = [
      CONCAT,
  ]
  # (bool, bool) -> bool operators.
  LOGICAL_KIND_LIST = [
      LOGICAL_AND,
      LOGICAL_OR,
      XOR,
  ]
  # (T, T) -> bool operators.
  COMPARISON_KIND_LIST = [
      EQ,
      NE,
      GT,
      GE,
      LT,
      LE,
  ]

  # Binary operators that are ok for enum values.
  ENUM_OK_KINDS = [
      EQ,
      NE,
  ]

  SHIFTS = set([SHLL, SHRL, SHRA])
  SAME_TYPE_KINDS = set(SAME_TYPE_KIND_LIST)
  COMPARISON_KINDS = set(COMPARISON_KIND_LIST)
  LOGICAL_KINDS = set(LOGICAL_KIND_LIST)
  CONCAT_TYPE_KINDS = set(CONCAT_TYPE_KIND_LIST)
  ALL_KINDS = SAME_TYPE_KINDS | COMPARISON_KINDS | LOGICAL_KINDS | CONCAT_TYPE_KINDS

  def __init__(self, operator: Token, lhs: Expr, rhs: Expr):
    super(Binop, self).__init__(operator.span)
    self.operator = operator
    assert operator.is_kind_or_keyword(self.ALL_KINDS), \
        'Unknown operator for binop AST: {}'.format(operator.kind)
    self.lhs = lhs
    self.rhs = rhs

  def __str__(self) -> Text:
    return '({}) {} ({})'.format(self.lhs, self.operator.kind.value, self.rhs)

  def __repr__(self) -> Text:
    return 'Binop(operator={!r}, lhs={!r}, rhs={!r})'.format(
        self.operator, self.lhs, self.rhs)

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    return self.lhs.get_free_variables(pos).union(
        self.rhs.get_free_variables(pos))

  def _accept_children(self, visitor) -> None:
    self.lhs.accept(visitor)
    self.rhs.accept(visitor)
