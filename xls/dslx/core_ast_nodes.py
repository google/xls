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

"""AST nodes that form circular references / are more 'core' to the grammar.

These are broken out largely to reduce pytype runtime on a monolithic AST file.
"""

import abc
import enum as enum_mod
from typing import Text, Optional, Union, Tuple, List, cast, Any

from absl import logging
import aenum

from xls.dslx import bit_helpers
from xls.dslx.ast_node import AstNode
from xls.dslx.ast_node import AstNodeOwner
from xls.dslx.ast_node import AstVisitor
from xls.dslx.concrete_type import BitsType
from xls.dslx.concrete_type import ConcreteType
from xls.dslx.free_variables import FreeVariables
from xls.dslx.scanner import Pos
from xls.dslx.scanner import Token
from xls.dslx.scanner import TokenKind
from xls.dslx.span import Span
from xls.dslx.xls_type_error import TypeInferenceError


class BuiltinNameDef(AstNode):
  """Represents the definition point of a built-in name.

  This node is for representation consistency; all references to names must have
  a corresponding definition where the name was bound. For primitive builtins
  there is no textual point, so we create positionless (in the text) definition
  points for them.
  """

  def __init__(self, owner: AstNodeOwner, identifier: Text):
    super().__init__(owner)
    self.identifier = identifier

  def __repr__(self) -> Text:
    return 'BuiltinNameDef(identifier={!r})'.format(self.identifier)

  def __str__(self) -> Text:
    return self.identifier


class NameDef(AstNode):
  """Represents the definition of a name (identifier)."""

  def __init__(self, owner: AstNodeOwner, span: Span, identifier: str):
    super().__init__(owner)
    self.span = span
    self.identifier = identifier

  def __repr__(self) -> Text:
    return f'NameDef({self.identifier!r})'

  def __str__(self) -> str:
    return self.identifier

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return FreeVariables()


class WildcardPattern(AstNode):
  """Represents a wildcard pattern in a 'match' construct."""

  def __init__(self, owner: AstNodeOwner, span: Span):
    super().__init__(owner)
    self.span = span

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

  def __init__(self, owner: AstNodeOwner, span: Span):
    super().__init__(owner)
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

  def __init__(self, owner: AstNodeOwner, span: Span, members: Tuple[Expr, ...],
               has_ellipsis: bool):
    super().__init__(owner, span)
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

  def __init__(self, owner: AstNodeOwner, span: Span, text: Text,
               type_def: Union['ModRef', 'TypeDef', 'Enum']):
    super().__init__(owner, span)
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


class NameRef(Expr):
  """Represents a reference to a name (identifier)."""

  def __init__(self, owner: AstNodeOwner, span: Span, identifier: str,
               name_def: Union['NameDef', 'BuiltinNameDef']):
    super().__init__(owner, span)
    self.identifier = identifier
    self.name_def = name_def

  def __repr__(self) -> Text:
    return (f'NameRef(identifier={self.identifier!r},'
            f'name_def={self.name_def!r})')

  def __str__(self) -> Text:
    return self.identifier

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    if not hasattr(self.name_def, 'span') or self.name_def.span.start < pos:
      return FreeVariables({self.identifier: [self]})
    return FreeVariables()


class ConstRef(NameRef):
  """Used to represent a named reference to a Constant name definition."""


class EnumRef(Expr):
  """Represents an enum-value reference (via ::, i.e. Foo::BAR)."""

  def __init__(self, owner: AstNodeOwner, span: Span,
               enum: Union['Enum', 'TypeDef'], value_tok: Token):
    super().__init__(owner, span)
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

  def __init__(self, owner: AstNodeOwner, span: Span, name: Tuple[Text, ...],
               name_def: NameDef, alias: Optional[Text]):
    super().__init__(owner)
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

  def __init__(self, owner: AstNodeOwner, span: Span, mod: Import,
               value_tok: Token):
    super().__init__(owner, span)
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

  def __init__(self, owner: AstNodeOwner, span: Span,
               tree: Union[LeafType, Tuple['NameDefTree', ...]]):
    super().__init__(owner)
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


class NumberKind(enum_mod.Enum):
  CHARACTER = 'character'
  BOOL = 'bool'
  OTHER = 'other'


class Number(Expr):
  """Represents a number."""

  def __init__(self,
               owner: AstNodeOwner,
               span: Span,
               value: str,
               kind: NumberKind = NumberKind.OTHER,
               type_: Optional['TypeAnnotation'] = None):
    super().__init__(owner, span)
    self.kind = kind
    self.value = value
    if kind == NumberKind.BOOL:
      assert value in ('true', 'false'), value
    if value in ('true', 'false'):
      assert kind == NumberKind.BOOL, kind
    self._type = type_
    assert isinstance(value, str), value
    if kind != NumberKind.CHARACTER:
      assert not value.endswith('L'), f'Number value had "L" suffix: {value!r}'

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
    return (f'Number(kind={self.kind!r}, value={self.value!r}, '
            f'type_={self.type_!r})')

  def get_value_as_int(self) -> int:
    """Returns the numerical value contained in the AST node as a Python int."""
    if self.kind == NumberKind.CHARACTER:
      return ord(self.value)
    if self.kind == NumberKind.BOOL:
      if self.value == 'true':
        return 1
      else:
        assert self.value == 'false'
        return 0
    assert isinstance(self.value, str), self.value
    if self.value.startswith(('0x', '-0x')):
      return int(self.value.replace('_', ''), 16)
    if self.value.startswith(('0b', '-0b')):
      return int(self.value.replace('_', ''), 2)
    return int(self.value.replace('_', ''))

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

  def __init__(self, owner: AstNodeOwner, span: Span):
    super().__init__(owner)
    self.span = span

  def __str__(self) -> str:
    raise NotImplementedError

  def __eq__(self, other: Any) -> bool:
    raise NotImplementedError

  def __ne__(self, other: Any) -> bool:
    return not self.__eq__(other)

  def _accept_children(self, visitor: AstVisitor) -> None:
    raise NotImplementedError(self)


class BuiltinType(aenum.Enum):
  """Enumerates built-in types (keywords).

  Note that some of these are "volumeless"; e.g. bits and uN and sN just
  indicate the signedness and have to be supplemented by an ArrayTypeAnnotation
  that wraps it and provides a size.

  Implementation note: the `u1, u2, ..., u64, s1, s2, ..., s64` variants are
  populated by `populate_builtin_types()` below.
  """

  BITS = 'bits'
  SN = 'sN'
  UN = 'uN'
  BOOL = 'bool'

  @classmethod
  def get(cls, signed: bool, width: int):
    prefix = 's' if signed else 'u'
    return getattr(cls, f'{prefix}{width}'.upper())

  @property
  def bits(self) -> int:
    if self.value in ('bits', 'uN', 'sN'):
      return 0
    if self.value == 'bool':
      return 1
    return int(self.value[1:])

  @property
  def signedness(self) -> bool:
    return True if self.value.startswith('s') else False

  @property
  def signedness_and_bits(self) -> Tuple[bool, int]:
    return (self.signedness, self.bits)


def populate_builtin_types():
  for i in range(1, 65):
    aenum.extend_enum(BuiltinType, f'U{i}', f'u{i}')
    aenum.extend_enum(BuiltinType, f'S{i}', f's{i}')


populate_builtin_types()


class BuiltinTypeAnnotation(TypeAnnotation):
  """Represents a builtin type annotation; e.g. `u32`, `bits`, etc."""

  def __init__(self, owner: AstNodeOwner, span: Span,
               builtin_type: BuiltinType):
    assert isinstance(builtin_type, BuiltinType), repr(builtin_type)
    super().__init__(owner, span)
    self.builtin_type = builtin_type

  @property
  def signedness_and_bits(self) -> Tuple[bool, int]:
    return self.builtin_type.signedness_and_bits

  @property
  def bits(self) -> int:
    return self.builtin_type.bits

  @property
  def signedness(self) -> bool:
    return self.builtin_type.signedness

  def __hash__(self) -> int:
    return hash(id(self))

  def __eq__(self, other: Any) -> bool:
    return isinstance(
        other,
        BuiltinTypeAnnotation) and self.builtin_type == other.builtin_type

  def __repr__(self) -> str:
    return (f'BuiltinTypeAnnotation(span={self.span!r}, '
            f'builtin_type={self.builtin_type!r})')

  def __str__(self) -> str:
    return self.builtin_type.value

  def _accept_children(self, visitor: AstVisitor) -> None:
    pass


class TupleTypeAnnotation(TypeAnnotation):
  """Represents a tuple type annotation; e.g. `(u32, s42)`."""

  def __init__(self, owner: AstNodeOwner, span: Span,
               members: Tuple[TypeAnnotation, ...]):
    assert isinstance(members, tuple), members
    super().__init__(owner, span)
    self.members = members

  def __hash__(self) -> int:
    return hash(id(self))

  def __eq__(self, other: TypeAnnotation) -> bool:
    return (isinstance(other, TupleTypeAnnotation) and
            self.members == other.members)

  def __str__(self) -> str:
    guts = ', '.join(str(member) for member in self.members)
    return f'({guts})'

  def _accept_children(self, visitor: AstVisitor) -> None:
    for member in self.members:
      member.accept(visitor)

  def is_nil(self) -> bool:
    return not self.members


class TypeRefTypeAnnotation(TypeAnnotation):
  """Represents a type reference annotation."""

  def __init__(self,
               owner: AstNodeOwner,
               span: Span,
               type_ref: TypeRef,
               parametrics: Optional[Tuple[Expr, ...]] = None):
    super().__init__(owner, span)
    self.type_ref = type_ref
    self.parametrics = parametrics

  def __hash__(self) -> int:
    return hash(id(self))

  def __eq__(self, other: TypeAnnotation) -> bool:
    return (isinstance(other, TypeRefTypeAnnotation) and
            self.type_ref == other.type_ref)

  def __str__(self) -> str:
    return str(self.type_ref)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.type_ref.accept(visitor)


class ArrayTypeAnnotation(TypeAnnotation):
  """Represents an array type annotation; e.g. `u32[5]`."""

  def __init__(self, owner: AstNodeOwner, span: Span,
               element_type: TypeAnnotation, dim: Expr):
    super().__init__(owner, span)
    self.element_type = element_type
    self.dim = dim

  def __hash__(self) -> int:
    return hash(id(self))

  def __eq__(self, other: TypeAnnotation) -> bool:
    return (isinstance(other, ArrayTypeAnnotation) and
            self.element_type == other.element_type and self.dim == other.dim)

  def __str__(self) -> str:
    return f'{self.element_type}[{self.dim}]'

  def __repr__(self) -> str:
    return (f'ArrayTypeAnnotation(span={self.span!r}, '
            f'element_type={self.element_type!r}, dim={self.dim!r})')

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.element_type.accept(visitor)
    self.dim.accept(visitor)

  def _get_all_dims(self, type_: TypeAnnotation):
    if isinstance(type_, ArrayTypeAnnotation):
      return type_.get_all_dims()
    return []

  def get_all_dims(self) -> Tuple[Expr, ...]:
    return tuple([self.dim] + self._get_all_dims(self.element_type))


class TypeDef(AstNode):
  """Represents a user-defined-type definition; e.g.

    type Foo = (u32, u32);
    type Bar = (u32, Foo);
  """

  def __init__(self, owner: AstNodeOwner, public: bool, name: NameDef,
               type_: TypeAnnotation):
    super().__init__(owner)
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

  def __init__(self, owner: AstNodeOwner, span: Span, public: bool,
               name: NameDef, type_: TypeAnnotation,
               values: Tuple[Tuple[NameDef, Union[NameRef, Number]], ...]):
    super().__init__(owner)
    self.span = span
    self.public = public
    self.name = name
    self.type_ = type_
    self.values = values
    # Signedness can be populated as a note on the AST node once types have
    # been resolved (i.e. during typechecking / after parsing).
    self._signedness: Optional[bool] = None

  def __repr__(self) -> Text:
    return 'ast.Enum(span={}, name={}, type_={}, values={})'.format(
        self.span, self.name, self.type_, self.values)

  def has_value(self, name: Text) -> bool:
    """Returns whether this enum defn has a value with the name 'name'."""
    return any(k.identifier == name for k, v in self.values)

  def get_value(self, name: Text) -> Union[NameRef, Number]:
    return next(v for k, v in self.values if k.identifier == name)

  @property
  def identifier(self) -> Text:
    return self.name.identifier

  def get_signedness(self) -> bool:
    if self._signedness is None:
      raise ValueError('Signedness is not noted on this Enum: %r' % self)
    return self._signedness

  def set_signedness(self, value: bool) -> None:
    assert value is not None
    self._signedness = value


class Ternary(Expr):
  """Represents the ternary expression.

  For example, in Pythonic style:

    consequent if test else alternate
  """

  def __init__(self, owner: AstNodeOwner, span: Span, test: Expr,
               consequent: Expr, alternate: Expr):
    super().__init__(owner, span)
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

  def __init__(self, owner: AstNodeOwner, operator: Token, lhs: Expr,
               rhs: Expr):
    super().__init__(owner, operator.span)
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
