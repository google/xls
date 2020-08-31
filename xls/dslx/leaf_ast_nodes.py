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

"""AST nodes that layer on top of the 'core' AST nodes in core_ast_nodes.

These generally do not circularly reference each other.

These are broken out largely to reduce pytype runtime on a monolithic AST file.
"""

import enum as enum_mod
from typing import Union, Text, List, Dict, Tuple, Optional, Set, Sequence, Any
from absl import logging

from xls.dslx import free_variables
from xls.dslx.ast_node import AstNode
from xls.dslx.ast_node import AstNodeOwner
from xls.dslx.ast_node import AstVisitor
from xls.dslx.core_ast_nodes import Array
from xls.dslx.core_ast_nodes import ConstRef
from xls.dslx.core_ast_nodes import Enum
from xls.dslx.core_ast_nodes import EnumRef
from xls.dslx.core_ast_nodes import Expr
from xls.dslx.core_ast_nodes import Import
from xls.dslx.core_ast_nodes import ModRef
from xls.dslx.core_ast_nodes import NameDef
from xls.dslx.core_ast_nodes import NameDefTree
from xls.dslx.core_ast_nodes import Number
from xls.dslx.core_ast_nodes import TypeAnnotation
from xls.dslx.core_ast_nodes import TypeDef
from xls.dslx.core_ast_nodes import WildcardPattern
from xls.dslx.free_variables import FreeVariables
from xls.dslx.scanner import Pos
from xls.dslx.span import Span


NodeToType = Any


class Param(AstNode):
  """Represents a function parameter."""

  def __init__(self, owner: AstNodeOwner, name: NameDef, type_: TypeAnnotation):
    super().__init__(owner)
    self.name = name
    self.type_ = type_

  def __str__(self) -> Text:
    return '{}: {}'.format(self.name, self.type_)

  def _accept_children(self, visitor) -> None:
    self.name.accept(visitor)
    self.type_.accept(visitor)

  @property
  def span(self) -> Span:
    return Span(self.name.span.start, self.type_.span.limit)


Params = Tuple[Param, ...]


class ParametricBinding(AstNode):
  """Represents a member in a parametric binding list.

  That is, in:

    fn [X: u32, Y: u32 = X+X] f(x: bits[X]) -> bits[Y] {
      x ++ x
    }

  There are two parametric bindings:

  * X is a u32.
  * Y is a value derived from the parametric binding of X.
  """

  def __init__(self, owner: AstNodeOwner, name: NameDef, type_: TypeAnnotation,
               expr: Optional[Expr]):
    super().__init__(owner)
    self.name = name
    self.type_ = type_
    self.expr = expr

  def __repr__(self) -> Text:
    return 'ParametricBinding(name={!r}, type_={!r}, expr={!r})'.format(
        self.name, self.type_, self.expr)

  @property
  def span(self) -> Span:
    return self.name.span


class Function(AstNode):
  """Represents a function definition."""

  def __init__(self, owner: AstNodeOwner, span: Span, name: NameDef,
               parametric_bindings: Tuple[ParametricBinding,
                                          ...], params: Tuple[Param, ...],
               return_type: Optional[TypeAnnotation], body: Expr, public: bool):
    super().__init__(owner)
    self.span = span
    self.name = name
    self.parametric_bindings = parametric_bindings
    self.params = params
    self.return_type = return_type
    self.body = body
    self.public = public

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.name.accept(visitor)
    for p in self.params:
      p.accept(visitor)
    if self.return_type:
      self.return_type.accept(visitor)
    self.body.accept(visitor)

  def is_parametric(self) -> bool:
    return bool(self.parametric_bindings)

  def get_parametric_binding(self, name: Text) -> ParametricBinding:
    return next(
        p for p in self.parametric_bindings if p.name.identifier == name)

  def get_parametric_keys(self) -> Set[Text]:
    return set([b.name.identifier for b in self.parametric_bindings])

  def get_free_parametric_keys(self) -> Set[Text]:
    """Returns 'freevar' parametric varnames (not populated by expression)."""
    return set(
        [b.name.identifier for b in self.parametric_bindings if b.expr is None])

  @property
  def identifier(self) -> Text:
    return self.name.identifier

  def format(self, include_body: bool = True) -> Text:
    """Returns a textual version of this function AST node."""
    parametric_bindings = ''
    if self.parametric_bindings:
      parametric_bindings = '[' + ', '.join(
          '{}: {}'.format(param.name.identifier, param.type_)
          for param in self.parametric_bindings) + '] '
    args = ', '.join('{}: {}'.format(param.name.identifier, param.type_)
                     for param in self.params)
    body = '\n  ' + '\n'.join(
        '  ' + line for line in str(self.body).splitlines()) + '\n'
    return_str = f' -> {self.return_type} ' if self.return_type else ''
    return '{}fn {}{}({}){}{{{}}}'.format('pub ' if self.public else '',
                                          parametric_bindings,
                                          self.name.identifier, args,
                                          return_str,
                                          body if include_body else ' ... ')

  def __str__(self) -> Text:
    return self.format(include_body=False)

  def __repr__(self) -> Text:
    return ('Function(name={0.name!r}, params={0.params!r}, '
            'return_type={0.return_type!r}, body={0.body!r}, '
            'public={0.public!r})').format(self)


class QuickCheck(AstNode):
  """Represents a function to be QuickChecked."""

  def __init__(self, owner: AstNodeOwner, span: Span, f: Function,
               test_count: Optional[int]):
    super().__init__(owner)
    self.span = span
    self.f = f

    if test_count is None:
      test_count = 1000

    self.test_count = test_count

  def __str__(self) -> Text:
    return f'QC: {self.f}'


class Proc(AstNode):
  """Represents a parsed 'process' specification in the DSL."""

  def __init__(self, owner: AstNodeOwner, span: Span, name_def: NameDef,
               proc_params: Params, iter_params: Params, iter_body: Expr,
               public: bool):
    super().__init__(owner)
    self.span = span
    self.name_def = name_def
    self.proc_params = proc_params
    self.iter_params = iter_params
    self.iter_body = iter_body
    self.public = public


class ConstantArray(Array):
  """Subtype of array that holds only constant values."""

  def __init__(self, owner: AstNodeOwner, span: Span, members: Tuple[Expr, ...],
               has_ellipsis: bool):
    super().__init__(owner, span, members, has_ellipsis)
    for member in members:
      assert Constant.is_constant(member), member


class Test(AstNode):
  """Represents a test definition."""

  def __init__(self, owner: AstNodeOwner, name: NameDef, body: Expr):
    super().__init__(owner)
    self.name = name
    self.body = body

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.body.accept(visitor)

  def __str__(self) -> Text:
    return 'test {} {{ ... }}'.format(self.name)


class TestFunction(Test):
  """Represents a new-style unit test construct.

  These are specified as follows:
    #![test]
    fn test_foo() { ... }

  We keep Test for backwards compatibility with old-style test constructs.
  """

  def __init__(self, owner: AstNodeOwner, fn: Function):
    super().__init__(owner, fn.name, fn.body)
    self.fn = fn


class Constant(AstNode):
  """Represents a constant definition."""

  def __init__(self, owner: AstNodeOwner, name: NameDef, value: Expr):
    super().__init__(owner)
    assert self.is_constant(value), (
        'Expr is not considered constant: {!r}'.format(value))
    self.name = name
    self.value = value

  @classmethod
  def is_constant(cls, expr: Expr) -> bool:
    """Returns true iff 'expr' is ok to hold in an ast.Constant value."""
    if isinstance(expr, ConstantArray):
      return True
    if isinstance(expr, EnumRef):
      return True
    if isinstance(expr, Number):
      return True
    if isinstance(expr, ConstRef):
      return True
    if isinstance(expr, XlsTuple):
      return all(cls.is_constant(m) for m in expr.members)
    if isinstance(expr, Cast):
      return cls.is_constant(expr.expr)
    logging.vlog(5, 'Not constant: %r', expr)
    return False

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.value.accept(visitor)

  def __repr__(self) -> Text:
    return 'Constant(name={!r}, value={!r})'.format(self.name, self.value)


class Struct(AstNode):
  """Represents a struct definition."""

  def __init__(self, owner: AstNodeOwner, public: bool,
               parametric_bindings: Tuple[ParametricBinding, ...],
               name: NameDef, members: Sequence[Tuple[NameDef,
                                                      TypeAnnotation]]):
    super().__init__(owner)
    self.name = name
    self.members = members
    self.public = public
    self.parametric_bindings = parametric_bindings

  def is_parametric(self) -> bool:
    return bool(self.parametric_bindings)

  def get_parametric_binding(self, name: Text) -> ParametricBinding:
    return next(
        p for p in self.parametric_bindings if p.name.identifier == name)

  def get_parametric_keys(self) -> Set[Text]:
    return set([b.name.identifier for b in self.parametric_bindings])

  def get_free_parametric_keys(self) -> Set[Text]:
    """Returns 'freevar' parametric varnames (not populated by expression)."""
    return set(
        [b.name.identifier for b in self.parametric_bindings if b.expr is None])

  def __repr__(self) -> Text:
    return 'Struct({!r}, {!r}, {!r})'.format(self.name,
                                             self.parametric_bindings,
                                             self.members)

  @property
  def identifier(self) -> Text:
    return self.name.identifier

  @property
  def member_names(self) -> Tuple[Text, ...]:
    return tuple(name.identifier for name, type in self.members)


StructInstanceMember = Tuple[Text, Expr]
StructInstanceMembers = Tuple[StructInstanceMember, ...]


def _struct_to_text(struct: Union[Struct, ModRef]) -> str:
  """Returns "error display name" of a struct from a struct instance."""
  if isinstance(struct, Struct):
    return struct.identifier
  else:
    assert isinstance(struct, ModRef)
    return str(struct)


class StructInstance(Expr):
  """Represents instantiation of a struct via member expressions."""

  def __init__(self, owner: AstNodeOwner, span: Span,
               struct: Union[ModRef, Struct], members: StructInstanceMembers):
    super().__init__(owner, span)
    self.struct = struct
    self._members = tuple(members)

  @property
  def unordered_members(self) -> StructInstanceMembers:
    """Returns instance members in their syntactic order."""
    return self._members

  @property
  def struct_text(self) -> str:
    return _struct_to_text(self.struct)

  def get_ordered_members(self, struct: Struct) -> StructInstanceMembers:
    """Returns instance members ordered according to the struct definition."""
    struct_names = struct.member_names
    return tuple(sorted(self._members, key=lambda m: struct_names.index(m[0])))

  def _accept_children(self, visitor: AstVisitor) -> None:
    for member in self._members:
      member[1].accept(visitor)

  def __repr__(self) -> Text:
    members_str = ', '.join('{}: {}'.format(k, v) for k, v in self._members)
    return '{} {{ {} }}'.format(self.struct_text, members_str)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    accum = FreeVariables()
    for _, member in self._members:
      accum = accum.union(member.get_free_variables(start_pos))
    return accum


class SplatStructInstance(Expr):
  """Rerepresents struct instantiation as delta from a 'splatted' original.

  Attributes:
    struct: The struct being instantiated.
    members: Sequence of members being changed from the splatted original; e.g.
      in `Point { y: new_y, ..orig_p }` this is `[('y', new_y)]`.
    splatted: Expression that's used as the original struct instance (that we're
      instantiating a delta from); e.g. `orig_p` in the example above.
  """

  def __init__(self, owner: AstNodeOwner, span: Span, struct: Union[ModRef,
                                                                    Struct],
               members: StructInstanceMembers, splatted: Expr):
    super().__init__(owner, span)
    self.struct = struct
    self.members = tuple(members)
    self.splatted = splatted

  def _accept_children(self, visitor: AstVisitor) -> None:
    for _, member in self.members:
      member.accept(visitor)
    self.splatted.accept(visitor)

  @property
  def struct_text(self) -> str:
    return _struct_to_text(self.struct)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    accum = FreeVariables()
    for _, member in self.members:
      accum = accum.union(member.get_free_variables(start_pos))
    accum = accum.union(self.splatted.get_free_variables(start_pos))
    return accum


ModuleMember = Union[Function, Test, QuickCheck, TypeDef, Struct, Constant,
                     Enum, Import]


class Module(AstNode):
  """Represents a syntactic module in the AST.

  Modules contain top-level definitions such as functions and tests.

  Attributes:
    name: Name of this module.
    top: Top-level module constructs; e.g. functions, tests. Given as a sequence
      instead of a mapping in case there are unnamed constructs at the module
      level (e.g. metadata, docstrings).
  """

  def __init__(self, name: Text, top: Tuple[ModuleMember, ...] = ()):
    super().__init__(None)
    self.name = name
    self._top = list(top)

  @property
  def top(self) -> Tuple[ModuleMember, ...]:
    return tuple(self._top)

  def add_top(self, member: ModuleMember) -> None:
    self._top.append(member)

  def _accept_children(self, visitor: AstVisitor) -> None:
    for element in self._top:
      element.accept(visitor)

  def get_functions(self) -> List['Function']:
    return [member for member in self._top if isinstance(member, Function)]

  def get_tests(self) -> List['Test']:
    return [member for member in self._top if isinstance(member, Test)]

  def get_structs(self) -> List['Struct']:
    return [member for member in self._top if isinstance(member, Struct)]

  def get_quickchecks(self) -> List['QuickCheck']:
    return [member for member in self._top if isinstance(member, QuickCheck)]

  def get_constants(self) -> List[Constant]:
    return [member for member in self._top if isinstance(member, Constant)]

  def get_function_by_name(self) -> Dict[Text, Function]:
    return {member.name.identifier: member for member in self.get_functions()}

  def get_constant_by_name(self) -> Dict[Text, Constant]:
    return {
        member.name.identifier: member
        for member in self._top
        if isinstance(member, Constant)
    }

  def get_enum(self, name: Text) -> Enum:
    return next(
        member for member in self._top
        if isinstance(member, Enum) and member.name.identifier == name)

  def get_typedefs(self) -> List[Union[TypeDef, Struct, Enum]]:
    return [
        member for member in self._top
        if isinstance(member, (TypeDef, Struct, Enum))
    ]

  def get_typedef(self, target: Text) -> Union[TypeDef, Struct, Enum]:
    return self.get_typedef_by_name()[target]

  def get_typedef_by_name(self) -> Dict[Text, Union[TypeDef, Struct, Enum]]:
    return {x.name.identifier: x for x in self.get_typedefs()}

  def get_test_names(self) -> List[Text]:
    """Returns names of all the 'test' constructs in the module."""
    return [
        member.name.identifier
        for member in self._top
        if isinstance(member, Test)
    ]

  def get_test(self, target_name: Text) -> Test:
    """Returns a particular test construct by name.

    Args:
      target_name: Name of the test construct to retrieve from the module.

    Raises:
      KeyError: if a test with the given target_name is not found.
    """
    for member in self._top:
      if isinstance(member, Test) and member.name.identifier == target_name:
        return member
    raise KeyError('No test in module with name: ' + target_name)

  def get_function(self, target_name: Text) -> 'Function':
    for member in self._top:
      if isinstance(member, Function) and member.name.identifier == target_name:
        return member
    raise KeyError('No function in module with name: ' + target_name)

  def format(self) -> Text:
    # Add more ModuleMembers as needed.
    typedefs = [
        t.format() for t in self.get_typedefs() if isinstance(t, TypeDef)
    ]
    functions = [f.format() for f in self.get_functions()]
    return '{}{}'.format('\n'.join(typedefs), '\n'.join(functions))


class Invocation(Expr):
  """Represents an invocation expression; e.g. ``f(a, b, c)``."""

  def __init__(self, owner: AstNodeOwner, span: Span, callee: Expr,
               args: Tuple[Expr, ...]):
    super().__init__(owner, span)
    self.callee = callee
    self.args = args

  def __str__(self) -> Text:
    return '{}({})'.format(self.callee, self.format_args())

  def format_args(self) -> Text:
    return ', '.join(str(arg) for arg in self.args)

  def __repr__(self) -> Text:
    return 'Invocation(callee={!r}, args={!r})'.format(self.callee, self.args)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.callee.accept(visitor)
    for arg in self.args:
      arg.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = self.callee.get_free_variables(start_pos)
    for arg in self.args:
      freevars = freevars.union(arg.get_free_variables(start_pos))
    return freevars


class Slice(AstNode):
  """Represents a slice in the AST.

  For example, we can have  x[-4:-2], where x is of bit width N.

  Attributes:
    span: The span of the slice expression.
    start: The annotated start of the slice (-4 above).
    limit: The annotated limit of the slice (-2 above).
    computed_start: The computed start index of the slice (N - 4 above).
    computed_width: The computed width of the slice ((N - 2) - (N - 4) = 2
      above).
  """

  def __init__(self, owner: AstNodeOwner, span: Span, start: Optional[Number],
               limit: Optional[Number]):
    super().__init__(owner)
    self.span = span
    self.start = start
    self.limit = limit

    # These attributes are populated by type inference.
    self.bindings_to_start_width = dict()

  def __str__(self) -> Text:
    if self.start and self.limit:
      return '{}:{}'.format(self.start, self.limit)
    if self.start:
      return '{}:'.format(self.start)
    if self.limit:
      return ':{}'.format(self.limit)
    return ':'

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    fv = (
        self.start.get_free_variables(start_pos)
        if self.start else FreeVariables())
    return fv.union(
        self.limit.get_free_variables(start_pos) if self
        .limit else FreeVariables())


class WidthSlice(AstNode):
  """Represents a slice in the AST; e.g. `-4+:u2`."""

  def __init__(self, owner: AstNodeOwner, span: Span, start: Expr,
               width: TypeAnnotation):
    super().__init__(owner)
    self.span = span
    self.start = start
    self.width = width

  def __str__(self) -> Text:
    return '{}+:{}'.format(self.start, self.width)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.start.get_free_variables(start_pos)


class Index(Expr):
  """Represents an index expression; e.g.

  ``a[i]``.

  Attributes:
    lhs: The expression that yields the value being indexed into; e.g. 'a' in
      'a[10]'.
    index: The index expression; e.g. '10' in 'a[10]'.
    span: The span of the indexing expression.
  """

  def __init__(self, owner: AstNodeOwner, span: Span, lhs: Expr,
               index: Union[Expr, Slice, WidthSlice]):
    super().__init__(owner, span)
    self.lhs = lhs
    self.index = index

  def __str__(self) -> Text:
    return '({})[{}]'.format(self.lhs, self.index)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.lhs.accept(visitor)
    self.index.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.lhs.get_free_variables(start_pos).union(
        self.index.get_free_variables(start_pos))


class Attr(Expr):
  """Represents an attribute access expression; e.g.

  ``a.x``
  """

  def __init__(self, owner: AstNodeOwner, span: Span, lhs: Expr, attr: NameDef):
    super().__init__(owner, span)
    self.lhs = lhs
    self.attr = attr

  def __str__(self) -> Text:
    return '{}.{}'.format(self.lhs, self.attr)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.lhs.accept(visitor)
    self.attr.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.lhs.get_free_variables(start_pos)


class XlsTuple(Expr):
  """Represents an XLS tuple expression."""

  def __init__(self, owner: AstNodeOwner, span: Span, members: Tuple[Expr]):
    super().__init__(owner, span)
    self.members = members

  def _accept_children(self, visitor: AstVisitor) -> None:
    for member in self.members:
      member.accept(visitor)

  def __str__(self) -> Text:
    if len(self.members) == 1:
      return '({},)'.format(self.members[0])
    return '({})'.format(', '.join(str(m) for m in self.members))

  def __len__(self) -> int:
    return len(self.members)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = FreeVariables()
    for member in self.members:
      freevars = freevars.union(member.get_free_variables(start_pos))
    return freevars


class Let(Expr):
  """Represents a let-binding expression.

  Attributes:
    name_def_tree: The names that are bound by this let expression; e.g. in `let
      (a, b, (c)) = (1, 2, (3,)) in ...` the name_def_tree is `(a, b, (c))`.
    type_: The optional annotated type on the let expression.
    rhs: The right hand side of the let; e.g. in `let a = b in c` this is `b`.
    body: The body of the let that has the expression to be evaluated with the
      let bindings; e.g. in `let a = b in c` this is `c`.
    span: Span of this let expression in the source.
    const: Whether or not this is a constant binding; constant bindings cannot
      be shadowed.
  """

  def __init__(self, owner: AstNodeOwner, name_def_tree: NameDefTree,
               type_: Optional[TypeAnnotation], rhs: Expr, body: Expr,
               span: Span, const: Optional[Constant]):
    super().__init__(owner, span)
    self.name_def_tree = name_def_tree
    self.type_ = type_
    self.rhs = rhs
    self.body = body
    self.const = const

  def __repr__(self) -> Text:
    return ('Let(name_def_tree={!r}, type_={!r}, rhs={!r}, body={!r}, '
            'const={!r})').format(self.name_def_tree, self.type_, self.rhs,
                                  self.body, self.const)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.name_def_tree.accept(visitor)
    if self.type_ is not None:
      self.type_.accept(visitor)
    self.rhs.accept(visitor)
    self.body.accept(visitor)
    if self.const:
      self.const.accept(visitor)

  def format(self) -> Text:
    return '{} {}{} = {};\n  {}'.format(
        'const' if self.const else 'let', self.name_def_tree,
        ': ' + str(self.type_) if self.type_ else '', self.rhs, self.body)

  def __str__(self) -> Text:
    return self.format()

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    return self.rhs.get_free_variables(pos).union(
        self.body.get_free_variables(pos))


class For(Expr):
  """Represents a for-loop expression.

  Attributes:
    names: The NameDefTree that is bound for the body of the for loop.
    type_: Type annotation corresponding to "names".
    iterable: Expression for the "thing to iterate over".
    body: Expression for the loop body, should evaluate to a value of type
      type_.
    init: Initial expression for the loop (start value(s) expression).
  """

  def __init__(self, owner: AstNodeOwner, span: Span, names: NameDefTree,
               type_: TypeAnnotation, iterable: Expr, body: Expr, init: Expr):
    super().__init__(owner, span)
    self.names = names
    self.type_ = type_
    self.iterable = iterable
    self.body = body
    self.init = init

  def __str__(self) -> Text:
    return """for {names}: {types} in {iterable} {{
  {body}
}}({init})
""".format(
    names=self.names,
    types=self.type_,
    iterable=self.iterable,
    body=self.body,
    init=self.init)

  def __repr__(self) -> Text:
    return ('For(span={!r}, names={!r}, type_={!r}, iterable={!r}, body={!r}, '
            'init={!r})').format(self.span, self.names, self.type_,
                                 self.iterable, self.body, self.init)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.init.accept(visitor)
    self.body.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    free_vars = [
        getattr(self, attr).get_free_variables(start_pos)
        for attr in 'names iterable body init'.split()
    ]
    return free_variables.union_all(free_vars)


class While(Expr):
  """Represents a while loop.

  Attributes:
    test: Expression to evaluate in order to determine whether the body should
      execute.
    body: Body to execute each time the test is true.
    init: Initial value to use for loop carry data.
  """

  def __init__(self,
               owner: AstNodeOwner,
               span: Span,
               test: Optional[Expr] = None,
               body: Optional[Expr] = None,
               init: Optional[Expr] = None):
    super().__init__(owner, span)
    self.test = test
    self.body = body
    self.init = init

  def _accept_children(self, visitor: AstVisitor) -> None:
    if self.test:
      self.test.accept(visitor)
    if self.body:
      self.body.accept(visitor)
    if self.init:
      self.init.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.body.get_free_variables(start_pos).union(
        self.init.get_free_variables(start_pos)).union(
            self.test.get_free_variables(start_pos))


class Carry(Expr):
  """Represents 'carry' keyword, refers to implicit loop-carry data in While."""

  def __init__(self, owner: AstNodeOwner, span: Span, loop: While):
    super().__init__(owner, span)
    self.loop = loop

  def __str__(self) -> Text:
    return 'carry'

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return FreeVariables()


class Next(Expr):
  """Represents 'next' keyword, refers to implicit loop-carry call in Proc."""

  def __str__(self) -> Text:
    return 'next'

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return FreeVariables()


class Cast(Expr):
  """Represents a cast expression; converting a new value to a target type.

  For example:

    foo() as u32

  Casts the result of the foo() invocation to a u32 value.
  """

  def __init__(self, owner: AstNodeOwner, type_: TypeAnnotation, expr: Expr):
    start_pos = min(type_.span.start, expr.span.start)
    limit_pos = max(type_.span.limit, expr.span.limit)
    super().__init__(owner, Span(start_pos, limit_pos))
    self.type_ = type_
    self.expr = expr

  def __str__(self) -> Text:
    return '({} as {})'.format(self.expr, self.type_)

  def __repr__(self) -> Text:
    return 'Cast(type_={!r}, expr={!r})'.format(self.type_, self.expr)

  def _accept_children(self, visitor) -> None:
    self.type_.accept(visitor)
    self.expr.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    return self.expr.get_free_variables(start_pos)


class UnopKind(enum_mod.Enum):
  INV = '!'
  NEG = '-'


# (T) -> T operators.
UnopKind.SAME_TYPE_KIND_LIST = [
    UnopKind.INV,
    UnopKind.NEG,
]

UnopKind.OPERATORS = frozenset(UnopKind.SAME_TYPE_KIND_LIST)


class Unop(Expr):
  """Represents a unary operation expression; e.g. "!"."""

  def __init__(self, owner: AstNodeOwner, span: Span, kind: UnopKind,
               operand: Expr):
    super().__init__(owner, span)
    self.kind = kind
    self.operand = operand

  def __str__(self) -> Text:
    return '{}({})'.format(self.kind.value, self.operand)

  def get_free_variables(self, pos: Pos) -> FreeVariables:
    return self.operand.get_free_variables(pos)

  def _accept_children(self, visitor) -> None:
    self.operand.accept(visitor)


class MatchArm(AstNode):
  """Represents a single arm in a match expression.

  Attributes:
    patterns: The pattern to match against to yield the value of 'expr'.
    expr: The expression to yield on a match.
    span: The span of the match arm (both matcher and expr).
  """

  def __init__(self, owner: AstNodeOwner, patterns: Tuple[NameDefTree, ...],
               expr: Expr):
    super().__init__(owner)
    self.patterns = patterns
    self.expr = expr

  def __str__(self) -> Text:
    return '{} => {}'.format(' | '.join(str(p) for p in self.patterns),
                             self.expr)

  def __repr__(self) -> Text:
    return 'MatchArm(patterns={!r}, expr={!r})'.format(self.patterns, self.expr)

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.expr.accept(visitor)

  @property
  def span(self) -> Span:
    return Span(self.patterns[0].span.start, self.expr.span.limit)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = self.expr.get_free_variables(start_pos)
    for pattern in self.patterns:
      freevars = freevars.union(pattern.get_free_variables(start_pos))
    return freevars


# TODO(leary): The matcher will be able to introduce name bindings at some
# point. For example:
#
#   match arr {
#     u8[2]:[3, x] => x
#     u8[2]:[5, x] => x*2
#     _ => fail!()
#   }
#
# TODO(leary): Need to check the matcher can conform to the type of the
# "matched" expression. For example:
#
#     match u32:42 {
#       u8:64 => ...;  // Should be a type error.
#     }
class Match(Expr):
  """Represents a match (pattern-match) expression."""

  def __init__(self, owner: AstNodeOwner, span: Span, matched: Expr,
               arms: Tuple[MatchArm, ...]):
    super().__init__(owner, span)
    self.matched = matched
    self.arms = arms

  def get_wildcard_arm(self) -> Optional[MatchArm]:
    try:
      return next(
          arm for arm in self.arms if any(
              isinstance(pattern, WildcardPattern) for pattern in arm.patterns))
    except StopIteration:
      return None

  def __repr__(self) -> Text:
    return 'Match(matched={!r}, arms={!r})'.format(self.matched, self.arms)

  def __str__(self) -> Text:
    return """match ({}) {{
  {}
}}""".format(self.matched, ';\n  '.join(str(arm) for arm in self.arms))

  def _accept_children(self, visitor: AstVisitor) -> None:
    self.matched.accept(visitor)
    for arm in self.arms:
      arm.accept(visitor)

  def get_free_variables(self, start_pos: Pos) -> FreeVariables:
    freevars = self.matched.get_free_variables(start_pos)
    for arm in self.arms:
      freevars = freevars.union(arm.get_free_variables(start_pos))
    return freevars
