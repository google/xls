# Lint as: python3
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

"""Recursive descent parser and AST nodes for XLS syntax.

The parser consumes token streams and produces AST output.

See sha256_sample.xls for an example of a sizable function that is parsed by
this grammar.
"""

import sys
from typing import Optional, Tuple, Union, List, TypeVar, Callable, Text, Dict, Any

from absl import logging
import dataclasses

from xls.dslx import ast
from xls.dslx import dslx_builtins
from xls.dslx import token_parser
from xls.dslx.bindings import Bindings
from xls.dslx.parse_error import ParseError
from xls.dslx.scanner import Keyword
from xls.dslx.scanner import Pos
from xls.dslx.scanner import Scanner
from xls.dslx.scanner import Token
from xls.dslx.scanner import TokenKind
from xls.dslx.scanner import TYPE_KEYWORDS
from xls.dslx.span import Span


# Helper data for noting which operator-signifying tokens bind tightly (strong
# arithmetic expressions) vs weakly (weak arithmetic expressions). We use
# grammar productions in lieu of precedence climbing because it's a bit easier
# to read and we're not concerned about parsing performance.
#
# TODO(leary) logical expressions, bitwise expressions (even weaker).
_STRONG_ARITHMETIC_KINDS = (
    TokenKind.STAR,
    TokenKind.SLASH,
    TokenKind.PERCENT,
)  # type: Tuple[TokenKind]
_WEAK_ARITHMETIC_KINDS = (
    TokenKind.PLUS,
    TokenKind.DOUBLE_PLUS,
    TokenKind.MINUS,
)  # type: Tuple[TokenKind]
_BITWISE_KINDS = (
    TokenKind.DOUBLE_OANGLE,
    TokenKind.DOUBLE_CANGLE,
    TokenKind.TRIPLE_CANGLE,
)  # type: Tuple[TokenKind]


@dataclasses.dataclass
class _ParserOptions:
  let_terminator_is_semi: bool


class Parser(token_parser.TokenParser):
  """Recursive-descent-parses the scanner's token stream into AST structures."""

  def __init__(self, scanner: Scanner):
    super(Parser, self).__init__(scanner)
    self._loop_stack = []
    self._options = _ParserOptions(let_terminator_is_semi=False)

  def _parse_comma_seq(self,
                       fparse: Callable[..., TypeVar('T')],
                       terminator: Union[TokenKind, Keyword],
                       args: Tuple[Any, ...] = ()) -> Tuple[TypeVar('T'), ...]:
    """Helper that parses a comma-delimited sequence of grammatical productions.

    Expects the caller to have popped the "initiator" token; however, this
    (callee) pops the terminator token so the caller does not need to.

    Permits a trailing comma.

    Args:
      fparse: Parses the grammatical production (i.e. the thing after each
        comma).
      terminator: Token that terminates the sequence; e.g. ')' or ']' or similar
        (may be a keyword).
      args: Arguments to be passed to fparse on each invocation as "*args".

    Returns:
      Tuple of the parsed things yielded by fparse.
    """

    def try_pop_terminator() -> bool:
      if isinstance(terminator, TokenKind):
        return self._try_popt(terminator)
      return self._try_pop_keyword(terminator)

    def drop_terminator_or_error():
      if isinstance(terminator, TokenKind):
        self._dropt_or_error(terminator)
      else:
        self._drop_keyword_or_error(terminator)

    parsed = []
    must_end = False
    while True:
      if try_pop_terminator():
        break
      if must_end:
        drop_terminator_or_error()
        break
      parsed.append(fparse(*args))
      must_end = not self._try_popt(TokenKind.COMMA)
    return tuple(parsed)

  def _parse_dims(self, bindings: Bindings) -> Tuple[ast.Expr, ...]:
    """Parses dimensions on a type; e.g. u32[3] => (3,); uN[2][3] => (3, 2)."""
    dims = []
    self._dropt_or_error(TokenKind.OBRACK)
    dims.append(self.parse_expression(bindings))
    self._dropt_or_error(TokenKind.CBRACK)
    # See if there are more dims like "bar" and "baz" in a T[foo][bar][baz] sort
    # of fashion.
    while self._try_popt(TokenKind.OBRACK):
      dims.append(self.parse_expression(bindings))
      self._dropt_or_error(TokenKind.CBRACK)
    # Reversed to emulate the old [major,minor] style of syntax.
    return tuple(reversed(dims))

  def _parse_mod_type_ref(self, bindings: Bindings,
                          start_tok: Token) -> ast.TypeRef:
    self._popt_or_error(TokenKind.DOUBLE_COLON)
    import_ = bindings.resolve_node(start_tok.value, start_tok.span)
    if not isinstance(import_, ast.Import):
      raise ParseError(
          start_tok.span, 'Expected module for module-reference; got {}'.format(
              import_.__class__.__name__))
    type_name = self._popt_or_error(TokenKind.IDENTIFIER)
    span = Span(start_tok.span.start, self._get_pos())
    mod_ref = ast.ModRef(span, import_, type_name)
    composite = '{}::{}'.format(start_tok.value, type_name.value)
    return ast.TypeRef(span, composite, mod_ref)

  def _parse_type_ref(self, bindings: Bindings, tok: Token) -> ast.TypeRef:
    """Parses a reference to type, may be a single token or module reference."""
    logging.vlog(5, 'Parsing type-ref; tok: %r', tok)
    if tok.kind != TokenKind.IDENTIFIER:
      raise ParseError(tok.span,
                       'Expected type; got {}'.format(tok.to_error_str()))

    if self._peekt_is(TokenKind.DOUBLE_COLON):
      return self._parse_mod_type_ref(bindings, tok)

    type_def = bindings.resolve_node(tok.value, tok.span)
    if not isinstance(type_def, (ast.TypeDef, ast.Enum, ast.Struct)):
      raise ParseError(
          tok.span,
          "Expected a type, but identifier {!r} doesn't resolve to a type, "
          'it resolved to a {}'.format(tok.value, type_def.__class__.__name__))

    return ast.TypeRef(tok.span, tok.value, type_def)

  def _parse_type_annotation(self,
                             bindings: Bindings,
                             tok: Optional[Token] = None) -> ast.TypeAnnotation:
    """Returns TypeAnnotation AST by parsing XLS-recognized type constructs."""
    if tok is None:
      tok = self._popt()

    if tok.is_keyword_in(TYPE_KEYWORDS):
      # Builtin types.
      if self._peekt_is(TokenKind.OBRACK):
        dims = self._parse_dims(bindings)
        return ast.TypeAnnotation(
            Span(tok.span.start, self._get_pos()), tok, dims)
      return ast.TypeAnnotation(tok.span, tok, dims=())

    if tok.kind == TokenKind.OPAREN:  # Tuple of types.
      types = self._parse_comma_seq(
          self._parse_type_annotation, TokenKind.CPAREN, args=(bindings,))
      span = Span(tok.span.start, self._get_pos())
      return ast.TypeAnnotation.make_tuple(span, tok, types)

    type_ref = self._parse_type_ref(bindings, tok)
    # Type ref may be followed by dimensions.
    if self._peekt_is(TokenKind.OBRACK):
      dims = self._parse_dims(bindings)
      return ast.TypeAnnotation(
          Span(tok.span.start, self._get_pos()), type_ref, dims)

    span = Span(tok.span.start, self._get_pos())
    return ast.TypeAnnotation(span, type_ref)

  def _parse_name_ref(self,
                      bindings: Bindings,
                      tok: Optional[Token] = None) -> ast.NameRef:
    """Parses a reference to a name (binding)."""
    if tok is None:
      tok = self._popt_or_error(TokenKind.IDENTIFIER)
    name_def = bindings.resolve(tok.value, tok.span)
    if isinstance(bindings.resolve_node(tok.value, tok.span), ast.Constant):
      return ast.ConstRef(tok, name_def)
    return ast.NameRef(tok, name_def)

  def _parse_colon_ref(self, bindings: Bindings,
                       subject_tok: Token) -> Union[ast.EnumRef, ast.ModRef]:
    """Parses a reference to a module or enum value."""
    logging.vlog(5, 'Parsing colon-ref; subject: %r', subject_tok)
    assert self._peekt_is(TokenKind.DOUBLE_COLON), self._peekt()
    defn = bindings.resolve_node(subject_tok.value, subject_tok.span)
    if not isinstance(defn, (ast.Enum, ast.Import, ast.TypeDef)):
      raise ParseError(
          subject_tok.span, 'Name {!r} does not refer to a module or type, '
          "expected module or type for '::' value reference.".format(
              subject_tok.value))
    self._popt_or_error(TokenKind.DOUBLE_COLON)
    value_tok = self._popt_or_error(TokenKind.IDENTIFIER)
    span = Span(subject_tok.span.start, value_tok.span.limit)
    if isinstance(defn, ast.Import):
      return ast.ModRef(span, defn, value_tok)

    assert isinstance(defn, (ast.Enum, ast.TypeDef)), defn
    return ast.EnumRef(span, defn, value_tok)

  def _parse_cast_or_enum_ref_or_struct_instance(
      self, tok: Token, bindings: Bindings) -> ast.Expr:
    logging.vlog(
        5, 'Parsing cast-or-enum-ref-or-struct-instance; start token: %s', tok)
    if self._peekt_is(TokenKind.DOUBLE_COLON):
      return self._parse_colon_ref(bindings, tok)
    type_ = self._parse_type_annotation(bindings, tok)
    if self._peekt_is(TokenKind.OBRACE):
      return self._parse_struct_instance(bindings, type_)
    return self._parse_cast(bindings, type_)

  def _resolve_struct(
      self, bindings: Bindings,
      type_: ast.TypeAnnotation) -> Union[ast.Struct, ast.ModRef]:
    assert isinstance(type_, ast.TypeAnnotation), type_
    assert type_.is_typeref(), type_
    typeref = type_.get_typeref()
    struct = typeref.type_def

    while isinstance(struct, ast.TypeDef):
      struct = self._resolve_struct(bindings, struct.type_)

    assert isinstance(struct, (ast.ModRef, ast.Struct)), struct
    return struct

  def _parse_struct_instance(
      self,
      bindings: Bindings,
      type_: Optional[ast.TypeAnnotation] = None) -> ast.Expr:
    """Parses a struct instantiation expression."""
    if type_ is None:
      type_ = self._parse_type_annotation(bindings)

    struct = self._resolve_struct(bindings, type_)

    self._dropt_or_error(
        TokenKind.OBRACE, context='Opening brace for struct instance.')

    def parse_struct_member() -> Tuple[Text, ast.Expr]:
      tok = self._popt_or_error(TokenKind.IDENTIFIER)
      self._dropt_or_error(TokenKind.COLON)
      e = self.parse_expression(bindings)
      return (tok.value, e)

    members = self._parse_comma_seq(parse_struct_member, TokenKind.CBRACE)
    limit_pos = self._get_pos()
    span = Span(type_.span.start, limit_pos)
    return ast.StructInstance(span, struct, members)

  def _parse_cast_or_struct_instance(self, bindings: Bindings) -> ast.Expr:
    logging.vlog(5, 'Parsing cast-or-struct-instance')
    type_ = self._parse_type_annotation(bindings)
    if self._peekt_is(TokenKind.COLON):
      return self._parse_cast(bindings, type_)
    return self._parse_struct_instance(bindings, type_)

  def _parse_name_or_colon_ref(
      self, bindings: Bindings) -> Union[ast.EnumRef, ast.NameRef, ast.ModRef]:
    """Parses either a name reference or an enum value reference."""
    logging.vlog(5, 'Parsing name-or-colon-ref.')
    tok = self._popt_or_error(TokenKind.IDENTIFIER)
    if self._peekt_is(TokenKind.DOUBLE_COLON):
      return self._parse_colon_ref(bindings, tok)
    return self._parse_name_ref(bindings, tok)

  def _parse_name_def(self, bindings: Bindings) -> ast.NameDef:
    tok = self._popt_or_error(TokenKind.IDENTIFIER)
    name_def = ast.NameDef(tok)
    bindings.add(name_def.identifier, name_def)
    return name_def

  def _parse_name_def_or_wildcard(self, bindings: Bindings
                                 ) -> Union[ast.NameDef, ast.WildcardPattern]:
    tok = self._try_pop_identifier_token('_')
    if tok:
      return ast.WildcardPattern(tok)
    return self._parse_name_def(bindings)

  def _parse_name_def_tree(self, bindings: Bindings) -> ast.NameDefTree:
    """Parses tree of name defs and return it.

    For example, the left hand side of:

      let (a, (b, (c)), d) = ...

    This is used for tuple-like (sometimes known as "destructuring") let
    binding.

    Args:
      bindings: Bindings to populate with NameDefs from the tree.

    Returns:
      The parsed tree of name-defs.
    """
    start = self._popt_or_error(TokenKind.OPAREN)

    def parse_name_def_or_tree(bindings: Bindings) -> ast.NameDefTree:
      if self._peekt_is(TokenKind.OPAREN):
        return self._parse_name_def_tree(bindings)
      name_def = self._parse_name_def_or_wildcard(bindings)
      return ast.NameDefTree(name_def.span, name_def)

    branches = self._parse_comma_seq(
        parse_name_def_or_tree, TokenKind.CPAREN, args=(bindings,))
    return ast.NameDefTree(
        Span(start.span.start, self._get_pos()), tuple(branches))

  def _parse_num(self, bindings: Bindings) -> ast.Number:
    """Returns a parsed number (literal number) expression."""
    tok = self._peekt()
    if tok.kind in (TokenKind.NUMBER, TokenKind.CHARACTER):
      return ast.Number(self._popt())
    if tok.is_keyword_in((Keyword.TRUE, Keyword.FALSE)):
      return ast.Number(self._popt())

    # Numbers can also be given as u32:4 -- last ditch effort to parse one of
    # those.
    try:
      cast_node = self._parse_cast(bindings)
    except ParseError:
      pass
    else:
      if isinstance(cast_node, ast.Number):
        return cast_node

    raise ParseError(tok.span,
                     'Expected number; got {} @ {}'.format(tok.kind, tok.span))

  def _parse_const_ref(self, bindings: Bindings) -> ast.NameRef:
    raise NotImplementedError

  def _parse_num_or_const_ref(
      self, bindings: Bindings) -> Union[ast.Number, ast.NameRef]:
    if self._peekt_is(TokenKind.IDENTIFIER):
      return self._parse_const_ref(bindings)
    return self._parse_num(bindings)

  def _parse_let(self, bindings: Bindings) -> ast.Let:
    """Returns a parsed 'let' expression."""
    new_bindings = Bindings(bindings)
    start_tok = self._popt()
    if start_tok.is_keyword(Keyword.LET):
      const = False
    elif start_tok.is_keyword(Keyword.CONST):
      const = True
    else:
      raise ParseError(
          start_tok.span, 'Expected "let" or "const"; got {} @ {}'.format(
              start_tok, start_tok.span))
    if self._peekt_is(TokenKind.OPAREN):  # Destructuring binding.
      name_def = None
      name_def_tree = self._parse_name_def_tree(new_bindings)
    else:
      name_def = self._parse_name_def(new_bindings)
      name_def_tree = ast.NameDefTree(name_def.span, name_def)
    if self._try_popt(TokenKind.COLON):
      annotated_type = self._parse_type_annotation(bindings)
    else:
      annotated_type = None
    self._dropt_or_error(TokenKind.EQUALS)
    rhs = self.parse_expression(bindings)
    if self._options.let_terminator_is_semi:
      self._dropt_or_error(TokenKind.SEMI)
    else:
      self._drop_keyword_or_error(Keyword.IN)
    if const and name_def:
      const = ast.Constant(name_def, rhs)
      new_bindings.add(name_def.identifier, const)
    else:
      const = None
    body = self.parse_expression(new_bindings)
    span = Span(start_tok.span.start, self._get_pos())
    return ast.Let(name_def_tree, annotated_type, rhs, body, span, const=const)

  def _parse_tuple_remainder(self, start_pos: Pos, first: ast.Expr,
                             bindings: Bindings) -> ast.XlsTuple:
    """Parses the remainder of a tuple expression.

    We can't tell until we've parsed the first expression whether we're parsing
    a parenthesized expression; e.g. '(x)' or a tuple expression '(x, y)' -- as
    a result we use this helper routine once we discover we're parsing a tuple
    instead of a parenthesized expression, which is why "first" is passed from
    the caller.

    Args:
      start_pos: The position of the '(' token that started this tuple.
      first: The parse expression in the tuple as already parsed by the caller.
      bindings: Bindings to use in the parsing of the tuple expression.

    Returns:
      The tuple-expression with all of its sub-expressions contained inside,
      including "first".
    """
    self._dropt_or_error(TokenKind.COMMA)
    es = self._parse_comma_seq(
        self.parse_expression, TokenKind.CPAREN, args=(bindings,))
    span = Span(start_pos, self._get_pos())
    return ast.XlsTuple(span, (first,) + es)

  def _parse_array(self, bindings: Bindings) -> ast.Array:
    """Parses an array AST node (starting with cursor over '[')."""
    start_pos = self._popt_or_error(TokenKind.OBRACK).span.start

    class EllipsisSentinel(object):
      """Sentinel object that represents an ellipsis "..." token in an array."""

      def __init__(self, span: Span):
        self.span = span

    def parse_ellipsis_or_expression(bindings
                                    ) -> Union[ast.Expr, EllipsisSentinel]:
      if self._peekt_is(TokenKind.ELLIPSIS):
        token = self._popt()
        return EllipsisSentinel(token.span)
      return self.parse_expression(bindings)

    members = self._parse_comma_seq(
        parse_ellipsis_or_expression, TokenKind.CBRACK, args=(bindings,))
    span = Span(start_pos, self._get_pos())
    has_trailing_elipsis = False
    for i, member in enumerate(members):
      if isinstance(member, EllipsisSentinel):
        if i + 1 == len(members):
          has_trailing_elipsis = True
          members = members[:-1]
        else:
          raise ParseError(member.span,
                           'Ellipsis may only be in trailing position.')
    if all(ast.Constant.is_constant(m) for m in members):
      return ast.ConstantArray(members, has_trailing_elipsis, span)
    return ast.Array(members, has_trailing_elipsis, span)

  def _parse_cast(self,
                  bindings: Bindings,
                  type_: Optional[ast.TypeAnnotation] = None) -> ast.Expr:
    """Parses a cast or an explicitly-typed number and returns it."""
    if type_ is None:
      try:
        type_ = self._parse_type_annotation(bindings)
      except ParseError as e:
        raise ParseError(
            e.span, 'Expected a type as part of a cast expression: {}'.format(
                e.message))
    self._dropt_or_error(TokenKind.COLON)
    term = self._parse_term(bindings)
    if isinstance(term, (ast.Number, ast.Array)):
      term.type_ = type_
      return term
    if isinstance(term, ast.XlsTuple) and all(
        ast.Constant.is_constant(m) for m in term.members):
      return term
    raise ParseError(
        type_.span,
        'Old-style cast only permitted for constant arrays/tuples and literal numbers.'
    )

  def _parse_term(self, bindings: Bindings) -> ast.Expr:
    """Parses a term as a component of an expression and returns it.

    Terms are more atomic than arithmetic expressions.

    Args:
      bindings: Bindings to use in the parsing of the term.

    Returns:
      The parsed term as an expression AST node.

    Raises:
      ParseError: When the current token is not a valid start of an expression
        term.
    """
    tok = self._peekt()
    logging.vlog(5, 'Parsing term; tok: %r', tok)
    if tok.kind in (TokenKind.NUMBER, TokenKind.CHARACTER) or tok.is_keyword_in(
        (Keyword.TRUE, Keyword.FALSE)):
      lhs = self._parse_num(bindings)
    elif tok.is_keyword(Keyword.NEXT):
      # Next is also used as a special function call inside a proc that says it
      # should proceed to the next iteration.
      self._dropt()
      self._try_popt(TokenKind.OPAREN)
      args = self._parse_comma_seq(
          self.parse_expression, TokenKind.CPAREN, args=(bindings,))
      lhs = ast.Next(tok.span)
      lhs = ast.Invocation(Span(tok.span.start, self._get_pos()), lhs, args)
    elif (tok.is_type_keyword() or
          (tok.kind == TokenKind.IDENTIFIER and isinstance(
              bindings.resolve_node_or_none(tok.value),
              (ast.Enum, ast.TypeDef, ast.Struct)))):
      lhs = self._parse_cast_or_enum_ref_or_struct_instance(
          self._popt(), bindings)
    elif tok.kind == TokenKind.IDENTIFIER:
      lhs = self._parse_name_or_colon_ref(bindings)
      if isinstance(lhs, ast.ModRef) and self._peekt_is(TokenKind.OBRACE):
        type_ = ast.TypeAnnotation(lhs.span,
                                   ast.TypeRef(lhs.span, str(lhs), lhs))
        return self._parse_struct_instance(bindings, type_)
    elif tok.is_keyword(Keyword.CARRY):
      self._dropt()
      lhs = ast.Carry(tok.span, self._loop_stack[-1])
    elif tok.kind == TokenKind.OPAREN:  # Parenthesized expression.
      oparen = self._popt()
      if self._try_popt(TokenKind.CPAREN):
        lhs = ast.XlsTuple(Span(oparen.span.start, self._get_pos()), tuple())
      else:
        lhs = self.parse_expression(bindings)
        if self._peekt_is(TokenKind.COMMA):
          lhs = self._parse_tuple_remainder(oparen.span.start, lhs, bindings)
        else:
          self._dropt_or_error(TokenKind.CPAREN, start=tok)
    elif tok.kind in (TokenKind.BANG, TokenKind.MINUS):
      return ast.Unop(self._popt(), self._parse_term(bindings))
    elif tok.is_keyword(Keyword.MATCH):
      return self._parse_match(bindings)
    elif tok.kind == TokenKind.OBRACK:
      lhs = self._parse_array(bindings)
    else:
      raise ParseError(
          tok.span,
          'Expected start of an expression; got: {}'.format(tok.to_error_str()))

    while True:
      new_pos = self._get_pos()

      if self._try_popt(TokenKind.OPAREN):  # Invocation.
        args = self._parse_comma_seq(
            self.parse_expression, TokenKind.CPAREN, args=(bindings,))
        lhs = ast.Invocation(Span(new_pos, self._get_pos()), lhs, args)
        continue

      if self._try_popt(TokenKind.DOT):  # Attribute.
        tok = self._popt_or_error(TokenKind.IDENTIFIER)
        attr = ast.NameDef(tok)
        span = Span(new_pos, self._get_pos())
        lhs = ast.Attr(span, lhs, attr)
        continue

      if self._try_popt(TokenKind.OBRACK):  # Indexing.
        if self._try_popt(TokenKind.COLON):  # Slice-from-beginning.
          lhs = self._parse_bitslice(new_pos, lhs, bindings, start=None)
          continue
        index = self.parse_expression(bindings)

        if self._try_popt(TokenKind.PLUS_COLON):  # Explicit-width slice.
          start = index
          width = self._parse_type_annotation(bindings)
          span = Span(new_pos, self._get_pos())
          width_slice = ast.WidthSlice(span, start, width)
          lhs = ast.Index(span, lhs, width_slice)
          self._popt_or_error(TokenKind.CBRACK)
          continue

        if self._try_popt(TokenKind.COLON):  # Slice-to-end.
          lhs = self._parse_bitslice(new_pos, lhs, bindings, start=index)
          continue

        self._popt_or_error(TokenKind.CBRACK)
        lhs = ast.Index(Span(new_pos, self._get_pos()), lhs, index)
        continue

      break

    return lhs

  def _parse_bitslice(self, start_pos: Pos, lhs: ast.Expr, bindings: Bindings,
                      start: Optional[ast.Expr]) -> ast.Index:
    """Parses a slicing index expression."""
    limit = None if self._peekt_is(
        TokenKind.CBRACK) else self.parse_expression(bindings)
    if not isinstance(start, (ast.Number, type(None))) or not isinstance(
        limit, (ast.Number, type(None))):
      raise ParseError(
          Span(start_pos, self._get_pos()),
          'Only constant numbers are currently allowed in slice expressions.')
    index = ast.Slice(
        Span(start_pos, self._get_pos()), start=start, limit=limit)
    self._popt_or_error(TokenKind.CBRACK)
    return ast.Index(Span(start_pos, self._get_pos()), lhs, index)

  def _parse_binop_chain(self, sub_production: Callable[..., ast.Expr],
                         target_tokens: Union[Tuple[TokenKind], Tuple[Keyword]],
                         *args) -> ast.Expr:
    """Parses a chain of binary operations at a given precedence level.

    For example, a sequence like "x + y + z" is left associative, so we form a
    left-leaning AST like:

      add(add(x, y), z)

    Generally a grammar production will join together two stronger production
    rules; e.g.

      WEAK_ARITHMETIC_EXPR ::=
        STRONG_ARITHMETIC_EXPR [+-] STRONG_ARITHMETIC_EXPR

    So that expressions like a*b + c*d work as expected, so the sub_production
    gives the more tightly binding production for this to call. After we call it
    for the "left hand side" we see if the token is in the target_token set
    (e.g. '+' or '-' in the example above), and if so, parse the "right hand
    side" to create a binary operation. If not, we simply return the result of
    the "left hand side" production (since we don't see the target token that
    indicates the kind of expression we're interested in).

    Args:
      sub_production: Parse method to delegate to that is more tightly binding
        (see explanation above).
      target_tokens: Tokens that form a binary operation at this level of the
        grammar (see explanation/example above).
      *args: Arguments to pass to the sub_production.

    Returns:
      An expression AST node.
    """
    lhs = sub_production(*args)
    while True:
      if self._peekt_in(target_tokens):
        op = self._popt()
        rhs = sub_production(*args)
        lhs = ast.Binop(op, lhs, rhs)
      else:
        break

    return lhs

  def _parse_cast_as_expression(self, bindings: Bindings) -> ast.Expr:
    lhs = self._parse_term(bindings)
    while self._try_pop_keyword(Keyword.AS):
      type_ = self._parse_type_annotation(bindings)
      lhs = ast.Cast(type_, lhs)
    return lhs

  def _parse_strong_arithmetic_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_cast_as_expression,
                                   _STRONG_ARITHMETIC_KINDS, bindings)

  def _parse_weak_arithmetic_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_strong_arithmetic_expression,
                                   _WEAK_ARITHMETIC_KINDS, bindings)

  def _parse_bitwise_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_weak_arithmetic_expression,
                                   _BITWISE_KINDS, bindings)

  def _parse_and_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_bitwise_expression,
                                   (TokenKind.AMPERSAND,), bindings)

  def _parse_xor_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_and_expression, (TokenKind.HAT,),
                                   bindings)

  def _parse_or_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_xor_expression, (TokenKind.BAR,),
                                   bindings)

  def _parse_comparison_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_or_expression,
                                   tuple(ast.Binop.COMPARISON_KINDS), bindings)

  def _parse_logical_and_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_comparison_expression,
                                   (TokenKind.DOUBLE_AMPERSAND,), bindings)

  def _parse_logical_or_expression(self, bindings: Bindings) -> ast.Expr:
    return self._parse_binop_chain(self._parse_logical_and_expression,
                                   (TokenKind.DOUBLE_BAR,), bindings)

  def _parse_ternary_expression(self, bindings: Bindings) -> ast.Expr:
    """Parses a ternary expression or expr of higher precedence.

    Example:

        foo if bar else baz

    Args:
      bindings: Current bindings for parsing this expression.

    Returns:
      An expression of ternary or higher precedence.
    """
    lhs = self._parse_logical_or_expression(bindings)
    new_pos = self._get_pos()
    if self._try_pop_keyword(Keyword.IF):  # Ternary.
      consequent = lhs
      test = self.parse_expression(bindings)
      self._pop_keyword_or_error(Keyword.ELSE)
      alternate = self.parse_expression(bindings)
      return ast.Ternary(
          Span(new_pos, self._get_pos()), test, consequent, alternate)
    return lhs

  def _parse_param(self, bindings: Bindings) -> ast.Param:
    name = self._parse_name_def(bindings)
    self._dropt_or_error(TokenKind.COLON)
    type_ = self._parse_type_annotation(bindings)
    logging.vlog(5, 'Parsed param name: %s type: %s', name, type_)
    return ast.Param(name, type_)

  def _parse_params(self, bindings: Bindings) -> Tuple[ast.Param, ...]:
    """Parses a sequence of parameters, starting with '(' ending after ')'.

    Permits trailing commas.

    Args:
      bindings: Bindings to populate with parameter name definitions / to
        resolve types against.

    Returns:
      The parsed sequence of parameter AST nodes.
    """
    self._dropt_or_error(TokenKind.OPAREN)
    return self._parse_comma_seq(
        self._parse_param, TokenKind.CPAREN, args=(bindings,))

  def _parse_tuple_pattern(self, start_pos: Pos,
                           bindings: Bindings) -> ast.NameDefTree:
    """Returns a parsed tuple from a pattern match."""
    members = []
    must_end = False
    while True:
      if self._try_popt(TokenKind.CPAREN):
        break
      if must_end:
        self._dropt_or_error(TokenKind.CPAREN)
        break
      members.append(self._parse_pattern(bindings))
      must_end = not self._try_popt(TokenKind.COMMA)
    span = Span(start_pos, self._get_pos())
    return ast.NameDefTree(span, tuple(members))

  def _parse_pattern(self, bindings: Bindings) -> ast.NameDefTree:
    """Returns a parsed pattern; e.g. one that would guard a match arm."""
    start_pos = self._get_pos()
    if self._try_popt(TokenKind.OPAREN):
      return self._parse_tuple_pattern(start_pos, bindings)

    if self._peekt_is(TokenKind.IDENTIFIER):
      tok = self._popt_or_error(TokenKind.IDENTIFIER)
      if tok.value == '_':
        return ast.NameDefTree(tok.span, ast.WildcardPattern(tok))
      if self._peekt_is(TokenKind.DOUBLE_COLON):
        return ast.NameDefTree(tok.span, self._parse_colon_ref(bindings, tok))
      resolved = bindings.resolve_or_none(tok.value)
      if resolved:
        assert isinstance(resolved, (ast.NameDef, ast.BuiltinNameDef)), resolved
        if isinstance(bindings.resolve_node(tok.value, tok.span), ast.Constant):
          ref = ast.ConstRef(tok, resolved)
        else:
          ref = ast.NameRef(tok, resolved)
        return ast.NameDefTree(tok.span, ref)
      name_def = ast.NameDef(tok)
      bindings.add(name_def.identifier, name_def)
      return ast.NameDefTree(tok.span, name_def)

    if self._peekt_in([
        TokenKind.NUMBER, TokenKind.CHARACTER, Keyword.TRUE, Keyword.FALSE
    ]) or self._peekt().is_keyword_in(TYPE_KEYWORDS):
      num = self._parse_num(bindings)
      return ast.NameDefTree(num.span, num)

    peekt = self._peekt()
    raise ParseError(peekt.span, 'Expected pattern; got {}'.format(peekt.kind))

  def _parse_match(self, bindings: Bindings) -> ast.Match:
    """Parses a match expression construct, starting on the 'match' keyword."""
    match_ = self._pop_keyword_or_error(Keyword.MATCH)

    matched = self.parse_expression(bindings)

    self._dropt_or_error(TokenKind.OBRACE)

    arms = []  # type: List[ast.MatchArm]
    must_end = False
    while True:
      if self._try_popt(TokenKind.CBRACE):
        break
      if must_end:
        self._dropt_or_error(
            TokenKind.CBRACE,
            context="Expected '}' since no ';' was seen "
            'to indicate an additional match case.')
        break
      arm_bindings = Bindings(bindings)
      patterns = [self._parse_pattern(arm_bindings)]
      while self._try_popt(TokenKind.BAR):
        if arm_bindings.has_local_bindings():
          raise ParseError(patterns[0].span,
                           'Cannot have multiple patterns that bind names.')
        patterns.append(self._parse_pattern(arm_bindings))
      self._dropt_or_error(TokenKind.FAT_ARROW)
      rhs = self.parse_expression(arm_bindings)
      arms.append(ast.MatchArm(tuple(patterns), rhs))
      must_end = not self._try_popt(TokenKind.SEMI)

    return ast.Match(
        Span(match_.span.start, self._get_pos()), matched, tuple(arms))

  def _parse_while(self, bindings: Bindings) -> ast.While:
    while_ = self._pop_keyword_or_error(Keyword.WHILE)
    while_bindings = Bindings(bindings)
    w = ast.While(while_.span)
    self._loop_stack.append(w)
    w.test = self.parse_expression(while_bindings)
    w.body = self._parse_block_expression(while_bindings)
    w.init = self._parse_parenthesized_expr(bindings)
    w.span = w.span.update_limit(self._get_pos())
    return w

  def _parse_for(self, bindings: Bindings) -> ast.For:
    """Parses a for loop construct; e.g.

      for (i, accum) in range(3) {
        accum + i
      }(0)

    The init value is passed to the loop and the body updates the value;
    ultimately the loop terminates and the final accum value is returned.

    Args:
      bindings: Bindings from the outer (function) scope.

    Returns:
      The parsed 'for' AST construct.
    """
    for_ = self._pop_keyword_or_error(Keyword.FOR)

    # We create a new binding scope for the per-iteration variables to be bound
    # inside of the loop.
    for_bindings = Bindings(bindings)
    names = self._parse_name_def_tree(for_bindings)
    self._dropt_or_error(
        TokenKind.COLON,
        context='expect type annotation on for-loop name bindings')
    type_ = self._parse_type_annotation(for_bindings)
    self._drop_keyword_or_error(Keyword.IN)
    iterable = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.OBRACE)

    body = self.parse_expression(for_bindings)
    self._dropt_or_error(TokenKind.CBRACE)
    self._dropt_or_error(
        TokenKind.OPAREN,
        for_,
        context='Need an initial accumulator value to start the for loop.')

    # We must be sure to use the outer bindings when parsing the init
    # expression, since the for loop bindings haven't happened yet (no loop
    # trips have iterated when the init value is evaluated).
    init = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.CPAREN)
    return ast.For(
        Span(for_.span.limit, self._get_pos()), names, type_, iterable, body,
        init)

  def _parse_enum(self, public: bool, bindings: Bindings) -> ast.Enum:
    """Parses an enum definition."""
    enum_tok = self._pop_keyword_or_error(Keyword.ENUM)
    name_def = self._parse_name_def(bindings)
    self._dropt_or_error(
        TokenKind.COLON,
        context="enum requires a ': type' annotation to indicate enum's underlying type."
    )
    type_ = self._parse_type_annotation(bindings)
    self._dropt_or_error(TokenKind.OBRACE)

    def parse_enum_entry(
    ) -> Tuple[ast.NameDef, Union[ast.Number, ast.NameRef]]:
      """Parses a single entry in an enum definition."""
      name_def = self._parse_name_def(bindings)
      self._dropt_or_error(TokenKind.EQUALS)
      value = self._parse_num_or_const_ref(bindings)
      if isinstance(value, ast.Number):
        if value.type_ is not None:
          raise ParseError(
              value.type_.span,
              'Type is annotated in enum value, but enum defines a type. '
              'Please remove the leading type-annotation.')
        value.type_ = type_
      entry = (name_def, value)
      logging.vlog(3, 'enum entry: %s', entry)
      return entry

    entries = self._parse_comma_seq(parse_enum_entry, TokenKind.CBRACE)
    enum = ast.Enum(enum_tok.span, public, name_def, type_, entries)
    bindings.add(name_def.identifier, enum)
    return enum

  def _parse_struct(self, public: bool, bindings: Bindings) -> ast.Struct:
    """Parses a struct definition."""
    self._drop_keyword_or_error(Keyword.STRUCT)
    name_def = self._parse_name_def(bindings)
    self._dropt_or_error(TokenKind.OBRACE)

    def parse_struct_member() -> Tuple[ast.NameDef, ast.TypeAnnotation]:
      tok = self._popt_or_error(TokenKind.IDENTIFIER)
      name_def = ast.NameDef(tok)
      self._dropt_or_error(TokenKind.COLON)
      type_ = self._parse_type_annotation(bindings)
      return (name_def, type_)

    members = self._parse_comma_seq(parse_struct_member, TokenKind.CBRACE)
    struct = ast.Struct(public, name_def, members)
    bindings.add(name_def.identifier, struct)
    return struct

  # Public API (sensible "entry" productions in the grammar).

  def parse_type_definition(self, public: bool,
                            bindings: Bindings) -> ast.TypeDef:
    self._drop_keyword_or_error(Keyword.TYPE)
    name_def = self._parse_name_def(bindings)
    self._dropt_or_error(TokenKind.EQUALS)
    type_ = self._parse_type_annotation(bindings)
    self._dropt_or_error(TokenKind.SEMI)
    type_def = ast.TypeDef(public, name_def, type_)
    bindings.add(name_def.identifier, type_def)
    return type_def

  def parse_expression(self, bindings: Bindings) -> ast.Expr:
    """Parses an expression out of the token stream.

    Args:
      bindings: Bindings to use for resolution in parsing the expression.

    Returns:
      The parsed expression as an AST node.
    """
    if sys.getrecursionlimit() < 65536:
      sys.setrecursionlimit(65536)
    tok = self._peekt()
    if tok.is_keyword(Keyword.LET) or tok.is_keyword(Keyword.CONST):
      return self._parse_let(bindings)
    if tok.is_keyword(Keyword.FOR):
      return self._parse_for(bindings)
    if tok.is_keyword(Keyword.WHILE):
      return self._parse_while(bindings)
    return self._parse_ternary_expression(bindings)

  def _parse_block_expression(self, bindings: Bindings) -> ast.Expr:
    self._dropt_or_error(TokenKind.OBRACE)
    e = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.CBRACE)
    return e

  def _parse_parenthesized_expr(self, bindings: Bindings) -> ast.Expr:
    self._dropt_or_error(TokenKind.OPAREN)
    e = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.CPAREN)
    return e

  def _parse_parametric_bindings(self, bindings: Bindings
                                ) -> Tuple[ast.ParametricBinding, ...]:
    """Parses parametric bindings that lead a function.

    For example:

      fn [X: u32, Y: u32 = X+X] f(x: bits[X]) { ... }
          ^-------------------^

    Note that some bindings have expressions and other do not, because they
    assume a value presented by the type of a formal parameter.

    Args:
      bindings: Bindings to populate with the parametric names.

    Returns:
      A tuple of the parsed parametric bindings.
    """

    def parse_parametric_binding() -> ast.ParametricBinding:
      name_def = self._parse_name_def(bindings)
      self._popt_or_error(TokenKind.COLON)
      type_ = self._parse_type_annotation(bindings)
      if self._try_popt(TokenKind.EQUALS):
        expr = self.parse_expression(bindings)
      else:
        expr = None
      return ast.ParametricBinding(name_def, type_, expr)

    return self._parse_comma_seq(parse_parametric_binding, TokenKind.CBRACK)

  def parse_proc(self, public: bool, outer_bindings: Bindings) -> ast.Proc:
    start_pos = self._get_pos()
    self._drop_keyword_or_error(Keyword.PROC)
    name_def = self._parse_name_def(outer_bindings)

    # Create bindings internal to this proc we're parsing, based off of the
    # symbols available in the outer bindings.
    bindings = Bindings(outer_bindings)

    proc_params = self._parse_params(bindings)
    self._dropt_or_error(TokenKind.OBRACE)

    # TODO(leary): Add support for configuration-time expressions in the proc
    # block.

    self._drop_keyword_or_error(Keyword.NEXT)
    iter_params = self._parse_params(bindings)
    self._dropt_or_error(TokenKind.OBRACE)
    body = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.CBRACE)

    self._dropt_or_error(TokenKind.CBRACE)
    return ast.Proc(
        Span(start_pos, self._get_pos()), name_def, proc_params, iter_params,
        body, public)

  def parse_function(self, public: bool,
                     outer_bindings: Bindings) -> ast.Function:
    """Parses a function out of the token stream.

    Args:
      public: Whether this function should be marked as public.
      outer_bindings: Bindings from the enclosing scope (containing this
        function being parsed).

    Returns:
      The parsed function as an AST node.
    """
    start_pos = self._get_pos()
    self._drop_keyword_or_error(Keyword.FN)

    # Create bindings internal to this function we're parsing, based off of the
    # symbols available in the outer bindings.
    bindings = Bindings(outer_bindings)

    parametric_bindings = ()
    if self._try_popt(TokenKind.OBRACK):  # Parametric.
      parametric_bindings = self._parse_parametric_bindings(bindings)

    name_def = self._parse_name_def(outer_bindings)
    bindings.add(name_def.identifier, name_def)
    params = self._parse_params(bindings)

    # Optional return type annotation.
    if self._try_popt(TokenKind.ARROW):
      return_type = self._parse_type_annotation(bindings)
      logging.vlog(5, 'Parsed return type: %s', return_type)
    else:
      return_type = None

    # Function body.
    self._dropt_or_error(TokenKind.OBRACE)
    body = self.parse_expression(bindings)
    logging.vlog(5, 'Function body: %r', body)
    self._dropt_or_error(
        TokenKind.CBRACE, context='Expected \'}\' at end of function body.')
    return ast.Function(
        Span(start_pos, self._get_pos()), name_def, parametric_bindings, params,
        return_type, body, public)

  def _parse_import(self, bindings: Bindings) -> ast.Import:
    """Parses an import statement into an Import AST node."""
    kw = self._pop_keyword_or_error(Keyword.IMPORT)

    toks = []
    tok = self._popt_or_error(TokenKind.IDENTIFIER)
    toks.append(tok)
    while self._try_popt(TokenKind.DOT):
      tok = self._popt_or_error(TokenKind.IDENTIFIER)
      toks.append(tok)
    subject = tuple(tok.value for tok in toks)

    if self._try_pop_keyword(Keyword.AS):
      name_def = self._parse_name_def(bindings)
      alias = name_def.identifier
    else:
      alias = None
      name_def = ast.NameDef(toks[-1])
    import_ = ast.Import(kw.span, subject, name_def, alias)
    bindings.add(name_def.identifier, import_)
    return import_

  def parse_test(self, outer_bindings: Bindings) -> ast.Test:
    self._drop_keyword_or_error(Keyword.TEST)
    fake_bindings = Bindings()
    name_def = self._parse_name_def(fake_bindings)
    bindings = Bindings(outer_bindings)
    self._dropt_or_error(TokenKind.OBRACE)
    body = self.parse_expression(bindings)
    self._dropt_or_error(TokenKind.CBRACE)
    return ast.Test(name_def, body)

  def parse_constant(self, bindings: Bindings) -> ast.Constant:
    """Parses a constant definition."""
    self._drop_keyword_or_error(Keyword.CONST)
    new_bindings = Bindings(parent=bindings)
    name_def = self._parse_name_def(new_bindings)

    # Explicitly check whether const bindings are shadowing anything, and give
    # an error if they do!
    name = name_def.identifier
    if bindings.has_name(name):
      raise ParseError(
          name_def.span,
          'Constant definition is shadowing an existing definition from {}'
          .format(bindings.resolve(name, name_def.span).get_span_or_fake()))

    self._dropt_or_error(TokenKind.EQUALS)
    expr = self._parse_cast(bindings)
    self._dropt_or_error(TokenKind.SEMI)
    if not ast.Constant.is_constant(expr):
      raise ParseError(expr.span,
                       'Value is not considered constant: {}'.format(expr))
    result = ast.Constant(name_def, expr)
    bindings.add(name_def.identifier, result)
    return result

  def _parse_directive(self) -> None:
    self._dropt_or_error(TokenKind.HASH)
    self._dropt_or_error(TokenKind.BANG)
    self._dropt_or_error(TokenKind.OBRACK)
    identifier = self._popt_or_error(TokenKind.IDENTIFIER)
    if identifier.value != 'cfg':
      raise ParseError(identifier.span,
                       'Unknown directive: {!r}'.format(identifier.value))
    self._dropt_or_error(TokenKind.OPAREN)
    config_name = self._popt_or_error(TokenKind.IDENTIFIER)
    self._dropt_or_error(TokenKind.EQUALS)
    config_value = self._popt_or_error(TokenKind.KEYWORD)
    if config_name.value == 'let_terminator_is_semi':
      if config_value.value not in (Keyword.TRUE, Keyword.FALSE):
        raise ParseError(
            config_value.span,
            'Invalid value for boolean directive; got {!r}'.format(
                config_value.value))
      self._options = dataclasses.replace(
          self._options,
          let_terminator_is_semi=config_value.value == Keyword.TRUE)
    else:
      raise ParseError(
          identifier.span,
          'Unknown configuration key in directive: {!r}'.format(
              config_name.value))
    self._dropt_or_error(TokenKind.CPAREN)
    self._dropt_or_error(TokenKind.CBRACK)

  def parse_module(self,
                   name: Text,
                   bindings: Optional[Bindings] = None) -> ast.Module:
    """Parses a module out of the token stream.

    This is the entry point that is generally called when parsing the text of an
    entire input syntax file.

    Args:
      name: Name that should be given to the parsed module.
      bindings: Optional initial bindings object (e.g. for use in testing).

    Returns:
      The parsed module as an AST node.

    Raises:
      ParseError: When an unexpected construct is enountered at module scope.
    """
    top = []  # type: List[ast.ModuleMember]

    # Populate the top-level bindings with the builtin names.
    bindings = bindings or Bindings()
    # Need this because pytype gets confused about whether it can be none.
    assert isinstance(bindings, Bindings), bindings
    for builtin_name in dslx_builtins.PARAMETRIC_BUILTIN_NAMES:
      bindings.add(builtin_name, ast.BuiltinNameDef(builtin_name))

    function_name_to_node = {}  # type: Dict[Text, ast.Function]

    def parse_function(public: bool) -> None:
      """Parses function w/given "public" visibility and appends it to top."""
      # Need this because pytype gets confused about whether it can be none.
      assert isinstance(bindings, Bindings), bindings
      f = self.parse_function(public, bindings)
      prior_node = function_name_to_node.get(f.identifier, None)
      if prior_node:
        raise ParseError(
            f.name.span,
            'Function {!r} is defined in this package multiple times;'
            'previously @ {}'.format(f.identifier, prior_node.name.span))
      function_name_to_node[f.identifier] = f
      top.append(f)

    while not self._at_eof():
      if self._peekt_is(TokenKind.EOF):
        break
      elif self._try_pop_keyword(Keyword.PUB):
        if self._peekt_is_keyword(Keyword.FN):
          parse_function(public=True)
        elif self._peekt_is_keyword(Keyword.STRUCT):
          top.append(self._parse_struct(True, bindings))
        elif self._peekt_is_keyword(Keyword.ENUM):
          top.append(self._parse_enum(True, bindings))
        elif self._peekt_is_keyword(Keyword.TYPE):
          top.append(self.parse_type_definition(True, bindings))
        else:
          raise ParseError(
              Span(self._get_pos(), self._get_pos()),
              'Expect function or struct after "pub" keyword.')
      elif self._peekt_is(TokenKind.HASH):
        self._parse_directive()
      elif self._peekt_is_keyword(Keyword.FN):
        parse_function(public=False)
      elif self._peekt_is_keyword(Keyword.TEST):
        top.append(self.parse_test(bindings))
      elif self._peekt_is_keyword(Keyword.IMPORT):
        top.append(self._parse_import(bindings))
      elif self._peekt_is_keyword(Keyword.TYPE):
        top.append(self.parse_type_definition(False, bindings))
      elif self._peekt_is_keyword(Keyword.STRUCT):
        top.append(self._parse_struct(False, bindings))
      elif self._peekt_is_keyword(Keyword.ENUM):
        top.append(self._parse_enum(False, bindings))
      elif self._peekt_is_keyword(Keyword.CONST):
        top.append(self.parse_constant(bindings))
      else:
        tok = self._peekt()
        raise ParseError(
            tok.span,
            'Expected start of top-level construct; got {}'.format(tok))
    return ast.Module(name, tuple(top))
