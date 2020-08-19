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

# pylint: disable=C6102,C6108,C6113,C6109,C6105,C6012,C6110,C6114
#
# Some grammar rule docstrings are over 80 chars.
# pylint: disable=line-too-long

"""Extends pycparser's parsing with select C++ features.
"""

from pycparser import c_parser

from xls.contrib.xlscc.parse import ext_c_ast as c_ast
from xls.contrib.xlscc.parse import ext_c_lexer


class CParserBase(c_parser.CParser):
  """Extends CParser to use XLSccParser.
  """

  def __init__(self, **kwds):
    kwds['lexer'] = self.lexer_class
    kwds['lextab'] = 'pycparserext.lextab'
    kwds['yacctab'] = 'pycparserext.yacctab'
    c_parser.CParser.__init__(self, **kwds)

  def parse(self, text, filename='', debuglevel=0, initial_type_symbols=()):
    self.clex.filename = filename
    self.clex.reset_lineno()

    # _scope_stack[-1] is the current (topmost) scope.

    initial_scope = dict((tpsym, 1) for tpsym in initial_type_symbols)
    initial_scope.update(
        dict((tpsym, 1) for tpsym in self.initial_type_symbols))
    self._scope_stack = [initial_scope]

    if not text or text.isspace():
      return c_ast.FileAST([])
    else:
      return self.cparser.parse(text, lexer=self.clex, debug=debuglevel)


class XLSccParser(CParserBase):
  """Extends CParser with select C++ features.
  """
  lexer_class = ext_c_lexer.XLSccLexer

  is_intrinsic_type = lambda self, name: False

  initial_type_symbols = set([])

  def _is_type_in_scope(self, name):
    if self.is_intrinsic_type(name):
      return True

    return c_parser.CParser._is_type_in_scope(self, name)  # pylint: disable=protected-access

  def p_cast_expression_3(self, p):
    """cast_expression : type_specifier LPAREN expression RPAREN """
    p[0] = c_ast.Cast(p[1], p[3], self._token_coord(p, 1))

  def p_pointer(self, p):
    """pointer : AND type_qualifier_list_opt"""
    coord = self._token_coord(p, 1)
    nested_type = c_ast.RefDecl(quals=p[2] or [], type=None, coord=coord)
    p[0] = nested_type

  def p_pointer_2(self, p):
    """pointer : TIMES type_qualifier_list_opt"""
    raise NotImplementedError("Pointers are not supported")

  def p_constant_4(self, p):
    """constant    : TRUE
                   | FALSE
      """
    p[0] = c_ast.Constant('bool', p[1], self._token_coord(p, 1))

  def p_template_decl(self, p):
    """template_decl   : typedef_name identifier
                       | type_specifier_no_typeid identifier
      """
    p[0] = None

  def p_template_decl_list(self, p):
    """template_decl_list    : template_decl
                             | template_decl_list COMMA template_decl
      """
    p[0] = None

  def p_struct_or_union_specifier_3(self, p):
    """ struct_or_union_specifier   : struct_or_union ID brace_open struct_declaration_list brace_close
                                    | struct_or_union ID brace_open brace_close
                                    | struct_or_union TYPEID brace_open struct_declaration_list brace_close
                                    | struct_or_union TYPEID brace_open brace_close
      """
    klass = self._select_struct_union_class(p[1])
    if len(p) == 5:
      p[0] = klass(
          name=p[2],
          decls=[],
          coord=self._token_coord(p, 2))
    else:
      p[0] = klass(
          name=p[2],
          decls=p[4],
          coord=self._token_coord(p, 2))
    self._add_typedef_name(p[2], self._token_coord(p, 2))

  def p_function_definition_base(self, p):
    """function_definition_base : declaration_specifiers id_declarator declaration_list_opt compound_statement"""
    spec = p[1]

    p[0] = self._build_function_definition(
        spec=spec, decl=p[2], param_decls=p[3], body=p[4])

  def p_function_definition_1(self, p):
    """function_definition : function_definition_base"""
    p[0] = p[1]

  def p_function_definition_2(self, p):
    """function_definition : TEMPLATE LT template_decl_list GT function_definition_base"""
    p[0] = p[5]

  def p_template_val_expression(self, p):
    """template_val_expression   : typedef_name
                                 | type_specifier_no_typeid
                                 | constant
                                 | identifier
                                 | template_val_expression PLUS template_val_expression
                                 | template_val_expression MINUS template_val_expression
                                 | template_val_expression TIMES template_val_expression
                                 | template_val_expression DIVIDE template_val_expression
                                 | template_val_expression MOD template_val_expression
                                 | template_val_expression CONDOP template_val_expression COLON template_val_expression
      """
    p[0] = p[1]

  def p_template_val_list(self, p):
    """template_val_list    : template_val_expression
                            | template_val_list COMMA template_val_expression
      """
    # ExprList
    if len(p) == 2:
      p[0] = c_ast.ExprList([p[1]], coord=self._token_coord(p, 1))
    else:
      p[1].exprs.append(p[2])
      p[0] = p[1]

  def p_template_inst(self, p):
    """template_inst  : ID LT template_val_list GT"""
    # TODO(seanhaskell): Use IdentifierType here too?
    p[0] = c_ast.TemplateInst(p[1], p[3], coord=self._token_coord(p, 1))

  def p_postfix_expression_4(self, p):
    """postfix_expression  : postfix_expression PERIOD ID
                           | postfix_expression PERIOD TYPEID
                           | postfix_expression ARROW ID
                           | postfix_expression ARROW TYPEID
                           | postfix_expression PERIOD template_inst
                           | postfix_expression DBLCOLON ID
      """
    field = c_ast.ID(p[3], self._token_coord(p, 3))
    p[0] = c_ast.StructRef(p[1], p[2], field, p[1].coord)

  def p_direct_id_declarator_7(self, p):
    """ direct_id_declarator   : id_declarator DBLCOLON id_declarator"""
    # Special case for FuncDecls
    # TODO(seanhaskell): Expand to generic support as in original XLS[cc]
    assert isinstance(p[1], c_ast.TypeDecl)
    assert isinstance(p[3], c_ast.FuncDecl)
    assert isinstance(p[3].type, c_ast.TypeDecl)
    f = p[3]
    f.type.declname = p[1].declname + '__' + f.type.declname
    p[0] = f

  def p_typedef_name(self, p):
    """typedef_name : TYPEID

                    | TYPEID LT template_val_list GT
      """
    id_ = c_ast.IdentifierType([p[1]], coord=self._token_coord(p, 1))
    if len(p) == 2:
      p[0] = id_
    else:
      p[0] = c_ast.TemplateInst(id_, p[3], coord=self._token_coord(p, 1))

  def p_type_specifier_no_typeid(self, p):
    """type_specifier_no_typeid  : VOID
                                 | CHAR
                                 | SHORT
                                 | INT
                                 | LONG
                                 | FLOAT
                                 | DOUBLE
                                 | _COMPLEX
                                 | SIGNED
                                 | UNSIGNED
                                 | __INT128
                                 | _BOOL
                                 | BOOL
    """
    p[0] = c_ast.IdentifierType([p[1]], coord=self._token_coord(p, 1))
