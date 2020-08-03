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

# pylint: disable=unused-import

# `type` is a kwarg in the interface of `Node`.
# pylint: disable=redefined-builtin

"""Extends pycparser's AST with select C++ features."""

# These are actually used: they are used in translator.py
# We need to import in this style to merge the old c_ast and extended
from pycparser.c_ast import ArrayDecl
from pycparser.c_ast import ArrayRef
from pycparser.c_ast import Assignment
from pycparser.c_ast import BinaryOp
from pycparser.c_ast import Break
from pycparser.c_ast import Case
from pycparser.c_ast import Cast
from pycparser.c_ast import Compound
from pycparser.c_ast import Constant
from pycparser.c_ast import Continue
from pycparser.c_ast import Decl
from pycparser.c_ast import DeclList
from pycparser.c_ast import Default
from pycparser.c_ast import EmptyStatement
from pycparser.c_ast import Enum
from pycparser.c_ast import Enumerator
from pycparser.c_ast import EnumeratorList
from pycparser.c_ast import ExprList
from pycparser.c_ast import For
from pycparser.c_ast import FuncCall
from pycparser.c_ast import FuncDecl
from pycparser.c_ast import FuncDef
from pycparser.c_ast import ID
from pycparser.c_ast import IdentifierType
from pycparser.c_ast import If
from pycparser.c_ast import InitList
from pycparser.c_ast import Node
from pycparser.c_ast import Pragma
from pycparser.c_ast import Return
from pycparser.c_ast import Struct
from pycparser.c_ast import StructRef
from pycparser.c_ast import Switch
from pycparser.c_ast import TernaryOp
from pycparser.c_ast import TypeDecl
from pycparser.c_ast import Typename
from pycparser.c_ast import UnaryOp


class RefDecl(Node):
  """C++ style reference declaration: type &x.
  """
  __slots__ = ('quals', 'type', 'coord', '__weakref__')

  def __init__(self, quals, type, coord=None):
    self.quals = quals
    self.type = type
    self.coord = coord

  def children(self):
    nodelist = []
    if self.type is not None:
      nodelist.append(('type', self.type))
    return tuple(nodelist)

  def __iter__(self):
    if self.type is not None:
      yield self.type

  attr_names = ('quals',)


class TemplateInst(Node):
  """C++ style template instantiation: expr<param, param, ...>.
  """

  __slots__ = ('expr', 'params_list', 'coord', '__weakref__')

  def __init__(self, expr, params_list, coord=None):
    self.expr = expr
    self.params_list = params_list
    self.coord = coord

  def children(self):
    nodelist = []
    nodelist.append(('expr', self.expr))
    nodelist.append(('params', self.params_list))
    return tuple(nodelist)

  def __iter__(self):
    if self.expr is not None:
      yield self.expr

  attr_names = ()
