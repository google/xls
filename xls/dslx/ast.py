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

"""Contains definitions of AST nodes that result from the parsing process.

These have been chopped into separate files to make smaller inputs for pytype to
chew on.
"""

# pylint: disable=unused-import
from xls.dslx.ast_node import AstNode
from xls.dslx.ast_node import AstVisitor
from xls.dslx.core_ast_nodes import Array
from xls.dslx.core_ast_nodes import ArrayTypeAnnotation
from xls.dslx.core_ast_nodes import Binop
from xls.dslx.core_ast_nodes import BuiltinNameDef
from xls.dslx.core_ast_nodes import BuiltinTypeAnnotation
from xls.dslx.core_ast_nodes import ConstRef
from xls.dslx.core_ast_nodes import Enum
from xls.dslx.core_ast_nodes import EnumRef
from xls.dslx.core_ast_nodes import Expr
from xls.dslx.core_ast_nodes import Import
from xls.dslx.core_ast_nodes import make_builtin_type_annotation
from xls.dslx.core_ast_nodes import make_type_ref_type_annotation
from xls.dslx.core_ast_nodes import ModRef
from xls.dslx.core_ast_nodes import NameDef
from xls.dslx.core_ast_nodes import NameDefTree
from xls.dslx.core_ast_nodes import NameRef
from xls.dslx.core_ast_nodes import Number
from xls.dslx.core_ast_nodes import NumberKind
from xls.dslx.core_ast_nodes import Ternary
from xls.dslx.core_ast_nodes import TupleTypeAnnotation
from xls.dslx.core_ast_nodes import TypeAnnotation
from xls.dslx.core_ast_nodes import TypeDef
from xls.dslx.core_ast_nodes import TypeRef
from xls.dslx.core_ast_nodes import TypeRefTypeAnnotation
from xls.dslx.core_ast_nodes import WildcardPattern

from xls.dslx.leaf_ast_nodes import Attr
from xls.dslx.leaf_ast_nodes import Carry
from xls.dslx.leaf_ast_nodes import Cast
from xls.dslx.leaf_ast_nodes import Constant
from xls.dslx.leaf_ast_nodes import ConstantArray
from xls.dslx.leaf_ast_nodes import For
from xls.dslx.leaf_ast_nodes import Function
from xls.dslx.leaf_ast_nodes import Index
from xls.dslx.leaf_ast_nodes import Invocation
from xls.dslx.leaf_ast_nodes import Let
from xls.dslx.leaf_ast_nodes import Match
from xls.dslx.leaf_ast_nodes import MatchArm
from xls.dslx.leaf_ast_nodes import Module
from xls.dslx.leaf_ast_nodes import ModuleMember
from xls.dslx.leaf_ast_nodes import Next
from xls.dslx.leaf_ast_nodes import Param
from xls.dslx.leaf_ast_nodes import ParametricBinding
from xls.dslx.leaf_ast_nodes import Proc
from xls.dslx.leaf_ast_nodes import QuickCheck
from xls.dslx.leaf_ast_nodes import Slice
from xls.dslx.leaf_ast_nodes import SplatStructInstance
from xls.dslx.leaf_ast_nodes import Struct
from xls.dslx.leaf_ast_nodes import StructInstance
from xls.dslx.leaf_ast_nodes import StructInstanceMembers
from xls.dslx.leaf_ast_nodes import Test
from xls.dslx.leaf_ast_nodes import TestFunction
from xls.dslx.leaf_ast_nodes import Unop
from xls.dslx.leaf_ast_nodes import While
from xls.dslx.leaf_ast_nodes import WidthSlice
from xls.dslx.leaf_ast_nodes import XlsTuple
