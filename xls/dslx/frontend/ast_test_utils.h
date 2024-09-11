// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_DSLX_FRONTEND_AST_TEST_UTILS_H_
#define XLS_DSLX_FRONTEND_AST_TEST_UTILS_H_

#include <cstdint>
#include <tuple>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// Returns an AST node with the following structure:
//
//  x as t < x
//
// We have to parenthesize the LHS to avoid ambiguity that the RHS of the cast
// might be a parametric type we're instantiating.
std::tuple<FileTable, Module, Binop*> MakeCastWithinLtComparison();

// Returns an AST node with the following structure:
//
//  (x as u32[4])[i]  // without parens noted in the AST
//
// This is an interesting test case for parenthesization purposes similar to the
// above example.
std::tuple<FileTable, Module, Index*> MakeCastWithinIndexExpression();

// Returns an AST node with the following structure:
//
//  (x[i]).2  // without parens noted in the AST.
//
// This is an interesting test case for parenthesization purposes similar to the
// above example.
std::tuple<FileTable, Module, TupleIndex*>
MakeIndexWithinTupleIndexExpression();

// Returns an AST node with the following structure:
//
// (x0,x1,...,x(n-1)[,])
//
// This can be used to test that tuples of various sizes with or without
// trailing commas are formatted correctly.
std::tuple<FileTable, Module, XlsTuple*> MakeNElementTupleExpression(
    int64_t n, bool has_trailing_comma);

// Returns an AST node with the following structure:
//
//  -(x as u32)  // without parens noted in the AST.
//
// This is an interesting test case for parenthesization purposes similar to the
// above example.
std::tuple<FileTable, Module, Unop*> MakeCastWithinNegateExpression();

// Returns an AST node with the following structure:
//
//  (x*y).my_attr  // without parens noted in the AST.
//
// This is an interesting test case for parenthesization purposes similar to the
// above example.
std::tuple<FileTable, Module, Attr*> MakeArithWithinAttrExpression();

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_TEST_UTILS_H_
