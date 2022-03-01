// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_EVALUATE_SYM_H_
#define XLS_DSLX_EVALUATE_SYM_H_

#include "absl/status/statusor.h"
#include "xls/dslx/abstract_interpreter.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/concolic_test_generator.h"
#include "xls/dslx/evaluate.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

// Symbolic counterparts to node evaluator functions.
//
// EvaluateSym* evalautes the node concretely similar to its twin Evaluate*;
// as well as the expression tree for that node.
#define DISPATCH_DECL(__expr_type)                                             \
  absl::StatusOr<InterpValue> EvaluateSym##__expr_type(                        \
      __expr_type* expr, InterpBindings* bindings, ConcreteType* type_context, \
      AbstractInterpreter* interp, ConcolicTestGenerator* test_generator);

DISPATCH_DECL(Array)
DISPATCH_DECL(Attr)
DISPATCH_DECL(Binop)
DISPATCH_DECL(Cast)
DISPATCH_DECL(ColonRef)
DISPATCH_DECL(ConstRef)
DISPATCH_DECL(For)
DISPATCH_DECL(Index)
DISPATCH_DECL(Let)
DISPATCH_DECL(Match)
DISPATCH_DECL(NameRef)
DISPATCH_DECL(Number)
DISPATCH_DECL(SplatStructInstance)
DISPATCH_DECL(String)
DISPATCH_DECL(StructInstance)
DISPATCH_DECL(Ternary)
DISPATCH_DECL(Unop)
DISPATCH_DECL(XlsTuple)

#undef DISPATCH_DECL

absl::StatusOr<InterpValue> EvaluateSymFunction(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    const SymbolicBindings& symbolic_bindings, AbstractInterpreter* interp,
    ConcolicTestGenerator* test_generator);

}  // namespace xls::dslx

#endif  // XLS_DSLX_EVALUATE_SYM_H_
