// Copyright 2020 The XLS Authors
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

#ifndef XLS_DSLX_CPP_EVALUATE_H_
#define XLS_DSLX_CPP_EVALUATE_H_

#include "absl/status/statusor.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"

namespace xls::dslx {

// Evaluates an index bit-slicing operation (as opposed to a width-slice), of
// the form `x[2:4]`.
absl::StatusOr<InterpValue> EvaluateIndexBitslice(TypeInfo* type_info,
                                                  Index* expr,
                                                  InterpBindings* bindings,
                                                  const Bits& bits);

// Note: all interpreter "node evaluators" have the same signature.

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context);

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context);

// Callback used to determine if a constant definition (at the module scope) is
// in the process of being evaluated -- this lets us detect re-entry (i.e. a top
// level constant that wants our top-level bindings to do the evaluation needs
// to make forward progress using definitions previous to it in the file).
using IsWipFn =
    std::function<bool(const std::shared_ptr<Module>&, ConstantDef*)>;

// Callback used to note that a constant evaluation is "work in progress"
// (underway) -- this is noted by passing nullopt before a call to evaluate it.
// Once the constant is fully evaluated, the callback will be invoked with the
// given value. If this callback returns a non-nullopt value, the constant had
// already been evaluated (and was cached).
using NoteWipFn = std::function<absl::optional<InterpValue>(
    const std::shared_ptr<Module>&, ConstantDef*, absl::optional<InterpValue>)>;

// Callback used to evaluate an expression (e.g. a constant at the top level).
using EvaluateFn = std::function<absl::StatusOr<InterpValue>(
    const std::shared_ptr<Module>&, Expr*, InterpBindings*)>;

// Bundles up the above callbacks so they can be passed around as a unit.
struct InterpCallbackData {
  TypecheckFn typecheck;
  EvaluateFn eval;
  IsWipFn is_wip;
  NoteWipFn note_wip;
  ImportCache* cache;
};

// Creates the top level bindings for a given module. We may not be able to
// create a *complete* set of bindings if we've re-entered this routine; e.g. in
// evaluating a top-level constant we recur to ask what enums (or similar) are
// available in the module scope -- in those cases we populate as many top level
// bindings as we can before we reach the work-in-progress point.
//
// Args:
//   module: The top-level module to make bindings for.
//   callbacks: Provide ability to call back into the interpreter facilities
//    e.g. on import or for evaluating constant value expressions.
absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, InterpCallbackData* callbacks);

using ConcretizeVariant = absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;

// Resolve "type" into a concrete type via expression evaluation.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    InterpCallbackData* callbacks);

// Resolves (parametric) dimensions from deduction vs the current bindings.
absl::StatusOr<int64> ResolveDim(
    absl::variant<Expr*, int64, ConcreteTypeDim> dim, InterpBindings* bindings);

using DerefVariant = absl::variant<TypeAnnotation*, EnumDef*, StructDef*>;

// Returns the type_definition dereferenced into a Struct or Enum or
// TypeAnnotation.
//
// Will produce TypeAnnotation in the case we bottom out in a tuple, for
// example.
//
// Args:
//   node: Node to resolve to a struct/enum/annotation.
//   bindings: Current bindings for evaluating the node.
absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    InterpCallbackData* callbacks);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EVALUATE_H_
