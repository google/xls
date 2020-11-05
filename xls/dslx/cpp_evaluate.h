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

// Creates the top level bindings for a given module. We may not be able to
// create a *complete* set of bindings if we've re-entered this routine; e.g. in
// evaluating a top-level constant we recur to ask what enums (or similar) are
// available in the module scope -- in those cases we populate as many top level
// bindings as we can before we reach the work-in-progress point.
//
// Args:
//  typecheck: Typecheck callback, as we may need to typecheck imported modules.
//  eval: Evaluation callback, as we may need to evaluate top level constants.
//  is_wip: Query for whether a constant is in the process of being evaluated
//    (re-entrancy check).
//  note_wip: Notes the evaluation state for a constant (either marks it as work
//    in progress via nullopt, or provides its value).
//  cache: Cache of imported modules and their type information.
absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, const TypecheckFn& typecheck,
    const EvaluateFn& eval, const IsWipFn& is_wip, const NoteWipFn& note_wip,
    ImportCache* cache);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CPP_EVALUATE_H_
