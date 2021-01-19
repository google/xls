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

#ifndef XLS_DSLX_INTERP_CALLBACK_DATA_H_
#define XLS_DSLX_INTERP_CALLBACK_DATA_H_

#include "absl/status/statusor.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Callback used to determine if a constant definition (at the module scope) is
// in the process of being evaluated -- this lets us detect re-entry (i.e. a top
// level constant that wants our top-level bindings to do the evaluation needs
// to make forward progress using definitions previous to it in the file).
using IsWipFn = std::function<bool(ConstantDef*)>;

// Callback used to note that a constant evaluation is "work in progress"
// (underway) -- this is noted by passing nullopt before a call to evaluate it.
// Once the constant is fully evaluated, the callback will be invoked with the
// given value. If this callback returns a non-nullopt value, the constant had
// already been evaluated (and was cached).
using NoteWipFn = std::function<absl::optional<InterpValue>(
    ConstantDef*, absl::optional<InterpValue>)>;

// Callback used to evaluate an expression (e.g. a constant at the top level).
using EvaluateFn = std::function<absl::StatusOr<InterpValue>(
    Expr*, InterpBindings*, std::unique_ptr<ConcreteType>)>;

// Callback used to call function values, either builtin or user-defined
// function.
using CallValueFn = std::function<absl::StatusOr<InterpValue>(
    const InterpValue&, absl::Span<const InterpValue>, const Span&, Invocation*,
    const SymbolicBindings*)>;

// Callback used to retrieve type information (from the interpreter state).
using GetTypeFn = std::function<const std::shared_ptr<TypeInfo>&()>;

// Bundles up the above callbacks so they can be passed around as a unit.
struct InterpCallbackData {
  TypecheckFn typecheck;
  EvaluateFn eval_fn;
  CallValueFn call_value_fn;
  IsWipFn is_wip;
  NoteWipFn note_wip;
  GetTypeFn get_type_info;
  ImportCache* cache;

  // Additional search paths to use on import.
  std::vector<std::string> additional_search_paths;

  // Wraps eval_fn to provide a default argument for type_context (as it is
  // often nullptr).
  absl::StatusOr<InterpValue> Eval(
      Expr* expr, InterpBindings* bindings,
      std::unique_ptr<ConcreteType> type_context = nullptr) {
    return this->eval_fn(expr, bindings, std::move(type_context));
  }
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_CALLBACK_DATA_H_
