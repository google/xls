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

#ifndef XLS_DSLX_TYPE_SYSTEM_DEDUCE_CTX_H_
#define XLS_DSLX_TYPE_SYSTEM_DEDUCE_CTX_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

enum class WithinProc : uint8_t {
  kNo,
  kYes,
};

// An entry on the "stack of functions/procs currently being deduced".
class FnStackEntry {
 public:
  // Creates an entry for type inference of function 'f' with the given symbolic
  // bindings.
  static FnStackEntry Make(Function& f, ParametricEnv parametric_env,
                           WithinProc within_proc) {
    return FnStackEntry(f, f.identifier(), f.owner(), std::move(parametric_env),
                        std::nullopt, within_proc);
  }

  static FnStackEntry Make(Function& f, ParametricEnv parametric_env,
                           const Invocation* invocation,
                           WithinProc within_proc) {
    return FnStackEntry(f, f.identifier(), f.owner(), std::move(parametric_env),
                        invocation, within_proc);
  }

  // Creates an entry for type inference of the top level of module 'module'.
  static FnStackEntry MakeTop(Module* module) { return FnStackEntry(module); }

  // Represents a "representation" string for use in debugging, as in Python.
  std::string ToReprString() const;

  const std::string& name() const { return name_; }

  // Note: f() can be nullptr when this entry represents a module-level
  // evaluation, i.e. created via `MakeTop()`.
  Function* f() const { return f_; }

  const Module* module() const { return module_; }
  const ParametricEnv& parametric_env() const { return parametric_env_; }
  std::optional<const Invocation*> invocation() { return invocation_; }
  WithinProc within_proc() const { return within_proc_; }

  bool operator!=(std::nullptr_t) const { return f_ != nullptr; }

 private:
  FnStackEntry(Function& f, std::string name, Module* module,
               ParametricEnv parametric_env,
               std::optional<const Invocation*> invocation,
               WithinProc within_proc)
      : f_(&f),
        name_(std::move(name)),
        module_(module),
        parametric_env_(std::move(parametric_env)),
        invocation_(invocation),
        within_proc_(within_proc) {}

  // Constructor overload for a module-level inference entry.
  explicit FnStackEntry(Module* module)
      : f_(nullptr),
        name_("<top>"),
        module_(module),
        within_proc_(WithinProc::kNo) {}

  // Note: can be nullptr when the entry represents the "top level" of a module.
  Function* f_;

  std::string name_;
  const Module* module_;
  ParametricEnv parametric_env_;
  std::optional<const Invocation*> invocation_;
  WithinProc within_proc_;
};

class DeduceCtx;  // Forward decl.

// Callback signature for the "top level" of the node type-deduction process.
using DeduceFn = std::function<absl::StatusOr<std::unique_ptr<Type>>(
    const AstNode*, DeduceCtx*)>;

// Signature used for typechecking a single function within a module (this is
// generally used for typechecking parametric instantiations).
using TypecheckFunctionFn = std::function<absl::Status(Function&, DeduceCtx*)>;

// Similar to TypecheckFunctionFn, but for a [parametric] invocation.
using TypecheckInvocationFn =
    std::function<absl::StatusOr<TypeAndParametricEnv>(
        DeduceCtx* ctx, const Invocation*,
        const absl::flat_hash_map<std::variant<const Param*, const ProcMember*>,
                                  InterpValue>&)>;

// A single object that contains all the state/callbacks used in the
// typechecking process.
class DeduceCtx {
 public:
  DeduceCtx(TypeInfo* type_info, Module* module, DeduceFn deduce_function,
            TypecheckFunctionFn typecheck_function,
            TypecheckModuleFn typecheck_module,
            TypecheckInvocationFn typecheck_invocation, ImportData* import_data,
            WarningCollector* warnings, DeduceCtx* parent);

  // Creates a new DeduceCtx reflecting the given type info and module.
  // Uses the same callbacks as this current context.
  //
  // Note that the resulting DeduceCtx has an empty fn_stack.
  std::unique_ptr<DeduceCtx> MakeCtx(TypeInfo* new_type_info,
                                     Module* new_module);

  // Helper that calls back to the top-level deduce procedure for the given
  // node.
  absl::StatusOr<std::unique_ptr<Type>> Deduce(const AstNode* node);

  // To report structured information on typechecking mismatches we record
  // metadata on the DeduceCtx object.
  //
  // Args:
  //  mismatch_span: The span that is displayed as the "source of" the type
  //    mismatch in the error message. We can currently only display a single
  //    span as "the source".
  //  lhs_node: Optional passing of the node that led to the "lhs" type, used
  //    only for diagnostic-message-formation purposes.
  //  lhs: Left hand side type that led to the mismatch.
  //  rhs_node: Akin to lhs_node above.
  //  rhs: Akin to lhs above.
  //  message; Message to be displayed on what the type mismatch came from /
  //    represents -- note this may end up supplemented by diagnostic
  //    information for display.
  absl::Status TypeMismatchError(Span mismatch_span, const AstNode* lhs_node,
                                 const Type& lhs, const AstNode* rhs_node,
                                 const Type& rhs, std::string message);

  bool WithinProc() const {
    return fn_stack().back().within_proc() == WithinProc::kYes;
  }

  ParametricEnv GetCurrentParametricEnv() const {
    return fn_stack().empty() ? ParametricEnv()
                              : fn_stack().back().parametric_env();
  }

  std::vector<FnStackEntry>& fn_stack() { return fn_stack_; }
  const std::vector<FnStackEntry>& fn_stack() const { return fn_stack_; }

  const WarningCollector* warnings() const { return warnings_; }
  WarningCollector* warnings() { return warnings_; }

  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }

  // Creates a new TypeInfo that has the current type_info_ as its parent.
  //
  // Returns the new (derived) type info object. This can later be passed to
  // "PopDerivedTypeInfo()" to check soundness of our type info stack.
  TypeInfo* AddDerivedTypeInfo();

  // Puts the given TypeInfo on top of the current stack.
  absl::Status PushTypeInfo(TypeInfo* ti);

  // Pops the current type_info_ and sets the type_info_ to be the popped
  // value's parent (conceptually an inverse of AddDerivedTypeInfo()).
  absl::Status PopDerivedTypeInfo(TypeInfo* expect_popped);

  // Adds an entry to the stack of functions currently being deduced.
  void AddFnStackEntry(FnStackEntry entry);

  // Gets the current function stack as a string suitable for debugging.
  //
  // E.g. XLS_VLOG_LINES(3, ctx->GetFnStackDebugString());
  std::string GetFnStackDebugString() const;

  // Pops an entry from the stack of functions currently being deduced and
  // returns it, conceptually the inverse of AddFnStackEntry().
  std::optional<FnStackEntry> PopFnStackEntry();

  const TypecheckModuleFn& typecheck_module() const {
    return typecheck_module_;
  }
  const TypecheckFunctionFn& typecheck_function() const {
    return typecheck_function_;
  }
  const TypecheckInvocationFn& typecheck_invocation() const {
    return typecheck_invocation_;
  }

  ImportData* import_data() const { return import_data_; }
  TypeInfoOwner& type_info_owner() const {
    return import_data_->type_info_owner();
  }

  bool in_typeless_number_ctx() const { return in_typeless_number_ctx_; }
  void set_in_typeless_number_ctx(bool in_typeless_number_ctx) {
    in_typeless_number_ctx_ = in_typeless_number_ctx;
  }

  const std::optional<TypeMismatchErrorData>& type_mismatch_error_data() const {
    return type_mismatch_error_data_;
  }

 private:
  // Maps AST nodes to their deduced types.
  TypeInfo* type_info_ = nullptr;

  // The (entry point) module we are typechecking.
  Module* module_;

  // -- Callbacks

  // Callback used to enter the top-level deduction routine.
  DeduceFn deduce_function_;

  TypecheckFunctionFn typecheck_function_;
  TypecheckModuleFn typecheck_module_;
  TypecheckInvocationFn typecheck_invocation_;

  // Cache used for imported modules, may be nullptr.
  ImportData* import_data_;

  // Object used for collecting warnings flagged in the type checking process.
  WarningCollector* warnings_;

  // If there is a "parent" deduce ctx, closer to the root of the typechecking
  // process.
  DeduceCtx* parent_;

  // True if we're in a context where we could process an unannotated number,
  // such as when deducing an array index.
  bool in_typeless_number_ctx_ = false;

  // -- Metadata

  // Keeps track of the function we're currently typechecking and the symbolic
  // bindings that deduction is running on.
  std::vector<FnStackEntry> fn_stack_;

  // Keeps track of any type mismatch error that is currently active.
  std::optional<TypeMismatchErrorData> type_mismatch_error_data_;
};

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(const ParametricEnv& parametric_env);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_DEDUCE_CTX_H_
