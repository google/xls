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

#ifndef XLS_DSLX_DEDUCE_CTX_H_
#define XLS_DSLX_DEDUCE_CTX_H_

#include <filesystem>

#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/type_and_bindings.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// An entry on the "stack of functions/procs currently being deduced".
class FnStackEntry {
 public:
  // Creates an entry for type inference of function 'f' with the given symbolic
  // bindings.
  static FnStackEntry Make(Function* f, SymbolicBindings symbolic_bindings) {
    return FnStackEntry(f, f->identifier(), f->owner(),
                        std::move(symbolic_bindings), absl::nullopt);
  }

  static FnStackEntry Make(Function* f, SymbolicBindings symbolic_bindings,
                           const Invocation* invocation) {
    return FnStackEntry(f, f->identifier(), f->owner(),
                        std::move(symbolic_bindings), invocation);
  }

  // Creates an entry for type inference of the top level of module 'module'.
  static FnStackEntry MakeTop(Module* module) { return FnStackEntry(module); }

  // Represents a "representation" string for use in debugging, as in Python.
  std::string ToReprString() const;

  const std::string& name() const { return name_; }
  Function* f() const { return f_; }
  const Module* module() const { return module_; }
  const SymbolicBindings& symbolic_bindings() const {
    return symbolic_bindings_;
  }
  std::optional<const Invocation*> invocation() { return invocation_; }

  bool operator!=(std::nullptr_t) const { return f_ != nullptr; }

 private:
  FnStackEntry(Function* f, std::string name, Module* module,
               SymbolicBindings symbolic_bindings,
               std::optional<const Invocation*> invocation)
      : f_(f),
        name_(name),
        module_(module),
        symbolic_bindings_(symbolic_bindings),
        invocation_(invocation) {}

  // Constructor overload for a module-level inference entry.
  explicit FnStackEntry(Module* module)
      : f_(static_cast<Function*>(nullptr)), name_("<top>"), module_(module) {}

  Function* f_;
  std::string name_;
  const Module* module_;
  SymbolicBindings symbolic_bindings_;
  std::optional<const Invocation*> invocation_;
};

class DeduceCtx;  // Forward decl.

// Callback signature for the "top level" of the node type-deduction process.
using DeduceFn = std::function<absl::StatusOr<std::unique_ptr<ConcreteType>>(
    const AstNode*, DeduceCtx*)>;

// Signature used for typechecking a single function within a module (this is
// generally used for typechecking parametric instantiations).
using TypecheckFunctionFn = std::function<absl::Status(Function*, DeduceCtx*)>;

// Similar to TypecheckFunctionFn, but for a [parametric] invocation.
using TypecheckInvocationFn = std::function<absl::StatusOr<TypeAndBindings>(
    DeduceCtx* ctx, const Invocation*,
    const absl::flat_hash_map<const Param*, InterpValue>&)>;

// A single object that contains all the state/callbacks used in the
// typechecking process.
class DeduceCtx {
 public:
  DeduceCtx(TypeInfo* type_info, Module* module, DeduceFn deduce_function,
            TypecheckFunctionFn typecheck_function,
            TypecheckModuleFn typecheck_module,
            TypecheckInvocationFn typecheck_invocation, ImportData* import_data,
            WarningCollector* warnings);

  // Creates a new DeduceCtx reflecting the given type info and module.
  // Uses the same callbacks as this current context.
  //
  // Note that the resulting DeduceCtx has an empty fn_stack.
  std::unique_ptr<DeduceCtx> MakeCtx(TypeInfo* new_type_info,
                                     Module* new_module) const {
    return std::make_unique<DeduceCtx>(
        new_type_info, new_module, deduce_function_, typecheck_function_,
        typecheck_module_, typecheck_invocation_, import_data_, warnings_);
  }

  // Helper that calls back to the top-level deduce procedure for the given
  // node.
  absl::StatusOr<std::unique_ptr<ConcreteType>> Deduce(const AstNode* node) {
    XLS_RET_CHECK_EQ(node->owner(), type_info()->module())
        << "node: `" << node->ToString() << "` from module "
        << node->owner()->name()
        << " vs type info module: " << type_info()->module()->name();
    return deduce_function_(node, this);
  }

  std::vector<FnStackEntry>& fn_stack() { return fn_stack_; }
  const std::vector<FnStackEntry>& fn_stack() const { return fn_stack_; }

  const WarningCollector* warnings() const { return warnings_; }
  WarningCollector* warnings() { return warnings_; }

  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }

  // Creates a new TypeInfo that has the current type_info_ as its parent.
  void AddDerivedTypeInfo() {
    type_info_ = type_info_owner().New(module(), /*parent=*/type_info_).value();
  }

  // Puts the given TypeInfo on top of the current stack.
  absl::Status PushTypeInfo(TypeInfo* ti) {
    XLS_RET_CHECK_EQ(ti->parent(), type_info_);
    type_info_ = ti;
    return absl::OkStatus();
  }

  // Pops the current type_info_ and sets the type_info_ to be the popped
  // value's parent (conceptually an inverse of AddDerivedTypeInfo()).
  absl::Status PopDerivedTypeInfo() {
    XLS_RET_CHECK(type_info_->parent() != nullptr);
    type_info_ = type_info_->parent();
    return absl::OkStatus();
  }

  // Adds an entry to the stack of functions currently being deduced.
  void AddFnStackEntry(FnStackEntry entry) {
    fn_stack_.push_back(std::move(entry));
  }

  // Pops an entry from the stack of functions currently being deduced and
  // returns it, conceptually the inverse of AddFnStackEntry().
  std::optional<FnStackEntry> PopFnStackEntry() {
    if (fn_stack_.empty()) {
      return absl::nullopt;
    }
    FnStackEntry result = fn_stack_.back();
    fn_stack_.pop_back();
    return result;
  }

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

  // True if we're in a context where we could process an unannotated number,
  // such as when deducing an array index.
  bool in_typeless_number_ctx_ = false;

  // -- Metadata

  // Keeps track of the function we're currently typechecking and the symbolic
  // bindings that deduction is running on.
  std::vector<FnStackEntry> fn_stack_;
};

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(
    const SymbolicBindings& symbolic_bindings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DEDUCE_CTX_H_
