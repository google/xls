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

#include "xls/common/status/ret_check.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"

namespace xls::dslx {

// An entry on the "stack of functions currently being deduced".
class FnStackEntry {
 public:
  // Creates an entry for type inference of function 'f' with the given symbolic
  // bindings.
  static FnStackEntry Make(Function* f, SymbolicBindings symbolic_bindings) {
    return FnStackEntry(f, std::move(symbolic_bindings));
  }
  // Creates an entry for type inference of the top level of module 'module'.
  static FnStackEntry MakeTop(Module* module) { return FnStackEntry(module); }

  // Represents a "representation" string for use in debugging, as in Python.
  std::string ToReprString() const;

  // Returns true if this entry describes the given function.
  bool Matches(const Function* f) const;

  const std::string& name() const { return name_; }
  Function* function() const { return function_; }
  const Module* module() const { return module_; }
  const SymbolicBindings& symbolic_bindings() const {
    return symbolic_bindings_;
  }

 private:
  FnStackEntry(Function* f, SymbolicBindings symbolic_bindings)
      : function_(f),
        name_(f->identifier()),
        module_(f->owner()),
        symbolic_bindings_(symbolic_bindings) {}

  // Constructor overload for a module-level inference entry.
  explicit FnStackEntry(Module* module)
      : function_(nullptr), name_("<top>"), module_(module) {}

  Function* function_;
  std::string name_;
  const Module* module_;
  SymbolicBindings symbolic_bindings_;
};

class DeduceCtx;  // Forward decl.

// Callback signature for the "top level" of the node type-deduction process.
using DeduceFn = std::function<absl::StatusOr<std::unique_ptr<ConcreteType>>(
    AstNode*, DeduceCtx*)>;

// Signature used for typechecking a single function within a module (this is
// generally used for typechecking parametric instantiations).
using TypecheckFunctionFn = std::function<absl::Status(Function*, DeduceCtx*)>;

// A single object that contains all the state/callbacks used in the
// typechecking process.
class DeduceCtx {
 public:
  DeduceCtx(TypeInfo* type_info, Module* module, DeduceFn deduce_function,
            TypecheckFunctionFn typecheck_function,
            TypecheckFn typecheck_module,
            absl::Span<const std::filesystem::path> additional_search_paths,
            ImportData* import_data);

  // Creates a new DeduceCtx reflecting the given type info and module.
  // Uses the same callbacks as this current context.
  //
  // Note that the resulting DeduceCtx has an empty fn_stack.
  std::unique_ptr<DeduceCtx> MakeCtx(TypeInfo* new_type_info,
                                     Module* new_module) const {
    return absl::make_unique<DeduceCtx>(
        new_type_info, new_module, deduce_function_, typecheck_function_,
        typecheck_module_, additional_search_paths_, import_data_);
  }

  // Helper that calls back to the top-level deduce procedure for the given
  // node.
  absl::StatusOr<std::unique_ptr<ConcreteType>> Deduce(AstNode* node) {
    XLS_RET_CHECK_EQ(node->owner(), type_info()->module());
    return deduce_function_(node, this);
  }

  std::vector<FnStackEntry>& fn_stack() { return fn_stack_; }
  const std::vector<FnStackEntry>& fn_stack() const { return fn_stack_; }

  Module* module() const { return module_; }
  TypeInfo* type_info() const { return type_info_; }

  // Creates a new TypeInfo that has the current type_info_ as its parent.
  void AddDerivedTypeInfo() {
    type_info_ = type_info_owner().New(module(), /*parent=*/type_info_).value();
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
  absl::optional<FnStackEntry> PopFnStackEntry() {
    if (fn_stack_.empty()) {
      return absl::nullopt;
    }
    FnStackEntry result = fn_stack_.back();
    fn_stack_.pop_back();
    return result;
  }

  const TypecheckFn& typecheck_module() const { return typecheck_module_; }
  const TypecheckFunctionFn& typecheck_function() const {
    return typecheck_function_;
  }

  absl::Span<std::filesystem::path const> additional_search_paths() const {
    return additional_search_paths_;
  }
  ImportData* import_data() const { return import_data_; }
  TypeInfoOwner& type_info_owner() const {
    return import_data_->type_info_owner();
  }

 private:
  // Maps AST nodes to their deduced types.
  TypeInfo* type_info_ = nullptr;

  // The (entry point) module we are typechecking.
  Module* module_;

  // -- Callbacks

  // Callback used to enter the top-level deduction routine.
  DeduceFn deduce_function_;

  // Typechecks parametric functions that are not in this module.
  TypecheckFunctionFn typecheck_function_;

  // Callback used to typecheck a module and get its type info (e.g. on import).
  TypecheckFn typecheck_module_;

  // Additional paths to search on import.
  std::vector<std::filesystem::path> additional_search_paths_;

  // Cache used for imported modules, may be nullptr.
  ImportData* import_data_;

  // -- Metadata

  // Keeps track of the function we're currently typechecking and the symbolic
  // bindings that deduction is running on.
  std::vector<FnStackEntry> fn_stack_;
};

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(
    const SymbolicBindings& symbolic_bindings);

// Creates a (stylized) TypeInferenceError status message that will be thrown as
// an exception when it reaches the Python pybind boundary.
absl::Status TypeInferenceErrorStatus(const Span& span,
                                      const ConcreteType* type,
                                      absl::string_view message);

// Creates a (stylized) XlsTypeError status message that will be thrown as an
// exception when it reaches the Python pybind boundary.
absl::Status XlsTypeErrorStatus(const Span& span, const ConcreteType& lhs,
                                const ConcreteType& rhs,
                                absl::string_view message);

// Pair of:
// * A node whose type is missing, and:
// * Optionally, what we observed was the user of that node.
//
// For example, a NameDef node may be missing a type, and the user of the
// NameDef may be an Invocation node.
struct NodeAndUser {
  AstNode* node;
  AstNode* user;
};

// Parses the AST node values out of the TypeMissingError message.
//
// TODO(leary): 2020-12-14 This is totally horrific (laundering these pointer
// values through Statuses that get thrown as Python exceptions), but it will
// get us through the port...
NodeAndUser ParseTypeMissingErrorMessage(absl::string_view s);

// Creates a TypeMissingError status value referencing the given node (which has
// its type missing) and user (which found that its type was missing). User will
// often be null at the start, and the using deduction rule will later populate
// it into an updated status.
absl::Status TypeMissingErrorStatus(AstNode* node, AstNode* user);

// Returns whether the "status" is a TypeMissingErrorStatus().
//
// If so, it should be possible to call ParseTypeMissingErrorMessage() on its
// message contents.
bool IsTypeMissingErrorStatus(const absl::Status& status);

absl::Status ArgCountMismatchErrorStatus(const Span& span,
                                         absl::string_view message);

// Returned when an invalid identifier (invalid at some position in the
// compilation chain, DSLX, IR, or Verilog) is encountered.
absl::Status InvalidIdentifierErrorStatus(const Span& span,
                                          absl::string_view message);

// Makes a constexpr environment suitable for passing to
// Interpreter::InterpExpr(). This will be populated with symbolic bindings
// as well as a constexpr freevars of "node", which is useful when there are
// local const bindings closed over e.g. in function scope.
//
// `type_info` is required to look up the value of previously computed
// constexprs.
absl::flat_hash_map<std::string, InterpValue> MakeConstexprEnv(
    Expr* node, const SymbolicBindings& symbolic_bindings, TypeInfo* type_info);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DEDUCE_CTX_H_
