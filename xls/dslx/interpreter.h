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

#ifndef XLS_DSLX_INTERPRETER_H_
#define XLS_DSLX_INTERPRETER_H_

#include "xls/dslx/abstract_interpreter.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

class Interpreter {
 public:
  // Function signature for a "post function-evaluation hook" -- this is invoked
  // after a function is evaluated by the interpreter. This is useful for e.g.
  // externally-implementing and hooking-in comparison to the JIT execution
  // mode.
  using PostFnEvalHook = std::function<absl::Status(
      Function* f, absl::Span<const InterpValue> args, const SymbolicBindings*,
      const InterpValue& got)>;

  // Helper used by type inference to evaluate "constexpr" expressions at type
  // checking time (e.g. derived parametric expressions, forced constexpr
  // evaluations for dimensions, etc.) to integral values.
  //
  // Creates top level bindings for the entry_module, adds the "env" to those
  // bindings, and then evaluates "expr" via an interpreter instance in that
  // environment.
  //
  // Args:
  //  entry_module: Entry-point module to be used in creating the interpreter.
  //  type_info: Type information (derived for the entry point) to be used in
  //    creating the interpreter.
  //  typecheck/import_data: Supplemental helpers used for import statements.
  //  env: Envionment of current parametric bindings.
  //  bit_widths: Bit widths for parametric bindings.
  //  expr: (Derived parametric) expression to evaluate.
  //  fn_ctx: Current function context.
  //
  // TODO(leary): 2020-11-24 This signature is for backwards compatibility with
  // a Python API, we can likely eliminate it when everything is ported over to
  // C++, or at least consolidate the env/bit_widths maps.
  static absl::StatusOr<InterpValue> InterpretExpr(
      Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
      absl::Span<std::filesystem::path const> additional_search_paths,
      ImportData* import_data,
      const absl::flat_hash_map<std::string, InterpValue>& env, Expr* expr,
      const FnCtx* fn_ctx = nullptr, ConcreteType* type_context = nullptr);

  // The same as above, but ensures the returned value is Bits-typed.
  static absl::StatusOr<Bits> InterpretExprToBits(
      Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
      absl::Span<std::filesystem::path const> additional_search_paths,
      ImportData* import_data,
      const absl::flat_hash_map<std::string, InterpValue>& env, Expr* expr,
      const FnCtx* fn_ctx = nullptr, ConcreteType* type_context = nullptr);

  // Creates an interpreter that can be used to interpreting entities
  // (functions, tests) within the given module.
  //
  // Note: typecheck and import_data args will likely be-provided or
  // not-be-provided together because they are both used in service of import
  // facilities.
  //
  // Args:
  //  module: "Entry" module wherein functions / tests are being interpreted by
  //    this interpreter.
  //  type_info: Type information associated with the given module -- evaluation
  //    of some AST nodes relies on this type information.
  //  typecheck: Optional, callback used to check modules on import.
  //  additional_search_paths: Additional paths to search for imported modules.
  //  import_data: Optional, cache for imported modules.
  //  trace_all: Whether to trace "all" (really most "non-noisy") expressions in
  //    the interpreter evaluation.
  //  trace_format_preference: The preferred format to use when executing
  //    `trace!()` builtins.
  //  post_fn_eval: Optional callback run after function evaluation. See
  //    PostFnEvalHook above.
  Interpreter(
      Module* entry_module, TypecheckFn typecheck,
      absl::Span<std::filesystem::path const> additional_search_paths,
      ImportData* import_data, bool trace_all = false,
      FormatPreference trace_format_preference = FormatPreference::kDefault,
      PostFnEvalHook post_fn_eval = nullptr);

  // Since we capture pointers to "this" in lambdas, we don't want this object
  // to move/copy/assign.
  Interpreter(Interpreter&&) = delete;
  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;

  // Runs a function with the given "name" from the module associated with the
  // interpreter, using the given "args" for the entry point invocation.
  // If this function is parametric, then symbolic_bindings needs to contain an
  // entry for each function parameter.
  absl::StatusOr<InterpValue> RunFunction(
      absl::string_view name, absl::Span<const InterpValue> args,
      SymbolicBindings symbolic_bindings = SymbolicBindings());

  // Searches for a test function with the given name in this interpreter's
  // module and, if found, runs it.
  absl::Status RunTest(absl::string_view name);

  absl::StatusOr<InterpValue> EvaluateLiteral(Expr* expr);

  Module* entry_module() const { return entry_module_; }
  TypeInfo* current_type_info() const { return current_type_info_; }

  FormatPreference trace_format_preference() const {
    return trace_format_preference_;
  }

 private:
  friend struct TypeInfoSwap;
  friend class Evaluator;
  friend class AbstractInterpreterAdapter;

  struct TypeInfoSwap {
    TypeInfoSwap(Interpreter* parent, absl::optional<TypeInfo*> new_type_info)
        : parent_(parent), old_type_info_(parent->current_type_info_) {
      if (new_type_info.has_value()) {
        parent->current_type_info_ = new_type_info.value();
      }
    }

    ~TypeInfoSwap() { parent_->current_type_info_ = old_type_info_; }

    Interpreter* parent_;
    TypeInfo* old_type_info_;
  };

  // Entry point for evaluating an expression to a value.
  //
  // Args:
  //   expr: Expression AST node to evaluate.
  //   bindings: Current bindings for this evaluation (i.e. ident: value map).
  //   type_context: If a type is deduced from surrounding context, it is
  //     provided via this argument.
  //
  // Returns:
  //   The value that the AST node evaluates to.
  //
  // Raises:
  //   EvaluateError: If an error occurs during evaluation. This also attempts
  //   to
  //     print a rough expression-stack-trace for determining the provenance of
  //     an error to ERROR log.
  absl::StatusOr<InterpValue> Evaluate(Expr* expr, InterpBindings* bindings,
                                       ConcreteType* type_context);

  // Evaluates an Invocation AST node to a value.
  absl::StatusOr<InterpValue> EvaluateInvocation(Invocation* expr,
                                                 InterpBindings* bindings,
                                                 ConcreteType* type_context);

  // Wraps function evaluation to compare with JIT execution.
  //
  // If this interpreter was not created with an IR package, this simply
  // evaluates the function. Otherwise, the function is executed with the LLVM
  // JIT and its return value is compared against the interpreted value as a
  // consistency check.
  //
  // TODO(leary): 2020-11-19 This is ok to run twice because there are no side
  // effects -- need to consider what happens when there are side effects (e.g.
  // fatal errors).
  //
  // Args:
  //  f: Function to evaluate
  //  args: Arguments used to invoke f
  //  span: Span of the invocation causing this evaluation
  //  invocation: Optional invocation node causing this function evaluation.
  //    Note invocation may be in a different module.
  //  symbolic_bindings: Used if the function is parametric
  //
  // Returns the value that results from interpretation.
  //
  // Note: doesn't establish the type information based on "f"'s module-level
  // type information; e.g. this can be used for parametric calls where the type
  // info is determined and set externally.
  absl::StatusOr<InterpValue> EvaluateAndCompareInternal(
      Function* f, absl::Span<const InterpValue> args, const Span& span,
      Invocation* invocation, const SymbolicBindings* symbolic_bindings);

  // Calls function values, either a builtin or user defined function.
  absl::StatusOr<InterpValue> CallFnValue(
      const InterpValue& fv, absl::Span<InterpValue const> args,
      const Span& span, Invocation* invocation,
      const SymbolicBindings* symbolic_bindings);

  absl::StatusOr<InterpValue> RunBuiltin(
      Builtin builtin, absl::Span<InterpValue const> args, const Span& span,
      Invocation* invocation, const SymbolicBindings* symbolic_bindings);

  // Helpers used for annotating work-in-progress constants on import, in case
  // of recursive calls in to the interpreter (e.g. when evaluating constant
  // expressions at the top of a module).
  bool IsWip(AstNode* node) const;

  // Notes that "node" is in work in progress state indicated by value: nullopt
  // means 'about to evaluate', a value means 'finished evaluatuing to this
  // value'. Returns the current state for 'node' (so we can check whether
  // 'node' had a cached result value already).
  //
  // The API was intended to be used as follows (minimizing the number of
  // callables required, see the todo below):
  //
  //    // Notes we're about to start computing top_node, and gives this API a
  //    // chance to say "oh hey you already did that".
  //    absl::optional<InterpValue> maybe_done = interp->NoteWip(
  //      top_node, absl::nullopt);
  //    if (maybe_done) { ... no need to compute ... }
  //    InterpValue really_done = ComputeIt();
  //    interp->NoteWip(top_node, really_done);  // Notes final computed value.
  //
  // TODO(leary): 2020-11-20 Rework this "note WIP" interface, it's too
  // overloaded in terms of what it's doing, was done this way to minimize the
  // number of callbacks passed across Python/C++ boundary but that should no
  // longer be a concern.
  absl::optional<InterpValue> NoteWip(AstNode* node,
                                      absl::optional<InterpValue> value);

  Module* const entry_module_;

  // Note that the "current" type info changes over time as we execute:
  // sometimes it will be the "root" type info for a module we're currently
  // executing inside, sometimes it will be a "derived" type info when we're
  // executing inside of a parametric context. The interpreter guts use
  // TypeInfoSwap to set up / tear down the appropriate current_type_info_ as
  // execution occurs.
  TypeInfo* current_type_info_;

  PostFnEvalHook post_fn_eval_hook_;
  TypecheckFn typecheck_;
  std::vector<std::filesystem::path> additional_search_paths_;
  ImportData* import_data_;
  bool trace_all_;
  FormatPreference trace_format_preference_;

  std::unique_ptr<AbstractInterpreter> abstract_adapter_;

  // Tracking for incomplete module evaluation status; e.g. on recursive calls
  // during module import; see IsWip().
  absl::flat_hash_map<AstNode*, absl::optional<InterpValue>> wip_;
};

// Converts the values to matched the signedness of the concrete type.
//
// Converts bits-typed Values contained within the given Value to match the
// signedness of the ConcreteType. Examples:
//
// invocation: sign_convert_value(s8, u8:64)
// returns: s8:64
//
// invocation: sign_convert_value(s3, u8:7)
// returns: s3:-1
//
// invocation: sign_convert_value((s8, u8), (u8:42, u8:10))
// returns: (s8:42, u8:10)
//
// This conversion functionality is required because the Values used in the DSLX
// may be signed while Values in IR interpretation and Verilog simulation are
// always unsigned.
//
// Args:
//   concrete_type: ConcreteType to match.
//   value: Input value.
//
// Returns:
//   Sign-converted value.
absl::StatusOr<InterpValue> SignConvertValue(const ConcreteType& concrete_type,
                                             const InterpValue& value);

// As above, but a handy vectorized form for application on parameters of a
// function.
absl::StatusOr<std::vector<InterpValue>> SignConvertArgs(
    const FunctionType& fn_type, absl::Span<const InterpValue> args);

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERPRETER_H_
