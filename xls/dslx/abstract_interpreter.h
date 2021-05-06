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

#ifndef XLS_DSLX_ABSTRACT_INTERPRETER_H_
#define XLS_DSLX_ABSTRACT_INTERPRETER_H_

#include "absl/status/statusor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_bindings.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Abstract interface for interpreter facilities that is not coupled to other
// translation units the interpreter needs for implementation; i.e. this helps
// break circular dependencies.
class AbstractInterpreter {
 public:
  virtual ~AbstractInterpreter() = default;

  // Used to evaluate an expression (e.g. a constant at the top level).
  virtual absl::StatusOr<InterpValue> Eval(
      Expr* expr, InterpBindings* bindings,
      std::unique_ptr<ConcreteType> type_context = nullptr) = 0;

  // Calls function values, either builtin or user-defined function.
  virtual absl::StatusOr<InterpValue> CallValue(
      const InterpValue& value, absl::Span<const InterpValue> args,
      const Span& invocation_span, Invocation* invocation,
      const SymbolicBindings* sym_bindings) = 0;

  // Returns a typecheck lambda analogous to the DoTypecheck() call above.
  virtual TypecheckFn GetTypecheckFn() = 0;

  // Determines if a node (at the module scope) is in the process of being
  // evaluated -- this lets us detect re-entry (i.e.  a top level constant that
  // wants our top-level bindings to do the evaluation needs to make forward
  // progress using definitions previous to it in the file).
  virtual bool IsWip(AstNode* node) = 0;

  // Notes that a constant evaluation is "work in progress"
  // (underway) -- this is noted by passing nullopt before a call to evaluate
  // it. Once the constant is fully evaluated, the callback will be invoked with
  // the given value. If this callback returns a non-nullopt value, the constant
  // had already been evaluated (and was cached).
  virtual absl::optional<InterpValue> NoteWip(
      AstNode* node, absl::optional<InterpValue> value) = 0;

  // Retrieves the current type information being used (from interpreter state).
  virtual TypeInfo* GetCurrentTypeInfo() = 0;

  virtual ImportData* GetImportData() = 0;

  // Retrieves the format preference to use in `trace!()` builtin execution.
  virtual FormatPreference GetTraceFormatPreference() const = 0;

  // Law-of-Demeter helper for getting the root type info for a module (via the
  // import cache).
  absl::StatusOr<TypeInfo*> GetRootTypeInfo(Module* module) {
    return GetImportData()->GetRootTypeInfo(module);
  }

  // Returns the additional search paths to use on import.
  virtual absl::Span<const std::filesystem::path>
  GetAdditionalSearchPaths() = 0;

  // RAII type that can be use to swap the type information on an
  // AbstractInterpreter to a new value within a given lifetime.
  class ScopedTypeInfoSwap {
   public:
    ScopedTypeInfoSwap(AbstractInterpreter* interp, TypeInfo* updated);

    // Helper overload that uses the root TypeInfo associated with "module".
    //
    // Note: Check-fails if the module does not have TypeInfo in the import
    // cache.
    ScopedTypeInfoSwap(AbstractInterpreter* interp, Module* module);

    // Helper overload that uses the module associated with "node" to determine
    // the TypeInfo to establish, see Module overload above.
    ScopedTypeInfoSwap(AbstractInterpreter* interp, AstNode* node);

    ~ScopedTypeInfoSwap();

   private:
    AbstractInterpreter* interp;
    TypeInfo* original;
  };

 protected:
  friend class ScopedTypeInfoSwap;

  // Sets the current type info on the interpreter -- only intended for user use
  // via ScopedTypeInfoSwap so kept protected with ScopedTypeInfoSwap friended.
  virtual void SetCurrentTypeInfo(TypeInfo* current_type_info) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_ABSTRACT_INTERPRETER_H_
