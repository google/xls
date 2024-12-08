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
#ifndef XLS_DSLX_CONSTEXPR_EVALUATOR_H_
#define XLS_DSLX_CONSTEXPR_EVALUATOR_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Simple visitor to perform automatic dispatch to constexpr evaluate AST
// expressions.
//
// Note that WarningCollector is given as a pointer and not a reference because
// it is optional; e.g. sometimes constexpr evaluation is performed during IR
// conversion, at which point we don't want to be collecting warnings since we
// expect to have flagged all warnings (e.g. rollovers) in the type checking
// phase.
class ConstexprEvaluator : public xls::dslx::ExprVisitor {
 public:
  // Evaluates the given expression to determine if it's constexpr or not, and
  // updates `type_info` accordingly. Returns success as long as no error
  // occurred during evaluation, even if `expr` is non-constexpr.
  static absl::Status Evaluate(ImportData* import_data, TypeInfo* type_info,
                               WarningCollector* warning_collector,
                               const ParametricEnv& bindings, const Expr* expr,
                               const Type* type = nullptr);

  // Performs the same action as `Evaluate`, but returns the resulting constexpr
  // value. Returns an error status if `expr` is non-constexpr.
  static absl::StatusOr<InterpValue> EvaluateToValue(
      ImportData* import_data, TypeInfo* type_info,
      WarningCollector* warning_collector, const ParametricEnv& bindings,
      const Expr* expr, const Type* type = nullptr);

  // A concrete type is only necessary when:
  //  - Deducing a Number that is undecorated and whose type is specified by
  //    context, e.g., an element in a constant array:
  //    `u32[4]:[0, 1, 2, 3]`. It can be nullptr in all other circumstances.
  //  - Deducing a constant array whose declaration terminates in an ellipsis:
  //    `u32[4]:[0, 1, ...]`. The type is needed to determine the number of
  //    elements to fill in.
  // In all other cases, `type` can be nullptr.
  ~ConstexprEvaluator() override = default;

  absl::Status HandleArray(const Array* expr) override;
  absl::Status HandleAttr(const Attr* expr) override;
  absl::Status HandleBinop(const Binop* expr) override;
  absl::Status HandleStatementBlock(const StatementBlock* expr) override;
  absl::Status HandleCast(const Cast* expr) override;
  absl::Status HandleChannelDecl(const ChannelDecl* expr) override;
  absl::Status HandleColonRef(const ColonRef* expr) override;
  absl::Status HandleConstAssert(const ConstAssert* const_assert) override;
  absl::Status HandleConstRef(const ConstRef* expr) override;
  absl::Status HandleFor(const For* expr) override;
  absl::Status HandleFunctionRef(const FunctionRef* expr) override;
  absl::Status HandleFormatMacro(const FormatMacro* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleZeroMacro(const ZeroMacro* expr) override;
  absl::Status HandleAllOnesMacro(const AllOnesMacro* expr) override;
  absl::Status HandleIndex(const Index* expr) override;
  absl::Status HandleInvocation(const Invocation* expr) override;
  absl::Status HandleLet(const Let* expr) override { return absl::OkStatus(); }
  absl::Status HandleMatch(const Match* expr) override;
  absl::Status HandleNameRef(const NameRef* expr) override;
  absl::Status HandleNumber(const Number* expr) override;
  absl::Status HandleRange(const Range* expr) override;
  absl::Status HandleSpawn(const Spawn* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleString(const String* expr) override;
  absl::Status HandleStructInstance(const StructInstance* expr) override;
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* expr) override;
  absl::Status HandleConditional(const Conditional* expr) override;
  absl::Status HandleTupleIndex(const TupleIndex* expr) override;
  absl::Status HandleUnop(const Unop* expr) override;
  absl::Status HandleUnrollFor(const UnrollFor* expr) override;
  absl::Status HandleVerbatimNode(const VerbatimNode* node) override;
  absl::Status HandleXlsTuple(const XlsTuple* expr) override;

  static absl::StatusOr<InterpValue> CreateChannelValue(const Type* type);

 private:
  ConstexprEvaluator(ImportData* import_data, TypeInfo* type_info,
                     WarningCollector* warning_collector,
                     ParametricEnv bindings, const Type* type)
      : import_data_(import_data),
        type_info_(type_info),
        warning_collector_(warning_collector),
        bindings_(std::move(bindings)),
        type_(type) {}

  // Interprets the given expression. Prior to calling this function, it's
  // necessary to determine that all expression components are constexpr.
  absl::Status InterpretExpr(const Expr* expr);

  ImportData* const import_data_;
  TypeInfo* const type_info_;
  WarningCollector* const warning_collector_;
  const ParametricEnv bindings_;
  const Type* const type_;
};

// Holds the results of `MakeConstexprEnv()` -- generally users will use `env`,
// but `freevars` and `non_constexpr` provide useful information in cases where
// something goes wrong or needs to be reported via error messages.
struct ConstexprEnvData {
  // Free variables for the given expression.
  FreeVariables freevars;

  // Constexpr environment we were able to construct.
  //
  // Note that it seems ok for now for the keys to be std::string rather than
  // NameDef* because we only use this map in the context of a single
  // expression, and free variables from the expression for a given identifier
  // will all point to the same thing, there is no partial shadowing that can
  // occur.
  absl::flat_hash_map<std::string, InterpValue> env;

  // Free variable references that we were unable to resolve to a constexpr
  // value.
  //
  // Note that this can include things like references to imported modules,
  // as those are not themselves constexpr.
  absl::flat_hash_set<const NameRef*> non_constexpr;
};

// Creates a map of `{symbol_name: value}` for all known symbols in the required
// environment for `env`.
//
// This will be populated with parametric bindings as well as
// constexpr freevars of `node`, which is useful when there are local
// const bindings closed over e.g. in function scope.
//
// `type_info` is required to look up the value of previously computed
// constexprs.
absl::StatusOr<ConstexprEnvData> MakeConstexprEnv(
    ImportData* import_data, TypeInfo* type_info,
    WarningCollector* warning_collector, const Expr* node,
    const ParametricEnv& parametric_env);

// Returns a helpful text representation fo the given environment map, suitable
// for putting into error messages and similar, along the lines of:
//
//    {MOL: u32:42, TWO: u32:2}
//
// Note the returned string is deterministic.
std::string EnvMapToString(
    const absl::flat_hash_map<std::string, InterpValue>& map);

// Evaluates a Number AST node to an InterpValue.
absl::StatusOr<InterpValue> EvaluateNumber(const Number& expr,
                                           const Type& type);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONSTEXPR_EVALUATOR_H_
