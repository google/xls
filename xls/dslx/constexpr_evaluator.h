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

#include "absl/status/status.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {

// Simple visitor to perform automatic dispatch to constexpr evaluate AST
// expressions.
class ConstexprEvaluator : public xls::dslx::ExprVisitor {
 public:
  // Evaluates the given expression to determine if it's constexpr or not, and
  // updates `type_info` accordingly. Returns success as long as no error
  // occurred during evaluation, even if `expr` is non-constexpr.
  static absl::Status Evaluate(ImportData* import_data, TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               const Expr* expr,
                               const ConcreteType* concrete_type = nullptr);

  // Performs the same action as `Evaluate`, but returns the resulting constexpr
  // value. Returns an error status if `expr` is non-constexpr.
  static absl::StatusOr<InterpValue> EvaluateToValue(
      ImportData* import_data, TypeInfo* type_info,
      const SymbolicBindings& bindings, const Expr* expr,
      const ConcreteType* concrete_type = nullptr);

  // A concrete type is only necessary when:
  //  - Deducing a Number that is undecorated and whose type is specified by
  //    context, e.g., an element in a constant array:
  //    `u32[4]:[0, 1, 2, 3]`. It can be nullptr in all other circumstances.
  //  - Deducing a constant array whose declaration terminates in an ellipsis:
  //    `u32[4]:[0, 1, ...]`. The type is needed to determine the number of
  //    elements to fill in.
  // In all other cases, `concrete_type` can be nullptr.
  ~ConstexprEvaluator() override {}

  absl::Status HandleArray(const Array* expr) override;
  absl::Status HandleAttr(const Attr* expr) override;
  absl::Status HandleBinop(const Binop* expr) override;
  absl::Status HandleBlock(const Block* expr) override;
  absl::Status HandleCast(const Cast* expr) override;
  absl::Status HandleChannelDecl(const ChannelDecl* expr) override;
  absl::Status HandleColonRef(const ColonRef* expr) override;
  absl::Status HandleConstantArray(const ConstantArray* expr) override;
  absl::Status HandleConstRef(const ConstRef* expr) override;
  absl::Status HandleFor(const For* expr) override;
  absl::Status HandleFormatMacro(const FormatMacro* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleIndex(const Index* expr) override;
  absl::Status HandleInvocation(const Invocation* expr) override;
  absl::Status HandleJoin(const Join* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleLet(const Let* expr) override { return absl::OkStatus(); }
  absl::Status HandleMatch(const Match* expr) override;
  absl::Status HandleNameRef(const NameRef* expr) override;
  absl::Status HandleNumber(const Number* expr) override;
  absl::Status HandleRange(const Range* expr) override;
  absl::Status HandleRecv(const Recv* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvIf(const RecvIf* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvIfNonBlocking(const RecvIfNonBlocking* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvNonBlocking(const RecvNonBlocking* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSend(const Send* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSendIf(const SendIf* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleSpawn(const Spawn* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleString(const String* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleStructInstance(const StructInstance* expr) override;
  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* expr) override;
  absl::Status HandleTernary(const Ternary* expr) override;
  absl::Status HandleTupleIndex(const TupleIndex* expr) override;
  absl::Status HandleUnop(const Unop* expr) override;
  absl::Status HandleUnrollFor(const UnrollFor* expr) override;
  absl::Status HandleXlsTuple(const XlsTuple* expr) override;

  static absl::StatusOr<InterpValue> CreateChannelValue(
      const ConcreteType* concrete_type);

 private:
  ConstexprEvaluator(ImportData* import_data, TypeInfo* type_info,
                     SymbolicBindings bindings,
                     const ConcreteType* concrete_type)
      : import_data_(import_data),
        type_info_(type_info),
        bindings_(bindings),
        concrete_type_(concrete_type) {}

  bool IsConstExpr(const Expr* expr);

  // Interprets the given expression. Prior to calling this function, it's
  // necessary to determine that all expression components are constexpr.
  // `bypass_env` is a set of NameDefs to skip when constructing the constexpr
  // env. This is needed for `for` loop evaluation, in cases where a loop-scoped
  // variable shadows the initial value. Constexpr env creation is string value
  // indexed, so this is needed so we can "skip" loop-declared variables.
  absl::Status InterpretExpr(
      const Expr* expr, absl::flat_hash_set<const NameDef*> bypass_env = {});

  ImportData* import_data_;
  TypeInfo* type_info_;
  SymbolicBindings bindings_;
  const ConcreteType* concrete_type_;
};

// Creates a map of symbol name to value for all known symbols in the current
// environment. This will be populated with symbolic bindings as well as
// constexpr freevars of "node", which is useful when there are local
// const bindings closed over e.g. in function scope.
//
// `type_info` is required to look up the value of previously computed
// constexprs.
// `bypass_env` is a set of NameDefs to skip when constructing the constexpr
// env. This is needed for `for` loop constexpr eval, in cases where a
// loop-scoped variable shadows the initial value, to be able to resolve the
// outer [constexpr] value.
//
// TODO(rspringer): 2022-05-29: `bypass_env` seems pretty nonintuitive and
// fragile; could there be another way of ignoring for loop-declared NameDefs?
// AIUI, they're the only reason this is needed.
absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>> MakeConstexprEnv(
    ImportData* import_data, TypeInfo* type_info, const Expr* node,
    const SymbolicBindings& symbolic_bindings,
    absl::flat_hash_set<const NameDef*> bypass_env = {});

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONSTEXPR_EVALUATOR_H_
