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
#include "xls/dslx/deduce_ctx.h"

namespace xls::dslx {

// Simple visitor to perform automatic dispatch to constexpr evaluate AST
// expressions.
// TODO(rspringer): 2021-10-15, issue #508: Not all expression nodes are
// currently covered, but will need to be shortly.
class ConstexprEvaluator : public xls::dslx::ExprVisitor {
 public:
  static absl::Status Evaluate(DeduceCtx* ctx, const Expr* expr,
                               const ConcreteType* concrete_type = nullptr) {
    ConstexprEvaluator evaluator(ctx, concrete_type);
    return expr->AcceptExpr(&evaluator);
  }

  // A concrete type is only necessary when:
  //  - Deducing a Number that is undecorated and whose type is specified by
  //    context, e.g., an element in a constant array:
  //    `u32[4]:[0, 1, 2, 3]`. It can be nullptr in all other circumstances.
  //  - Deducing a constant array whose declaration terminates in an ellipsis:
  //    `u32[4]:[0, 1, ...]`. The type is needed to determine the number of
  //    elements to fill in.
  // In all other cases, `concrete_type` can be nullptr.
  ConstexprEvaluator(DeduceCtx* ctx, const ConcreteType* concrete_type)
      : ctx_(ctx), concrete_type_(concrete_type) {}
  ~ConstexprEvaluator() override {}

  absl::Status HandleArray(const Array* expr) override;
  absl::Status HandleAttr(const Attr* expr) override;
  absl::Status HandleBinop(const Binop* expr) override;
  absl::Status HandleCast(const Cast* expr) override;
  absl::Status HandleChannelDecl(const ChannelDecl* expr) override;
  absl::Status HandleColonRef(const ColonRef* expr) override;
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
  absl::Status HandleRecv(const Recv* expr) override {
    return absl::OkStatus();
  }
  absl::Status HandleRecvIf(const RecvIf* expr) override {
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
  absl::Status HandleUnop(const Unop* expr) override;
  absl::Status HandleXlsTuple(const XlsTuple* expr) override;

 private:
  bool IsConstExpr(const Expr* expr);

  // Interprets the given expression. Prior to calling this function, it's
  // necessary to determine that all expression components are constexpr.
  // `bypass_env` is a set of NameDefs to skip when constructing the constexpr
  // env. This is needed for `for` loop evaluation, in cases where a loop-scoped
  // variable shadows the initial value. Constexpr env creation is string value
  // indexed, so this is needed so we can "skip" loop-declared variables.
  absl::Status InterpretExpr(
      const Expr* expr, absl::flat_hash_set<const NameDef*> bypass_env = {});

  DeduceCtx* ctx_;
  const ConcreteType* concrete_type_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONSTEXPR_EVALUATOR_H_
