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
    expr->AcceptExpr(&evaluator);
    return evaluator.status();
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

  void HandleArray(const Array* expr) override;
  void HandleAttr(const Attr* expr) override;
  void HandleBinop(const Binop* expr) override;
  void HandleCast(const Cast* expr) override;
  void HandleChannelDecl(const ChannelDecl* expr) override;
  void HandleColonRef(const ColonRef* expr) override;
  void HandleConstRef(const ConstRef* expr) override;
  void HandleFor(const For* expr) override;
  void HandleFormatMacro(const FormatMacro* expr) override {}
  void HandleIndex(const Index* expr) override;
  void HandleInvocation(const Invocation* expr) override;
  void HandleJoin(const Join* expr) override {}
  void HandleLet(const Let* expr) override {}
  void HandleMatch(const Match* expr) override;
  void HandleNameRef(const NameRef* expr) override;
  void HandleNumber(const Number* expr) override;
  void HandleRecv(const Recv* expr) override {}
  void HandleRecvIf(const RecvIf* expr) override {}
  void HandleSend(const Send* expr) override {}
  void HandleSendIf(const SendIf* expr) override {}
  void HandleSpawn(const Spawn* expr) override {}
  void HandleString(const String* expr) override {}
  void HandleStructInstance(const StructInstance* expr) override;
  void HandleSplatStructInstance(const SplatStructInstance* expr) override;
  void HandleTernary(const Ternary* expr) override;
  void HandleUnop(const Unop* expr) override;
  void HandleXlsTuple(const XlsTuple* expr) override;

  absl::Status status() { return status_; }

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
  absl::Status status_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONSTEXPR_EVALUATOR_H_
