// Copyright 2025 The XLS Authors
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

#include "xls/contrib/xlscc/expr_clone.h"

#include <vector>

#include "clang/AST/StmtVisitor.h"

namespace xlscc {
namespace {
using namespace clang;

class ExprClone : public StmtVisitor<ExprClone, Expr*> {
 public:
  explicit ExprClone(ASTContext& ctx_) : ctx(ctx_) {}

  Expr* VisitDeclRefExpr(DeclRefExpr* expr) {
    TemplateArgumentListInfo template_args;
    expr->copyTemplateArgumentsInto(template_args);
    return DeclRefExpr::Create(
        ctx, expr->getQualifierLoc(), expr->getTemplateKeywordLoc(),
        expr->getDecl(), expr->refersToEnclosingVariableOrCapture(),
        expr->getNameInfo(), expr->getType(), expr->getValueKind(),
        expr->getFoundDecl(), &template_args, expr->isNonOdrUse());
  }

  Expr* VisitImplicitCastExpr(ImplicitCastExpr* expr) {
    return ImplicitCastExpr::Create(
        ctx, expr->getType(), expr->getCastKind(), Visit(expr->getSubExpr()),
        nullptr, expr->getValueKind(), expr->getFPFeatures());
  }

  Expr* VisitCallExpr(CallExpr* expr) {
    std::vector<Expr*> args;
    for (Expr* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return CallExpr::Create(ctx, Visit(expr->getCallee()), args,
                            expr->getType(), expr->getValueKind(),
                            expr->getRParenLoc(), expr->getFPFeatures(),
                            expr->getNumArgs(), expr->getADLCallKind());
  }

 private:
  ASTContext& ctx;
};

}  // namespace

Expr* Clone(ASTContext& ctx, const Expr* expr) {
  ExprClone cloner(ctx);
  llvm::outs() << "=> Original:\n";
  expr->dump(llvm::outs(), ctx);
  // FIXME: get rid of this const cast
  auto* cloned = cloner.Visit(const_cast<Expr*>(expr));
  llvm::outs() << "=> Cloned:\n";
  cloned->dump(llvm::outs(), ctx);
  return cloned;
}

}  // namespace xlscc
