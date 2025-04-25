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

#include <clang/AST/Expr.h>

#include <vector>

#include "clang/AST/StmtVisitor.h"

namespace xlscc {
namespace {
using namespace clang;

class ExprClone : public StmtVisitor<ExprClone, Expr*> {
 public:
  explicit ExprClone(ASTContext& ctx_) : ctx(ctx_) {}

  Expr* VisitIntegerLiteral(IntegerLiteral* expr) {
    return IntegerLiteral::Create(ctx, expr->getValue(), expr->getType(),
                                  expr->getLocation());
  }

  Expr* VisitCharacterLiteral(CharacterLiteral* expr) {
    return new (ctx) CharacterLiteral(expr->getValue(), expr->getKind(),
                                      expr->getType(), expr->getLocation());
  }

  Expr* VisitFloatingLiteral(FloatingLiteral* expr) {
    return FloatingLiteral::Create(ctx, expr->getValue(), expr->isExact(),
                                   expr->getType(), expr->getLocation());
  }

  Expr* VisitStringLiteral(StringLiteral* expr) {
    return StringLiteral::Create(ctx, expr->getString(), expr->getKind(),
                                 expr->isPascal(), expr->getType(),
                                 expr->getExprLoc());
  }

  Expr* VisitUserDefinedLiteral(UserDefinedLiteral* expr) {
    std::vector<Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return UserDefinedLiteral::Create(
        ctx, Visit(expr->getCallee()), args, expr->getType(),
        expr->getValueKind(), expr->getRParenLoc(), expr->getUDSuffixLoc(),
        expr->getFPFeatures());
  }

  Expr* VisitCompoundLiteralExpr(CompoundLiteralExpr* expr) {
    return new (ctx) CompoundLiteralExpr(
        expr->getLParenLoc(), expr->getTypeSourceInfo(), expr->getType(),
        expr->getValueKind(), Visit(expr->getInitializer()),
        expr->getObjectKind());
  }

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

  Expr* VisitCStyleCastExpr(CStyleCastExpr* expr) {
    return CStyleCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getFPFeatures(),
        expr->getTypeInfoAsWritten(), expr->getLParenLoc(),
        expr->getRParenLoc());
  }

  Expr* VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr* expr) {
    return CXXFunctionalCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(),
        expr->getTypeInfoAsWritten(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getFPFeatures(),
        expr->getLParenLoc(), expr->getRParenLoc());
  }

  Expr* VisitCXXStaticCastExpr(CXXStaticCastExpr* expr) {
    return CXXStaticCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getFPFeatures(), expr->getOperatorLoc(), expr->getRParenLoc(),
        expr->getAngleBrackets());
  }

  Expr* VisitCXXDynamicCastExpr(CXXDynamicCastExpr* expr) {
    return CXXDynamicCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getOperatorLoc(), expr->getRParenLoc(), expr->getAngleBrackets());
  }

  Expr* VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr* expr) {
    return CXXReinterpretCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getOperatorLoc(), expr->getRParenLoc(), expr->getAngleBrackets());
  }

  Expr* VisitCXXConstCastExpr(CXXConstCastExpr* expr) {
    return CXXConstCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), Visit(expr->getSubExpr()),
        expr->getTypeInfoAsWritten(), expr->getOperatorLoc(),
        expr->getRParenLoc(), expr->getAngleBrackets());
  }

  Expr* VisitMemberExpr(MemberExpr* expr) {
    TemplateArgumentListInfo template_args;
    if (expr->hasExplicitTemplateArgs()) {
      expr->copyTemplateArgumentsInto(template_args);
    }
    return MemberExpr::Create(
        ctx, Visit(expr->getBase()), expr->isArrow(), expr->getOperatorLoc(),
        expr->getQualifierLoc(), expr->getTemplateKeywordLoc(),
        expr->getMemberDecl(), expr->getFoundDecl(), expr->getMemberNameInfo(),
        &template_args, expr->getType(), expr->getValueKind(),
        expr->getObjectKind(), expr->isNonOdrUse());
  }

  Expr* VisitCallExpr(CallExpr* expr) {
    std::vector<Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return CallExpr::Create(ctx, Visit(expr->getCallee()), args,
                            expr->getType(), expr->getValueKind(),
                            expr->getRParenLoc(), expr->getFPFeatures(),
                            expr->getNumArgs(), expr->getADLCallKind());
  }

  Expr* VisitUnaryOperator(UnaryOperator* expr) {
    return UnaryOperator::Create(
        ctx, Visit(expr->getSubExpr()), expr->getOpcode(), expr->getType(),
        expr->getValueKind(), expr->getObjectKind(), expr->getOperatorLoc(),
        expr->canOverflow(), expr->getFPOptionsOverride());
  }

  Expr* VisitBinaryOperator(BinaryOperator* expr) {
    return BinaryOperator::Create(
        ctx, Visit(expr->getLHS()), Visit(expr->getRHS()), expr->getOpcode(),
        expr->getType(), expr->getValueKind(), expr->getObjectKind(),
        expr->getOperatorLoc(), expr->getFPFeatures());
  }

  Expr* VisitConditionalOperator(ConditionalOperator* expr) {
    return new (ctx) ConditionalOperator(
        Visit(expr->getCond()), expr->getQuestionLoc(), Visit(expr->getLHS()),
        expr->getColonLoc(), Visit(expr->getRHS()), expr->getType(),
        expr->getValueKind(), expr->getObjectKind());
  }

  Expr* VisitParenExpr(ParenExpr* expr) {
    return new (ctx) ParenExpr(expr->getBeginLoc(), expr->getEndLoc(),
                               Visit(expr->getSubExpr()));
  }

  Expr* VisitArraySubscriptExpr(ArraySubscriptExpr* expr) {
    return new (ctx) ArraySubscriptExpr(
        Visit(expr->getLHS()), Visit(expr->getRHS()), expr->getType(),
        expr->getValueKind(), expr->getObjectKind(), expr->getRBracketLoc());
  }

  Expr* VisitInitListExpr(InitListExpr* expr) {
    std::vector<Expr*> inits;
    for (auto* init : expr->inits()) {
      inits.push_back(Visit(init));
    }
    auto* cloned = new (ctx)
        InitListExpr(ctx, expr->getLBraceLoc(), inits, expr->getRBraceLoc());
    cloned->setType(expr->getType());
    return cloned;
  }

  Expr* VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr* expr) {
    return new (ctx) DesignatedInitUpdateExpr(
        ctx, expr->getBeginLoc(), Visit(expr->getBase()), expr->getEndLoc());
  }

  Expr* VisitCXXOperatorCallExpr(CXXOperatorCallExpr* expr) {
    std::vector<Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return CXXOperatorCallExpr::Create(
        ctx, expr->getOperator(), Visit(expr->getCallee()), args,
        expr->getType(), expr->getValueKind(), expr->getRParenLoc(),
        expr->getFPFeatures());
  }

  Expr* VisitCXXMemberCallExpr(CXXMemberCallExpr* expr) {
    std::vector<Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return CXXMemberCallExpr::Create(
        ctx, Visit(expr->getCallee()), args, expr->getType(),
        expr->getValueKind(), expr->getRParenLoc(), expr->getFPFeatures());
  }

  Expr* VisitCXXTemporaryObjectExpr(CXXTemporaryObjectExpr* expr) {
    std::vector<Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return CXXTemporaryObjectExpr::Create(
        ctx, expr->getConstructor(), expr->getType(), expr->getTypeSourceInfo(),
        args, expr->getParenOrBraceRange(), expr->hadMultipleCandidates(),
        expr->isListInitialization(), expr->isStdInitListInitialization(),
        expr->requiresZeroInitialization());
  }

 private:
  ASTContext& ctx;
};

}  // namespace

Expr* Clone(ASTContext& ctx, const Expr* expr) {
  ExprClone cloner(ctx);
  // FIXME: get rid of this const cast
  return cloner.Visit(const_cast<Expr*>(expr));
}

}  // namespace xlscc
