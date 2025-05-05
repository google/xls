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

#include "absl/strings/str_cat.h"
#include "clang/AST/StmtVisitor.h"

namespace xlscc {
namespace {

class ExprClone : public clang::ConstStmtVisitor<ExprClone, clang::Expr*> {
 public:
  explicit ExprClone(clang::ASTContext& ctx_) : ctx(ctx_) {}

  clang::Expr* VisitIntegerLiteral(const clang::IntegerLiteral* expr) {
    return clang::IntegerLiteral::Create(ctx, expr->getValue(), expr->getType(),
                                         expr->getLocation());
  }

  clang::Expr* VisitCharacterLiteral(const clang::CharacterLiteral* expr) {
    return new (ctx)
        clang::CharacterLiteral(expr->getValue(), expr->getKind(),
                                expr->getType(), expr->getLocation());
  }

  clang::Expr* VisitFloatingLiteral(const clang::FloatingLiteral* expr) {
    return clang::FloatingLiteral::Create(ctx, expr->getValue(),
                                          expr->isExact(), expr->getType(),
                                          expr->getLocation());
  }

  clang::Expr* VisitStringLiteral(const clang::StringLiteral* expr) {
    return clang::StringLiteral::Create(ctx, expr->getString(), expr->getKind(),
                                        expr->isPascal(), expr->getType(),
                                        expr->getExprLoc());
  }

  clang::Expr* VisitUserDefinedLiteral(const clang::UserDefinedLiteral* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return clang::UserDefinedLiteral::Create(
        ctx, Visit(expr->getCallee()), args, expr->getType(),
        expr->getValueKind(), expr->getRParenLoc(), expr->getUDSuffixLoc(),
        expr->getFPFeatures());
  }

  clang::Expr* VisitCompoundLiteralExpr(
      const clang::CompoundLiteralExpr* expr) {
    return new (ctx) clang::CompoundLiteralExpr(
        expr->getLParenLoc(), expr->getTypeSourceInfo(), expr->getType(),
        expr->getValueKind(), Visit(expr->getInitializer()),
        expr->getObjectKind());
  }

  clang::Expr* VisitDeclRefExpr(const clang::DeclRefExpr* expr) {
    clang::TemplateArgumentListInfo template_args;
    expr->copyTemplateArgumentsInto(template_args);
    // `Create()` requires non-const decl pointers.
    auto* decl = const_cast<clang::ValueDecl*>(expr->getDecl());
    auto* found_decl = const_cast<clang::NamedDecl*>(expr->getFoundDecl());
    // Ultimately, `Clone()` return a const pointer, so its users cannot break
    // const correctness.
    return clang::DeclRefExpr::Create(
        ctx, expr->getQualifierLoc(), expr->getTemplateKeywordLoc(), decl,
        expr->refersToEnclosingVariableOrCapture(), expr->getNameInfo(),
        expr->getType(), expr->getValueKind(), found_decl, &template_args,
        expr->isNonOdrUse());
  }

  clang::Expr* VisitImplicitCastExpr(const clang::ImplicitCastExpr* expr) {
    return clang::ImplicitCastExpr::Create(
        ctx, expr->getType(), expr->getCastKind(), Visit(expr->getSubExpr()),
        nullptr, expr->getValueKind(), expr->getFPFeatures());
  }

  clang::Expr* VisitCStyleCastExpr(const clang::CStyleCastExpr* expr) {
    return clang::CStyleCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getFPFeatures(),
        expr->getTypeInfoAsWritten(), expr->getLParenLoc(),
        expr->getRParenLoc());
  }

  clang::Expr* VisitCXXFunctionalCastExpr(
      const clang::CXXFunctionalCastExpr* expr) {
    return clang::CXXFunctionalCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(),
        expr->getTypeInfoAsWritten(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getFPFeatures(),
        expr->getLParenLoc(), expr->getRParenLoc());
  }

  clang::Expr* VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* expr) {
    return clang::CXXStaticCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getFPFeatures(), expr->getOperatorLoc(), expr->getRParenLoc(),
        expr->getAngleBrackets());
  }

  clang::Expr* VisitCXXDynamicCastExpr(const clang::CXXDynamicCastExpr* expr) {
    return clang::CXXDynamicCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getOperatorLoc(), expr->getRParenLoc(), expr->getAngleBrackets());
  }

  clang::Expr* VisitCXXReinterpretCastExpr(
      const clang::CXXReinterpretCastExpr* expr) {
    return clang::CXXReinterpretCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        Visit(expr->getSubExpr()), nullptr, expr->getTypeInfoAsWritten(),
        expr->getOperatorLoc(), expr->getRParenLoc(), expr->getAngleBrackets());
  }

  clang::Expr* VisitCXXConstCastExpr(const clang::CXXConstCastExpr* expr) {
    return clang::CXXConstCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), Visit(expr->getSubExpr()),
        expr->getTypeInfoAsWritten(), expr->getOperatorLoc(),
        expr->getRParenLoc(), expr->getAngleBrackets());
  }

  clang::Expr* VisitMemberExpr(const clang::MemberExpr* expr) {
    clang::TemplateArgumentListInfo template_args;
    if (expr->hasExplicitTemplateArgs()) {
      expr->copyTemplateArgumentsInto(template_args);
    }
    return clang::MemberExpr::Create(
        ctx, Visit(expr->getBase()), expr->isArrow(), expr->getOperatorLoc(),
        expr->getQualifierLoc(), expr->getTemplateKeywordLoc(),
        expr->getMemberDecl(), expr->getFoundDecl(), expr->getMemberNameInfo(),
        &template_args, expr->getType(), expr->getValueKind(),
        expr->getObjectKind(), expr->isNonOdrUse());
  }

  clang::Expr* VisitCallExpr(const clang::CallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return clang::CallExpr::Create(ctx, Visit(expr->getCallee()), args,
                                   expr->getType(), expr->getValueKind(),
                                   expr->getRParenLoc(), expr->getFPFeatures(),
                                   expr->getNumArgs(), expr->getADLCallKind());
  }

  clang::Expr* VisitUnaryOperator(const clang::UnaryOperator* expr) {
    return clang::UnaryOperator::Create(
        ctx, Visit(expr->getSubExpr()), expr->getOpcode(), expr->getType(),
        expr->getValueKind(), expr->getObjectKind(), expr->getOperatorLoc(),
        expr->canOverflow(), expr->getFPOptionsOverride());
  }

  clang::Expr* VisitBinaryOperator(const clang::BinaryOperator* expr) {
    return clang::BinaryOperator::Create(
        ctx, Visit(expr->getLHS()), Visit(expr->getRHS()), expr->getOpcode(),
        expr->getType(), expr->getValueKind(), expr->getObjectKind(),
        expr->getOperatorLoc(), expr->getFPFeatures());
  }

  clang::Expr* VisitConditionalOperator(
      const clang::ConditionalOperator* expr) {
    return new (ctx) clang::ConditionalOperator(
        Visit(expr->getCond()), expr->getQuestionLoc(), Visit(expr->getLHS()),
        expr->getColonLoc(), Visit(expr->getRHS()), expr->getType(),
        expr->getValueKind(), expr->getObjectKind());
  }

  clang::Expr* VisitParenExpr(const clang::ParenExpr* expr) {
    return new (ctx) clang::ParenExpr(expr->getBeginLoc(), expr->getEndLoc(),
                                      Visit(expr->getSubExpr()));
  }

  clang::Expr* VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* expr) {
    return new (ctx) clang::ArraySubscriptExpr(
        Visit(expr->getLHS()), Visit(expr->getRHS()), expr->getType(),
        expr->getValueKind(), expr->getObjectKind(), expr->getRBracketLoc());
  }

  clang::Expr* VisitInitListExpr(const clang::InitListExpr* expr) {
    std::vector<clang::Expr*> inits;
    for (auto* init : expr->inits()) {
      inits.push_back(Visit(init));
    }
    auto* cloned = new (ctx) clang::InitListExpr(ctx, expr->getLBraceLoc(),
                                                 inits, expr->getRBraceLoc());
    cloned->setType(expr->getType());
    return cloned;
  }

  clang::Expr* VisitCXXOperatorCallExpr(
      const clang::CXXOperatorCallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return clang::CXXOperatorCallExpr::Create(
        ctx, expr->getOperator(), Visit(expr->getCallee()), args,
        expr->getType(), expr->getValueKind(), expr->getRParenLoc(),
        expr->getFPFeatures());
  }

  clang::Expr* VisitCXXMemberCallExpr(const clang::CXXMemberCallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return clang::CXXMemberCallExpr::Create(
        ctx, Visit(expr->getCallee()), args, expr->getType(),
        expr->getValueKind(), expr->getRParenLoc(), expr->getFPFeatures());
  }

  clang::Expr* VisitCXXTemporaryObjectExpr(
      const clang::CXXTemporaryObjectExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      args.push_back(Visit(arg));
    }
    return clang::CXXTemporaryObjectExpr::Create(
        ctx, expr->getConstructor(), expr->getType(), expr->getTypeSourceInfo(),
        args, expr->getParenOrBraceRange(), expr->hadMultipleCandidates(),
        expr->isListInitialization(), expr->isStdInitListInitialization(),
        expr->requiresZeroInitialization());
  }

  clang::Expr* VisitExpr(const clang::Expr* expr) { return nullptr; }

 private:
  clang::ASTContext& ctx;
};

}  // namespace

absl::StatusOr<const clang::Expr*> Clone(clang::ASTContext& ctx,
                                         const clang::Expr* expr) {
  ExprClone cloner(ctx);
  auto cloned = cloner.Visit(expr);
  if (cloned) {
    return cloned;
  }
  return absl::UnimplementedError(
      absl::StrCat("Unsupported: clone ", expr->getStmtClassName()));
}

}  // namespace xlscc
