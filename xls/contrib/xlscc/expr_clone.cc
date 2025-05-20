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

#include <clang/AST/DeclCXX.h>

#include <vector>

#include "absl/strings/str_cat.h"
#include "clang/AST/StmtVisitor.h"

namespace xlscc {
namespace {

#define VISIT(lhs, rhs)       \
  clang::Expr* lhs = nullptr; \
  do {                        \
    auto result = Visit(rhs); \
    if (!result.ok()) {       \
      return result;          \
    }                         \
    lhs = *result;            \
  } while (false)

class ExprClone
    : public clang::ConstStmtVisitor<ExprClone, absl::StatusOr<clang::Expr*>> {
 public:
  explicit ExprClone(clang::ASTContext& ctx_) : ctx(ctx_) {}

  absl::StatusOr<clang::Expr*> VisitIntegerLiteral(
      const clang::IntegerLiteral* expr) {
    return clang::IntegerLiteral::Create(ctx, expr->getValue(), expr->getType(),
                                         expr->getLocation());
  }

  absl::StatusOr<clang::Expr*> VisitCharacterLiteral(
      const clang::CharacterLiteral* expr) {
    return new (ctx)
        clang::CharacterLiteral(expr->getValue(), expr->getKind(),
                                expr->getType(), expr->getLocation());
  }

  absl::StatusOr<clang::Expr*> VisitFloatingLiteral(
      const clang::FloatingLiteral* expr) {
    return clang::FloatingLiteral::Create(ctx, expr->getValue(),
                                          expr->isExact(), expr->getType(),
                                          expr->getLocation());
  }

  absl::StatusOr<clang::Expr*> VisitStringLiteral(
      const clang::StringLiteral* expr) {
    return clang::StringLiteral::Create(ctx, expr->getString(), expr->getKind(),
                                        expr->isPascal(), expr->getType(),
                                        expr->getExprLoc());
  }

  absl::StatusOr<clang::Expr*> VisitUserDefinedLiteral(
      const clang::UserDefinedLiteral* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      VISIT(cloned_arg, arg);
      args.push_back(cloned_arg);
    }
    VISIT(callee, expr->getCallee());
    return clang::UserDefinedLiteral::Create(
        ctx, callee, args, expr->getType(), expr->getValueKind(),
        expr->getRParenLoc(), expr->getUDSuffixLoc(), expr->getFPFeatures());
  }

  absl::StatusOr<clang::Expr*> VisitCompoundLiteralExpr(
      const clang::CompoundLiteralExpr* expr) {
    VISIT(init, expr->getInitializer());
    return new (ctx) clang::CompoundLiteralExpr(
        expr->getLParenLoc(), expr->getTypeSourceInfo(), expr->getType(),
        expr->getValueKind(), init, expr->getObjectKind());
  }

  absl::StatusOr<clang::Expr*> VisitDeclRefExpr(
      const clang::DeclRefExpr* expr) {
    clang::TemplateArgumentListInfo template_args;
    expr->copyTemplateArgumentsInto(template_args);
    // `Create()` requires non-const decl pointers.
    auto* decl = const_cast<clang::ValueDecl*>(expr->getDecl());
    auto* found_decl = const_cast<clang::NamedDecl*>(expr->getFoundDecl());
    // Ultimately, `Clone()` returns a const pointer, so its users cannot break
    // const correctness.
    return clang::DeclRefExpr::Create(
        ctx, expr->getQualifierLoc(), expr->getTemplateKeywordLoc(), decl,
        expr->refersToEnclosingVariableOrCapture(), expr->getNameInfo(),
        expr->getType(), expr->getValueKind(), found_decl, &template_args,
        expr->isNonOdrUse());
  }

  absl::StatusOr<clang::Expr*> VisitImplicitCastExpr(
      const clang::ImplicitCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::ImplicitCastExpr::Create(
        ctx, expr->getType(), expr->getCastKind(), sub_expr, nullptr,
        expr->getValueKind(), expr->getFPFeatures());
  }

  absl::StatusOr<clang::Expr*> VisitCStyleCastExpr(
      const clang::CStyleCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CStyleCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        sub_expr, nullptr, expr->getFPFeatures(), expr->getTypeInfoAsWritten(),
        expr->getLParenLoc(), expr->getRParenLoc());
  }

  absl::StatusOr<clang::Expr*> VisitCXXFunctionalCastExpr(
      const clang::CXXFunctionalCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CXXFunctionalCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(),
        expr->getTypeInfoAsWritten(), expr->getCastKind(), sub_expr, nullptr,
        expr->getFPFeatures(), expr->getLParenLoc(), expr->getRParenLoc());
  }

  absl::StatusOr<clang::Expr*> VisitCXXStaticCastExpr(
      const clang::CXXStaticCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CXXStaticCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        sub_expr, nullptr, expr->getTypeInfoAsWritten(), expr->getFPFeatures(),
        expr->getOperatorLoc(), expr->getRParenLoc(), expr->getAngleBrackets());
  }

  absl::StatusOr<clang::Expr*> VisitCXXDynamicCastExpr(
      const clang::CXXDynamicCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CXXDynamicCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        sub_expr, nullptr, expr->getTypeInfoAsWritten(), expr->getOperatorLoc(),
        expr->getRParenLoc(), expr->getAngleBrackets());
  }

  absl::StatusOr<clang::Expr*> VisitCXXReinterpretCastExpr(
      const clang::CXXReinterpretCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CXXReinterpretCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), expr->getCastKind(),
        sub_expr, nullptr, expr->getTypeInfoAsWritten(), expr->getOperatorLoc(),
        expr->getRParenLoc(), expr->getAngleBrackets());
  }

  absl::StatusOr<clang::Expr*> VisitCXXConstCastExpr(
      const clang::CXXConstCastExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::CXXConstCastExpr::Create(
        ctx, expr->getType(), expr->getValueKind(), sub_expr,
        expr->getTypeInfoAsWritten(), expr->getOperatorLoc(),
        expr->getRParenLoc(), expr->getAngleBrackets());
  }

  absl::StatusOr<clang::Expr*> VisitMemberExpr(const clang::MemberExpr* expr) {
    clang::TemplateArgumentListInfo template_args;
    VISIT(base, expr->getBase());
    return clang::MemberExpr::Create(
        ctx, base, expr->isArrow(), expr->getOperatorLoc(),
        expr->getQualifierLoc(), expr->getTemplateKeywordLoc(),
        expr->getMemberDecl(), expr->getFoundDecl(), expr->getMemberNameInfo(),
        &template_args, expr->getType(), expr->getValueKind(),
        expr->getObjectKind(), expr->isNonOdrUse());
  }

  absl::StatusOr<clang::Expr*> VisitCallExpr(const clang::CallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      VISIT(cloned_arg, arg);
      args.push_back(cloned_arg);
    }
    VISIT(callee, expr->getCallee());
    return clang::CallExpr::Create(ctx, callee, args, expr->getType(),
                                   expr->getValueKind(), expr->getRParenLoc(),
                                   expr->getFPFeatures(), expr->getNumArgs(),
                                   expr->getADLCallKind());
  }

  absl::StatusOr<clang::Expr*> VisitUnaryOperator(
      const clang::UnaryOperator* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::UnaryOperator::Create(
        ctx, sub_expr, expr->getOpcode(), expr->getType(), expr->getValueKind(),
        expr->getObjectKind(), expr->getOperatorLoc(), expr->canOverflow(),
        expr->getFPOptionsOverride());
  }

  absl::StatusOr<clang::Expr*> VisitBinaryOperator(
      const clang::BinaryOperator* expr) {
    VISIT(lhs, expr->getLHS());
    VISIT(rhs, expr->getRHS());
    return clang::BinaryOperator::Create(
        ctx, lhs, rhs, expr->getOpcode(), expr->getType(), expr->getValueKind(),
        expr->getObjectKind(), expr->getOperatorLoc(), expr->getFPFeatures());
  }

  absl::StatusOr<clang::Expr*> VisitConditionalOperator(
      const clang::ConditionalOperator* expr) {
    VISIT(cond, expr->getCond());
    VISIT(lhs, expr->getLHS());
    VISIT(rhs, expr->getRHS());
    return new (ctx) clang::ConditionalOperator(
        cond, expr->getQuestionLoc(), lhs, expr->getColonLoc(), rhs,
        expr->getType(), expr->getValueKind(), expr->getObjectKind());
  }

  absl::StatusOr<clang::Expr*> VisitParenExpr(const clang::ParenExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return new (ctx)
        clang::ParenExpr(expr->getBeginLoc(), expr->getEndLoc(), sub_expr);
  }

  absl::StatusOr<clang::Expr*> VisitArraySubscriptExpr(
      const clang::ArraySubscriptExpr* expr) {
    VISIT(lhs, expr->getLHS());
    VISIT(rhs, expr->getRHS());
    return new (ctx) clang::ArraySubscriptExpr(
        lhs, rhs, expr->getType(), expr->getValueKind(), expr->getObjectKind(),
        expr->getRBracketLoc());
  }

  absl::StatusOr<clang::Expr*> VisitInitListExpr(
      const clang::InitListExpr* expr) {
    std::vector<clang::Expr*> inits;
    for (auto* init : expr->inits()) {
      VISIT(cloned_init, init);
      inits.push_back(cloned_init);
    }
    auto* cloned = new (ctx) clang::InitListExpr(ctx, expr->getLBraceLoc(),
                                                 inits, expr->getRBraceLoc());
    cloned->setType(expr->getType());
    return cloned;
  }

  absl::StatusOr<clang::Expr*> VisitCXXOperatorCallExpr(
      const clang::CXXOperatorCallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      VISIT(cloned_arg, arg);
      args.push_back(cloned_arg);
    }
    VISIT(callee, expr->getCallee());
    return clang::CXXOperatorCallExpr::Create(
        ctx, expr->getOperator(), callee, args, expr->getType(),
        expr->getValueKind(), expr->getRParenLoc(), expr->getFPFeatures());
  }

  absl::StatusOr<clang::Expr*> VisitCXXMemberCallExpr(
      const clang::CXXMemberCallExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      VISIT(cloned_arg, arg);
      args.push_back(cloned_arg);
    }
    VISIT(callee, expr->getCallee());
    return clang::CXXMemberCallExpr::Create(
        ctx, callee, args, expr->getType(), expr->getValueKind(),
        expr->getRParenLoc(), expr->getFPFeatures());
  }

  absl::StatusOr<clang::Expr*> VisitCXXTemporaryObjectExpr(
      const clang::CXXTemporaryObjectExpr* expr) {
    std::vector<clang::Expr*> args;
    for (auto* arg : expr->arguments()) {
      VISIT(cloned_arg, arg);
      args.push_back(cloned_arg);
    }
    return clang::CXXTemporaryObjectExpr::Create(
        ctx, expr->getConstructor(), expr->getType(), expr->getTypeSourceInfo(),
        args, expr->getParenOrBraceRange(), expr->hadMultipleCandidates(),
        expr->isListInitialization(), expr->isStdInitListInitialization(),
        expr->requiresZeroInitialization());
  }

  absl::StatusOr<clang::Expr*> VisitMaterializeTemporaryExpr(
      const clang::MaterializeTemporaryExpr* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    // Constructor requires non-const decl pointers.
    auto* decl = const_cast<clang::LifetimeExtendedTemporaryDecl*>(
        expr->getLifetimeExtendedTemporaryDecl());
    // `Clone()` returns a const pointer, preserving const correctness.
    return new (ctx) clang::MaterializeTemporaryExpr(
        expr->getType(), sub_expr, expr->isBoundToLvalueReference(), decl);
  }

  absl::StatusOr<clang::Expr*> VisitExprWithCleanups(
      const clang::ExprWithCleanups* expr) {
    VISIT(sub_expr, expr->getSubExpr());
    return clang::ExprWithCleanups::Create(
        ctx, sub_expr, expr->cleanupsHaveSideEffects(), expr->getObjects());
  }

  absl::StatusOr<clang::Expr*> VisitExpr(const clang::Expr* expr) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported: clone ", expr->getStmtClassName(), " at ",
                     expr->getExprLoc().printToString(ctx.getSourceManager())));
  }

 private:
  clang::ASTContext& ctx;
};

}  // namespace

absl::StatusOr<const clang::Expr*> Clone(clang::ASTContext& ctx,
                                         const clang::Expr* expr) {
  ExprClone cloner(ctx);
  return cloner.Visit(expr);
}

}  // namespace xlscc
