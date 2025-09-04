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

#include "xls/dslx/frontend/semantics_analysis.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

namespace {
// Warns if the next-to-last statement in a block has a trailing semi and the
// last statement is a nil tuple expression, as this is redundant; i.e.
//
//    {
//      foo;
//      ()  <-- useless, semi on previous statement implies it
//    }
void DetectUselessTrailingTuplePattern(const StatementBlock* block,
                                       WarningCollector& warning_collector) {
  // TODO(https://github.com/google/xls/issues/1124) 2023-08-31 Proc config
  // parsing functions synthesize a tuple at the end, and we don't want to flag
  // that since the user didn't even create it.
  if (block->parent()->kind() == AstNodeKind::kFunction &&
      dynamic_cast<const Function*>(block->parent())->tag() ==
          FunctionTag::kProcConfig) {
    return;
  }

  // Need at least a statement (i.e. with semicolon after it) and an
  // expression-statement at the end to match this pattern.
  if (block->statements().size() < 2) {
    return;
  }

  // Make sure we ignore this if we're only following an implicit prologue (as
  // is used to convert implicit-token-parameter semantics for now).
  // TODO(https://github.com/google/xls/issues/1401): Remove once we no longer
  // support implicit token parameter semantics.
  const Statement* next_to_last_stmt =
      block->statements()[block->statements().size() - 2];
  if (next_to_last_stmt->GetSpan().has_value() &&
      next_to_last_stmt->GetSpan()->limit() <=
          block->span().start().BumpCol()) {
    return;
  }

  // Trailing statement has to be an expression-statement.
  const Statement* last_stmt = block->statements().back();
  if (!std::holds_alternative<Expr*>(last_stmt->wrapped())) {
    return;
  }

  // It has to be a tuple.
  const auto* last_expr = std::get<Expr*>(last_stmt->wrapped());
  auto* trailing_tuple = dynamic_cast<const XlsTuple*>(last_expr);
  if (trailing_tuple == nullptr) {
    return;
  }

  // Tuple has to be nil.
  if (!trailing_tuple->empty()) {
    return;
  }

  warning_collector.Add(
      trailing_tuple->span(), WarningKind::kTrailingTupleAfterSemi,
      absl::StrFormat("Block has a trailing nil (empty) tuple after a "
                      "semicolon -- this is implied, please remove it"));
}

bool IsBlockJustUnitTuple(const StatementBlock& block) {
  if (block.empty()) {
    return true;
  }
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* tuple = dynamic_cast<const XlsTuple*>(expr);
  if (tuple == nullptr) {
    return false;
  }
  return tuple->empty();
}

bool IsBlockWithOneFailStmt(const StatementBlock& block) {
  if (block.size() != 1) {
    return false;
  }
  const Statement* statement = block.statements().front();
  const Statement::Wrapped& wrapped = statement->wrapped();
  if (!std::holds_alternative<Expr*>(wrapped)) {
    return false;
  }
  const auto* expr = std::get<Expr*>(wrapped);
  auto* invocation = dynamic_cast<const Invocation*>(expr);
  if (invocation == nullptr) {
    return false;
  }
  Expr* callee = invocation->callee();
  auto* name_ref = dynamic_cast<const NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }
  AnyNameDef any_name_def = name_ref->name_def();
  if (!std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return false;
  }
  auto* bnd = std::get<BuiltinNameDef*>(any_name_def);
  return bnd->identifier() == "fail!";
}

void WarnOnConditionalContainingJustFailStatement(
    const Conditional& node, WarningCollector& warning_collector) {
  const StatementBlock* consequent = node.consequent();
  std::variant<StatementBlock*, Conditional*> alternate_ast_node =
      node.alternate();
  if (!std::holds_alternative<StatementBlock*>(alternate_ast_node)) {
    return;
  }
  const StatementBlock* alternate =
      std::get<StatementBlock*>(alternate_ast_node);

  if (IsBlockWithOneFailStmt(*consequent) && IsBlockJustUnitTuple(*alternate)) {
    std::string message = absl::StrFormat(
        "`if test { fail!(...) } else { () }` pattern should be replaced with "
        "`assert!(test, ...)`");
    warning_collector.Add(node.span(), WarningKind::kShouldUseAssert,
                          std::move(message));
  }
}

// Sees if the function is named with a `_test` suffix but not marked with a
// test annotation -- this is likely to be a user mistake, so we give a warning.
void WarnIfConfusinglyNamedLikeTest(const Function& f,
                                    WarningCollector& warning_collector) {
  if (!absl::EndsWith(f.identifier(), "_test")) {
    return;
  }
  AstNode* parent = f.parent();
  if (parent == nullptr || parent->kind() != AstNodeKind::kTestFunction) {
    warning_collector.Add(
        f.span(), WarningKind::kMisleadingFunctionName,
        absl::StrFormat("Function `%s` ends with `_test` but is "
                        "not marked as a unit test via #[test]",
                        f.identifier()));
  }
}

// Warn folks if it's not following
// https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
void WarnOnInappropriateMemberName(std::string_view member_name,
                                   const Span& span, const Module& module,
                                   WarningCollector& warning_collector) {
  if (!IsAcceptablySnakeCase(member_name) &&
      !module.attributes().contains(
          ModuleAttribute::kAllowNonstandardMemberNaming)) {
    warning_collector.Add(
        span, WarningKind::kMemberNaming,
        absl::StrFormat("Standard style is snake_case for struct member names; "
                        "got: `%s`",
                        member_name));
  }
}

// Checks whether an expression may have side-effects. It may have
// false-positives, for example an invocation may actually have no side-effects,
// but because we do not recursively look into the callee, we conservatively
// mark it as potentially side-effect causing. Conversely, if an expression is
// determined to have no side-effects, it is definitely useless and thus should
// be flagged.
class SideEffectExpressionFinder : public AstNodeVisitorWithDefault {
 public:
  SideEffectExpressionFinder() : has_side_effect_(false) {}

  absl::Status HandleInvocation(const Invocation* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status HandleSpawn(const Spawn* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status HandleFormatMacro(const FormatMacro* node) override {
    has_side_effect_ = true;
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
      // We only need to find one side-effect.
      if (has_side_effect_) {
        return absl::OkStatus();
      }
    }
    return absl::OkStatus();
  }

  bool HasSideEffect() const { return has_side_effect_; }

 private:
  bool has_side_effect_;
};

class PreTypecheckPass : public AstNodeVisitorWithDefault {
 public:
  PreTypecheckPass(WarningCollector& warning_collector)
      : warning_collector_(warning_collector) {}

  absl::Status HandleStatementBlock(const StatementBlock* node) override {
    for (size_t i = 0; i < node->statements().size(); ++i) {
      const Statement* s = node->statements()[i];
      // We only want to check the last statement for "useless
      // expression-statement" property if it is not yielding a value from a
      // block; e.g.
      //
      //    {
      //      my_invocation!();
      //      u32:42  // <- ok, no trailing semi
      //    }
      //
      // vs
      //
      //    {
      //      my_invocation!();
      //      u32:42;  // <- useless, trailing semi means block yields nil
      //    }
      bool should_check_useless_expression =
          std::holds_alternative<Expr*>(s->wrapped()) &&
          (i != node->statements().size() - 1 || node->trailing_semi());

      if (should_check_useless_expression) {
        SideEffectExpressionFinder visitor;
        XLS_RETURN_IF_ERROR(s->Accept(&visitor));
        if (!visitor.HasSideEffect()) {
          warning_collector_.Add(
              *(s->GetSpan()), WarningKind::kUselessExpressionStatement,
              absl::StrFormat("Expression statement `%s` appears "
                              "useless (i.e. has no side-effects)",
                              s->ToString()));
        }
      }
    }
    DetectUselessTrailingTuplePattern(node, warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleConditional(const Conditional* node) override {
    WarnOnConditionalContainingJustFailStatement(*node, warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    WarnIfConfusinglyNamedLikeTest(*node, warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleStructDef(const StructDef* node) override {
    for (const auto* member : node->members()) {
      WarnOnInappropriateMemberName(member->name(), member->name_def()->span(),
                                    *node->owner(), warning_collector_);
    }
    return DefaultHandler(node);
  }
  absl::Status HandleProcDef(const ProcDef* node) override {
    for (const auto* member : node->members()) {
      WarnOnInappropriateMemberName(member->name(), member->name_def()->span(),
                                    *node->owner(), warning_collector_);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleXlsTuple(const XlsTuple* node) override {
    // Give a warning if the tuple is on a single line, is more than one
    // element, but has a trailing comma.
    //
    // Note: warning diagnostics and type checking are currently fused together,
    // but this is a pure post-parsing warning -- currently type checking the
    // pass that has a warning collector available.
    if (node->span().start().lineno() == node->span().limit().lineno() &&
        node->members().size() > 1 && node->has_trailing_comma()) {
      std::string message = absl::StrFormat(
          "Tuple expression (with >1 element) is on a single "
          "line, but has a trailing comma.");
      warning_collector_.Add(node->span(),
                             WarningKind::kSingleLineTupleTrailingComma,
                             std::move(message));
    }
    return DefaultHandler(node);
  }

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    WarnOnInappropriateConstantName(node->identifier(), node->span(),
                                    *node->owner(), &warning_collector_);
    return DefaultHandler(node);
  }

  absl::Status HandleLet(const Let* node) override {
    if (node->name_def_tree()->IsWildcardLeaf()) {
      warning_collector_.Add(node->name_def_tree()->span(),
                             WarningKind::kUselessLetBinding,
                             "`let _ = expr;` statement can be simplified to "
                             "`expr;` -- there is no "
                             "need for a `let` binding here");
    }

    if (node->is_const()) {
      NameDef* name_def = node->name_def_tree()->GetNameDefs()[0];
      WarnOnInappropriateConstantName(name_def->identifier(), node->span(),
                                      *node->owner(), &warning_collector_);
    }

    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  WarningCollector& warning_collector_;
};

}  // namespace

absl::Status SemanticsAnalysis::RunPreTypeCheckPass(
    Module& module, WarningCollector& warning_collector) {
  PreTypecheckPass pass(warning_collector);
  return module.Accept(&pass);
}

}  // namespace xls::dslx
