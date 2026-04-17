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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
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

class CollectNameRefs : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleNameRef(const NameRef* node) override {
    if (node->IsBuiltin()) {
      return DefaultHandler(node);
    }
    if (node->GetDefiner() == nullptr ||
        (node->GetDefiner()->kind() != AstNodeKind::kFunction &&
         node->GetDefiner()->kind() != AstNodeKind::kImport)) {
      XLS_RETURN_IF_ERROR(AddNameRef(node));
      if (node->GetDefiner() != nullptr) {
        XLS_RETURN_IF_ERROR(node->GetDefiner()->Accept(this));
        XLS_RETURN_IF_ERROR(DefaultHandler(node->GetDefiner()));
      }
    }
    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    bool prev_in_type_annotation = in_type_annotation_;
    if (node->kind() == AstNodeKind::kTypeAnnotation) {
      in_type_annotation_ = true;
    }
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    in_type_annotation_ = prev_in_type_annotation;
    return absl::OkStatus();
  }

  const absl::flat_hash_set<const NameRef*>& NameRefsForDef(
      const NameDef* name_def) {
    static const absl::flat_hash_set<const NameRef*> empty_set = {};
    auto it = name_ref_info_.find(name_def);
    if (it == name_ref_info_.end()) {
      return empty_set;
    }
    return it->second.name_refs;
  }

  absl::flat_hash_set<const NameDef*> NameDefsDefinedPrior(
      const Pos start) const {
    absl::flat_hash_set<const NameDef*> result;
    for (const auto& [name_def, info] : name_ref_info_) {
      if (!info.any_used_in_type_annotation &&
          name_def->span().start() < start) {
        result.insert(name_def);
      }
    }
    return result;
  }

 private:
  struct NameRefInfo {
    bool any_used_in_type_annotation;
    absl::flat_hash_set<const NameRef*> name_refs;
  };

  absl::Status AddNameRef(const NameRef* name_ref) {
    const NameDef* nd = std::get<const NameDef*>(name_ref->name_def());
    XLS_RET_CHECK(nd != nullptr);
    if (!name_ref_info_.contains(nd)) {
      NameRefInfo& info = name_ref_info_[nd];
      info.name_refs.insert(name_ref);
      if (in_type_annotation_) {
        info.any_used_in_type_annotation = true;
      }
    } else {
      NameRefInfo info = NameRefInfo{in_type_annotation_, {name_ref}};
      name_ref_info_.emplace(nd, info);
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<const NameDef*, NameRefInfo> name_ref_info_;
  bool in_type_annotation_ = false;
};

class ReplaceLambdaWithInvocation : public AstNodeVisitorWithDefault {
 public:
  ReplaceLambdaWithInvocation(const FileTable& file_table)
      : file_table_(file_table) {}

  // Converts a lambda into a struct instance and an impl. For example,
  //
  //   fn add_two(arr: u32[5]) -> u32[5] {
  //     let x = u32:2;
  //     map(arr, |i: u32| -> u32 { x + i })
  //   }
  //
  // becomes:
  //
  //   struct lambda_capture { x: u32 }
  //
  //   impl lambda_capture {
  //     fn call(self) -> u32 { self.x + i }
  //   }
  //
  //   fn add_two(arr: u32[5]) -> u32[5] {
  //     let x = u32:2;
  //     map(arr, lambda_capture{x: x}.call)
  //   }
  absl::Status HandleLambda(const Lambda* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    Module* module = node->owner();
    Function* original_fn = node->function();
    Span span = node->span();
    CollectNameRefs collect_nr;
    XLS_RETURN_IF_ERROR(node->body()->Accept(&collect_nr));
    absl::flat_hash_set<const NameDef*> seen;

    // If there are any parametric bindings in the containing function that are
    // referenced in the lambda, they should be added as parametric bindings to
    // the `StructDef`.
    std::vector<ParametricBinding*> struct_parametrics;
    std::optional<const Function*> containing_fn = GetContainingFunction(node);
    std::vector<ExprOrType> containing_fn_parametric_vals;
    absl::flat_hash_set<const NameDef*> parametric_nds;
    absl::flat_hash_map<const NameRef*, NameRef*> name_ref_replacements;

    if (containing_fn.has_value()) {
      for (ParametricBinding* parametric_binding :
           (*containing_fn)->parametric_bindings()) {
        for (const NameRef* original_name_ref :
             collect_nr.NameRefsForDef(parametric_binding->name_def())) {
          NameDef* lambda_struct_nd = module->Make<NameDef>(
              parametric_binding->span(), parametric_binding->identifier(),
              /*definer=*/nullptr);
          XLS_ASSIGN_OR_RETURN(AstNode * cloned_ta,
                               CloneAst(parametric_binding->type_annotation()));

          AstNode* cloned_expr = nullptr;
          if (parametric_binding->expr() != nullptr) {
            XLS_ASSIGN_OR_RETURN(cloned_expr,
                                 CloneAst(parametric_binding->expr()));
          }
          ParametricBinding* lambda_struct_binding =
              module->Make<ParametricBinding>(
                  lambda_struct_nd, absl::down_cast<TypeAnnotation*>(cloned_ta),
                  absl::down_cast<Expr*>(cloned_expr));
          struct_parametrics.push_back(lambda_struct_binding);
          NameRef* parametric_nr = module->Make<NameRef>(
              parametric_binding->span(), parametric_binding->identifier(),
              parametric_binding->name_def());
          containing_fn_parametric_vals.push_back(parametric_nr);
          parametric_nds.insert(parametric_binding->name_def());
          name_ref_replacements.emplace(original_name_ref, parametric_nr);
        }
      }
    }

    // For any NameDef that is referenced in the lambda, but defined prior to
    // the lambda, it must be captured in the struct instance, unless it was
    // already added as a parametric binding.
    std::vector<StructMemberNode*> struct_members;
    std::vector<std::pair<std::string, Expr*>> struct_instance_members;
    for (const NameDef* original_name_def :
         collect_nr.NameDefsDefinedPrior(span.start())) {
      if (!parametric_nds.contains(original_name_def)) {
        // Create parametric binding with generic type to use for the context
        // variable type.
        GenericTypeAnnotation* gta =
            module->Make<GenericTypeAnnotation>(original_name_def->span());
        NameDef* generic_name_def = module->Make<NameDef>(
            original_name_def->span(),
            absl::Substitute("parametric_type_for_$0",
                             original_name_def->identifier()),
            /*definer=*/gta);
        NameRef* generic_name_ref = module->Make<NameRef>(
            original_name_def->span(), generic_name_def->identifier(),
            generic_name_def);
        struct_parametrics.push_back(module->Make<ParametricBinding>(
            generic_name_def, gta, /*expr=*/nullptr));

        NameDef* struct_member_nd = module->Make<NameDef>(
            original_name_def->span(), original_name_def->identifier(),
            /*definer=*/nullptr);
        TypeVariableTypeAnnotation* tvta =
            module->Make<TypeVariableTypeAnnotation>(generic_name_ref,
                                                     /*internal=*/true);
        StructMemberNode* struct_member = module->Make<StructMemberNode>(
            Span::None(), struct_member_nd, Span::None(), tvta);
        struct_members.push_back(struct_member);

        // Make a name ref that points to the original name def. Add as a member
        // to a new struct instance.
        NameRef* struct_instance_nr = module->Make<NameRef>(
            original_name_def->span(), original_name_def->identifier(),
            original_name_def);
        struct_instance_members.push_back(std::make_pair(
            original_name_def->identifier(), struct_instance_nr));
        seen.insert(original_name_def);
      }
    }

    NameDef* struct_nd =
        module->Make<NameDef>(span,
                              absl::Substitute("lambda_capture_struct_at_$0",
                                               span.ToString(file_table_)),
                              /*definer=*/nullptr);
    StructDef* full_struct_def =
        module->Make<StructDef>(span, struct_nd, struct_parametrics,
                                struct_members, /*is_public=*/false);
    TypeRefTypeAnnotation* struct_type_annotation =
        module->Make<TypeRefTypeAnnotation>(
            span, module->Make<TypeRef>(span, full_struct_def),
            containing_fn_parametric_vals);
    struct_nd->set_definer(full_struct_def);

    StructInstance* struct_instance = module->Make<StructInstance>(
        span, struct_type_annotation, struct_instance_members);

    Attr* instance_invocation = module->Make<Attr>(
        span, struct_instance, std::string(Lambda::kCallLambdaFn));

    // For every NameRef in the body, if it references a NameDef that has been
    // captured, replace it with a reference to the struct member.
    NameDef* self_nd = module->Make<NameDef>(
        span, KeywordToString(Keyword::kSelf), /*definer=*/nullptr);
    CloneReplacer insert_self =
        [self_nd, seen, name_ref_replacements](
            const AstNode* node, const Module* _,
            const absl::flat_hash_map<const AstNode*, AstNode*>& replacements)
        -> std::optional<AstNode*> {
      if (node->kind() == AstNodeKind::kNameRef) {
        const NameRef* name_ref = absl::down_cast<const NameRef*>(node);
        if (name_ref->IsBuiltin()) {
          return std::nullopt;
        }
        const auto* name_def = std::get<const NameDef*>(name_ref->name_def());
        if (name_def != nullptr && seen.contains(name_def)) {
          NameRef* self_nr = node->owner()->Make<NameRef>(
              name_ref->span(), self_nd->identifier(), self_nd);
          return node->owner()->Make<Attr>(name_def->span(), self_nr,
                                           name_def->identifier(),
                                           /* in_parens= */ false);
        } else if (name_ref_replacements.contains(name_ref)) {
          return name_ref_replacements.at(name_ref);
        }
      }
      return std::nullopt;
    };
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_body,
        CloneAst(original_fn->body(),
                 ChainCloneReplacers(&PreserveTypeDefinitionsReplacer,
                                     std::move(insert_self))));
    SelfTypeAnnotation* self_type = module->Make<SelfTypeAnnotation>(
        span, /*explicit_type=*/false, struct_type_annotation);
    std::vector<Param*> params = {module->Make<Param>(self_nd, self_type)};
    for (auto* param : original_fn->params()) {
      params.push_back(param);
    }
    Function* impl_fn = module->Make<Function>(
        original_fn->span(), original_fn->name_def(),
        original_fn->parametric_bindings(), params, original_fn->return_type(),
        absl::down_cast<StatementBlock*>(cloned_body),
        FunctionTag::kGeneratedFromLambda,
        /*is_public=*/false, /*is_stub=*/false);
    Impl* impl = module->Make<Impl>(span, struct_type_annotation,
                                    std::vector<ImplMember>{impl_fn},
                                    /*is_public=*/false);
    impl_fn->set_impl(impl);
    full_struct_def->set_impl(impl);

    // Swap the Lambda in its parent with the attr invocation. After this step,
    // Lambdas should no longer appear in the AST.
    std::optional<ModuleMember> containing_member =
        GetContainingModuleMember(node);
    XLS_RET_CHECK(containing_member.has_value());
    auto* parent_inv = absl::down_cast<Invocation*>(node->parent());
    XLS_RET_CHECK(parent_inv != nullptr);
    if (parent_inv->callee() == node) {
      parent_inv->set_callee(instance_invocation);
    } else {
      for (int i = 0; i < parent_inv->args().size(); ++i) {
        if (parent_inv->args()[i] == node) {
          parent_inv->set_arg(i, instance_invocation);
        }
      }
    }

    XLS_RETURN_IF_ERROR(module->InsertTopBefore(ToAstNode(*containing_member),
                                                full_struct_def));
    return module->InsertTopAfter(full_struct_def, impl);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  const FileTable& file_table_;
};

// An impl-style proc automatically derives the `Spawn` attribute whether the
// user says this or not. This visitor adds it to the `derive` list for each
// proc where not present, creating the whole `derive` attribute if necessary to
// achieve this.
class AddSpawnTraitToProcDefs : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleProcDef(const ProcDef* node) override {
    std::optional<Attribute*> existing_attribute =
        GetAttribute(node, AttributeKind::kDerive);
    if (existing_attribute.has_value()) {
      for (const AttributeData::Argument& argument :
           (*existing_attribute)->args()) {
        if (std::holds_alternative<std::string>(argument) &&
            std::get<std::string>(argument) == kSpawnTraitName) {
          return absl::OkStatus();
        }
      }

      (*existing_attribute)->AddArgument(std::string(kSpawnTraitName));
      return absl::OkStatus();
    }

    Attribute* new_attribute = node->owner()->Make<Attribute>(
        Span::None(), Span::None(),
        AttributeData(AttributeKind::kDerive,
                      std::vector<AttributeData::Argument>{
                          std::string(kSpawnTraitName)}));
    const_cast<ProcDef*>(node)->AddAttribute(new_attribute);
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  static constexpr std::string_view kSpawnTraitName = "Spawn";
};

class PreTypecheckPass : public AstNodeVisitorWithDefault {
 public:
  PreTypecheckPass(WarningCollector& warning_collector,
                   const FileTable& file_table)
      : warning_collector_(warning_collector), file_table_(file_table) {}

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

  absl::Status HandleFuzzTestFunction(const FuzzTestFunction* node) override {
    const Function& f = node->fn();
    if (f.IsParametric()) {
      return TypeInferenceErrorStatus(
          f.GetSpan().value(), nullptr,
          absl::StrFormat("Cannot fuzz test parametric function `%s`",
                          f.identifier()),
          file_table_);
    }
    if (f.params().empty()) {
      return TypeInferenceErrorStatus(
          f.GetSpan().value(), nullptr,
          absl::StrFormat("Can only fuzz test functions with at least 1 "
                          "parameter; function `%s` has 0",
                          node->identifier()),
          file_table_);
    }
    if (!node->domains().has_value()) {
      return DefaultHandler(node);
    }

    const XlsTuple* domains = *node->domains();
    int64_t domain_count = domain_count = domains->members().size();
    if (domain_count != f.params().size()) {
      return TypeInferenceErrorStatus(
          node->GetSpan().value(), nullptr,
          absl::StrFormat("fuzz_test attribute has %d domain argument%s, but "
                          "function `%s` has %d parameter%s",
                          domain_count, domain_count == 1 ? "" : "s",
                          f.identifier(), f.params().size(),
                          f.params().size() == 1 ? "" : "s"),
          file_table_);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleFunction(const Function* node) override {
    WarnIfConfusinglyNamedLikeTest(*node, warning_collector_);

    // If this function is an instance method on a parametric struct, ensure
    // that it doesn't duplicate any parametric bindings from the struct.
    if (node->IsMethodOnParametricStruct() && node->IsParametric()) {
      const auto* struct_ref = absl::down_cast<const TypeRefTypeAnnotation*>(
          (*node->impl())->struct_ref());
      StructDef* struct_def =
          std::get<StructDef*>(struct_ref->type_ref()->type_definition());

      ParametricBindings bindings(node->parametric_bindings());
      for (ParametricBinding* parametric_binding :
           struct_def->parametric_bindings()) {
        if (node->parametric_keys().contains(
                parametric_binding->identifier())) {
          return xls::dslx::ParseErrorStatus(
              bindings.at(parametric_binding->identifier())->span(),
              absl::StrFormat("Parametric binding `%s` shadows binding from "
                              "struct definition",
                              parametric_binding->identifier()),
              file_table_);
        }
      }
    }
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
    // If the tuple is generated by the parser as part of a FuzzTestFunction,
    // skip the warning.
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kFuzzTestFunction) {
      return DefaultHandler(node);
    }

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

  const FileTable& file_table_;
};

class CollectUseDef : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleNameDef(const NameDef* node) override {
    // Users can silence unused warnings by prefixing an identifier with an
    // underscore to make it more well documented; e.g.
    //  let (one, _two, three) = ...;  // _two can go unused
    if (!absl::StartsWith(node->identifier(), "_")) {
      defs_.insert(node);
    }
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    AddUse(node->name_def());
    return absl::OkStatus();
  }

  absl::Status HandleTypeRef(const TypeRef* node) override {
    AddUse(TypeDefinitionGetNameDef(node->type_definition()));
    return absl::OkStatus();
  }

  absl::Status HandleTypeAlias(const TypeAlias* node) override {
    // Do not mark type alias as unused.
    return node->type_annotation().Accept(this);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  const absl::flat_hash_set<const NameDef*>& Defs() const { return defs_; }
  const absl::flat_hash_set<const NameDef*>& Uses() const { return uses_; }

 private:
  void AddUse(const AnyNameDef& any_name_def) {
    if (const NameDef* const* name_def =
            std::get_if<const NameDef*>(&any_name_def)) {
      if (!uses_.insert(*name_def).second) {
        return;
      }

      // We make an exception to NameDefTree, that if any def in it is used, all
      // defs in the tree are considered used. This is because the user
      // typically wants to keep a meaningful name for each component of a
      // NameDefTree binding, even though only some of them will be used, and we
      // want to reduce superfluous warnings on that.
      const AstNode* node = *name_def;
      while (node->parent() &&
             node->parent()->kind() == AstNodeKind::kNameDefTree) {
        node = absl::down_cast<const NameDefTree*>(node->parent());
      }
      if (node != *name_def) {
        for (NameDefTree::Leaf& leaf :
             absl::down_cast<const NameDefTree*>(node)->Flatten()) {
          if (const NameDef* const* tree_name_def =
                  std::get_if<NameDef*>(&leaf)) {
            uses_.insert(*tree_name_def);
          }
        }
      }
    }
  }

  absl::flat_hash_set<const NameDef*> defs_;
  absl::flat_hash_set<const NameDef*> uses_;
};

// Replaces the type annotation for proc state members with
// State<TheOriginalType>. In legacy procs, this affects the next() param nodes.
// In impl-style procs, it affects the declared state members of the proc.
class ProcStateVisitor : public AstNodeVisitorWithDefault {
 public:
  // Creates a visitor using `import_data` and the given `StructDef` for the
  // builtin `State` struct.
  ProcStateVisitor(ImportData& import_data, StructDef* state_struct_def)
      : import_data_(import_data), state_struct_def_(state_struct_def) {}

  absl::Status HandleFunction(const Function* node) override {
    // Legacy proc has multiple pointers to functions within the AST. This could
    // cause the HandleParam visitor to run multiple times on the same function.
    if (!processed_fns_.insert(node).second) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleParam(const Param* node) override {
    if (node->parent() != nullptr &&
        node->parent()->kind() == AstNodeKind::kFunction) {
      const Function* fn = absl::down_cast<const Function*>(node->parent());
      if (fn->tag() == FunctionTag::kProcNext) {
        const_cast<Param*>(node)->set_type_annotation(
            CreateStateTypeAnnotation(node->owner(), node->type_annotation(),
                                      node->type_annotation()->span()));
      }
    }
    return absl::OkStatus();
  }

  absl::Status HandleProcDef(const ProcDef* node) override {
    for (StructMemberNode* member : node->members()) {
      TypeAnnotation* type = member->type();

      // Don't do the T -> State<T> conversion for channels, channel arrays, or
      // sub-procs.
      if (IsChannelOrChannelArrayAnnotation(type)) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                           GetStructOrProcDef(type, import_data_));
      if (def.has_value() && (*def)->kind() == AstNodeKind::kProcDef) {
        continue;
      }

      member->set_type(
          CreateStateTypeAnnotation(node->owner(), type, type->span()));
    }
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (auto child : node->GetChildren(false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  TypeAnnotation* CreateStateTypeAnnotation(Module* module,
                                            TypeAnnotation* underlying_type,
                                            const Span& span) {
    TypeRef* state_typeref = module->Make<TypeRef>(span, state_struct_def_);
    std::vector<ExprOrType> parametrics = {underlying_type};
    return module->Make<TypeRefTypeAnnotation>(span, state_typeref,
                                               parametrics);
  }

  ImportData& import_data_;
  StructDef* const state_struct_def_;
  absl::flat_hash_set<const Function*> processed_fns_;
};

}  // namespace

SemanticsAnalysis::SemanticsAnalysis(bool suppress_warnings)
    : suppress_warnings_(suppress_warnings) {}

absl::Status SemanticsAnalysis::RunPreTypeCheckPass(
    Module& module, WarningCollector& warning_collector,
    ImportData& import_data) {
  if (module.attributes().contains(ModuleAttribute::kExplicitStateAccess)) {
    XLS_ASSIGN_OR_RETURN(Module * builtins,
                         import_data.GetBuiltinStubsModule());
    XLS_ASSIGN_OR_RETURN(StructDef * state_struct_def,
                         builtins->GetMemberOrError<StructDef>("State"));
    ProcStateVisitor state_visitor(import_data, state_struct_def);
    XLS_RETURN_IF_ERROR(module.Accept(&state_visitor));
  }
  ReplaceLambdaWithInvocation lambda_pass(import_data.file_table());
  XLS_RETURN_IF_ERROR(module.Accept(&lambda_pass));

  AddSpawnTraitToProcDefs add_spawn_trait;
  XLS_RETURN_IF_ERROR(module.Accept(&add_spawn_trait));

  if (suppress_warnings_) {
    return absl::OkStatus();
  }
  PreTypecheckPass pass(warning_collector, import_data.file_table());

  for (const ModuleMember& top : module.top()) {
    if (const Function* const* func = std::get_if<Function*>(&top)) {
      CollectUseDef visitor;
      XLS_RETURN_IF_ERROR((*func)->body()->Accept(&visitor));

      maybe_unreferenced_defs.emplace_back(
          std::make_pair(*func, std::vector<const NameDef*>()));
      std::vector<const NameDef*>& defs_in_func =
          maybe_unreferenced_defs.back().second;

      for (const NameDef* def : visitor.Defs()) {
        if (!visitor.Uses().contains(def)) {
          defs_in_func.emplace_back(def);
          def_to_type_.try_emplace(def, nullptr);
        }
      }
    }
  }

  return module.Accept(&pass);
}

// If a possibly unused def is concretized to a non-token type at any possible
// context, it is truly unused.
void SemanticsAnalysis::SetNameDefType(const NameDef* def, const Type* type) {
  auto found = def_to_type_.find(def);
  if (found != def_to_type_.end()) {
    if ((found->second && found->second->IsToken()) || !found->second) {
      found->second = type->CloneToUnique();
    }
  }
}

absl::Status SemanticsAnalysis::RunPostTypeCheckPass(
    WarningCollector& warning_collector) {
  if (suppress_warnings_) {
    return absl::OkStatus();
  }
  // Report unused defs.
  for (auto& [f, unused_defs] : maybe_unreferenced_defs) {
    // Sort them for reporting stability.
    std::sort(
        unused_defs.begin(), unused_defs.end(),
        [](const NameDef* a, const NameDef* b) {
          return a->span() < b->span() ||
                 (a->span() == b->span() && a->identifier() < b->identifier());
        });
    for (const NameDef* def : unused_defs) {
      std::unique_ptr<Type>& type = def_to_type_.at(def);
      // Tokens are implicitly joined at the end of a proc `next()`, so we
      // don't warn on these.
      if (type && !type->IsToken()) {
        warning_collector.Add(
            def->span(), WarningKind::kUnusedDefinition,
            absl::StrFormat(
                "Definition of `%s` (type `%s`) is not used in function `%s`",
                def->identifier(), type->ToString(), f->identifier()));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
