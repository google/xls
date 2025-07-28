// Copyright 2022 The XLS Authors
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
#include "xls/dslx/frontend/ast_cloner.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {
namespace {

class AstCloner : public AstNodeVisitor {
 public:
  explicit AstCloner(Module* module, CloneReplacer replacer)
      : module_(module), replacer_(std::move(replacer)) {}

  absl::Status HandleArray(const Array* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> new_members;
    new_members.reserve(n->members().size());
    for (const Expr* old_member : n->members()) {
      new_members.push_back(down_cast<Expr*>(old_to_new_.at(old_member)));
    }

    Array* new_array = module_->Make<Array>(n->span(), new_members,
                                            n->has_ellipsis(), n->in_parens());
    if (n->type_annotation() != nullptr) {
      new_array->set_type_annotation(
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())));
    }
    old_to_new_[n] = new_array;
    return absl::OkStatus();
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    old_to_new_[n] = module_->Make<ArrayTypeAnnotation>(
        n->span(),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->element_type())),
        down_cast<Expr*>(old_to_new_.at(n->dim())), n->dim_is_min());
    return absl::OkStatus();
  }

  absl::Status HandleSelfTypeAnnotation(const SelfTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    if (!old_to_new_.contains(n->struct_ref())) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->struct_ref()));
    }
    TypeAnnotation* new_struct_ref =
        down_cast<TypeAnnotation*>(old_to_new_.at(n->struct_ref()));
    old_to_new_[n] = module_->Make<SelfTypeAnnotation>(
        n->span(), n->explicit_type(), new_struct_ref);
    return absl::OkStatus();
  }

  absl::Status HandleAttr(const Attr* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    // We must've seen the NameDef here, so we don't need to visit it.
    old_to_new_[n] = module_->Make<Attr>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->lhs())),
        std::string{n->attr()}, n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleBinop(const Binop* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<Binop>(
        n->span(), n->binop_kind(), down_cast<Expr*>(old_to_new_.at(n->lhs())),
        down_cast<Expr*>(old_to_new_.at(n->rhs())), n->op_span(),
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleStatementBlock(const StatementBlock* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    std::vector<Statement*> new_statements;
    new_statements.reserve(n->statements().size());
    for (Statement* old : n->statements()) {
      if (!old_to_new_.contains(old)) {
        return absl::InternalError(absl::StrCat(
            "Statement not found in old_to_new_: ", old->ToString()));
      }
      auto new_stmt = old_to_new_.at(old);
      if (new_stmt->kind() == AstNodeKind::kVerbatimNode) {
        auto* verbatim = down_cast<VerbatimNode*>(new_stmt);
        if (verbatim->IsEmpty()) {
          // Intentionally skip empty verbatim nodes.
          continue;
        }
        new_statements.push_back(module_->Make<Statement>(verbatim));
      } else {
        new_statements.push_back(down_cast<Statement*>(new_stmt));
      }
    }
    old_to_new_[n] = module_->Make<StatementBlock>(
        n->span(), std::move(new_statements), n->trailing_semi());
    return absl::OkStatus();
  }

  absl::Status HandleConstAssert(const ConstAssert* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<ConstAssert>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->arg())));
    return absl::OkStatus();
  }

  absl::Status HandleStatement(const Statement* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    XLS_ASSIGN_OR_RETURN(
        auto new_wrapped,
        Statement::NodeToWrapped(old_to_new_.at(ToAstNode(n->wrapped()))));
    old_to_new_[n] = module_->Make<Statement>(new_wrapped);
    return absl::OkStatus();
  }

  absl::Status HandleBuiltinNameDef(const BuiltinNameDef* n) override {
    if (!old_to_new_.contains(n)) {
      old_to_new_[n] = module_->GetOrCreateBuiltinNameDef(n->identifier());
    }
    return absl::OkStatus();
  }

  absl::Status HandleBuiltinTypeAnnotation(
      const BuiltinTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->builtin_name_def()));
    old_to_new_[n] = module_->Make<BuiltinTypeAnnotation>(
        n->span(), n->builtin_type(),
        down_cast<BuiltinNameDef*>(old_to_new_.at(n->builtin_name_def())));
    return absl::OkStatus();
  }

  absl::Status HandleCast(const Cast* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<Cast>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->expr())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())),
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleChannelDecl(const ChannelDecl* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::optional<std::vector<Expr*>> new_dims;
    if (n->dims().has_value()) {
      std::vector<Expr*> old_dims_vector = n->dims().value();
      std::vector<Expr*> new_dims_vector;
      new_dims_vector.reserve(old_dims_vector.size());
      for (const Expr* expr : old_dims_vector) {
        XLS_RETURN_IF_ERROR(ReplaceOrVisit(expr));
        AstNode* val = old_to_new_.at(expr);
        new_dims_vector.push_back(down_cast<Expr*>(val));
      }
      new_dims = std::move(new_dims_vector);
    }

    ChannelDeclMetadata new_metadata = n->metadata();
    if (n->fifo_depth().has_value()) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->fifo_depth().value()));
      new_metadata = down_cast<Expr*>(old_to_new_.at(n->fifo_depth().value()));
    }

    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->type()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(&n->channel_name_expr()));
    old_to_new_[n] = module_->Make<ChannelDecl>(
        n->span(), down_cast<TypeAnnotation*>(old_to_new_.at(n->type())),
        new_dims, new_metadata,
        *down_cast<Expr*>(old_to_new_.at(&n->channel_name_expr())),
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleChannelTypeAnnotation(
      const ChannelTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::optional<std::vector<Expr*>> new_dims;
    if (n->dims().has_value()) {
      std::vector<Expr*> old_dims_vector = n->dims().value();
      std::vector<Expr*> new_dims_vector;
      new_dims_vector.reserve(old_dims_vector.size());
      for (const Expr* expr : old_dims_vector) {
        new_dims_vector.push_back(down_cast<Expr*>(old_to_new_.at(expr)));
      }
      new_dims = new_dims_vector;
    }

    old_to_new_[n] = module_->Make<ChannelTypeAnnotation>(
        n->span(), n->direction(),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->payload())), new_dims);
    return absl::OkStatus();
  }

  absl::Status HandleColonRef(const ColonRef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    ColonRef::Subject new_subject = absl::visit(
        Visitor{[&](NameRef* name_ref) -> ColonRef::Subject {
                  AstNode* new_node = old_to_new_.at(name_ref);
                  // Do an integrity check that the NameRef variant indeed is
                  // holding a NameRef node.
                  CHECK_EQ(new_node->kind(), AstNodeKind::kNameRef);
                  return down_cast<NameRef*>(new_node);
                },
                [&](ColonRef* colon_ref) -> ColonRef::Subject {
                  return down_cast<ColonRef*>(old_to_new_.at(colon_ref));
                },
                [&](TypeRefTypeAnnotation* type_ref) -> ColonRef::Subject {
                  return down_cast<TypeRefTypeAnnotation*>(
                      old_to_new_.at(type_ref));
                }},
        n->subject());

    old_to_new_[n] = module_->Make<ColonRef>(n->span(), new_subject, n->attr(),
                                             n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleConstantDef(const ConstantDef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    TypeAnnotation* new_type_annotation = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type_annotation =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }
    old_to_new_[n] = module_->Make<ConstantDef>(
        n->span(), down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        new_type_annotation, down_cast<Expr*>(old_to_new_.at(n->value())),
        n->is_public());
    return absl::OkStatus();
  }

  absl::Status HandleEnumDef(const EnumDef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    NameDef* new_name_def = down_cast<NameDef*>(old_to_new_.at(n->name_def()));
    TypeAnnotation* new_type_annotation = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type_annotation =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }
    std::vector<EnumMember> new_values;
    for (const auto& member : n->values()) {
      new_values.push_back(EnumMember{
          .name_def = down_cast<NameDef*>(old_to_new_.at(member.name_def)),
          .value = down_cast<Expr*>(old_to_new_.at(member.value))});
    }

    EnumDef* new_enum_def =
        module_->Make<EnumDef>(n->span(), new_name_def, new_type_annotation,
                               new_values, n->is_public());
    new_name_def->set_definer(new_enum_def);
    if (n->extern_type_name().has_value()) {
      new_enum_def->set_extern_type_name(*n->extern_type_name());
    }

    old_to_new_[n] = new_enum_def;

    return absl::OkStatus();
  }

  absl::Status HandleFor(const For* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->names()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->type_annotation()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->iterable()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->body()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->init()));

    auto new_type_annotation =
        n->type_annotation() == nullptr
            ? nullptr
            : down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));

    old_to_new_[n] = module_->Make<For>(
        n->span(), down_cast<NameDefTree*>(old_to_new_.at(n->names())),
        new_type_annotation, down_cast<Expr*>(old_to_new_.at(n->iterable())),
        down_cast<StatementBlock*>(old_to_new_.at(n->body())),
        down_cast<Expr*>(old_to_new_.at(n->init())));
    return absl::OkStatus();
  }

  absl::Status HandleFormatMacro(const FormatMacro* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> new_args;
    new_args.reserve(n->args().size());
    for (const Expr* arg : n->args()) {
      new_args.push_back(down_cast<Expr*>(old_to_new_.at(arg)));
    }

    std::optional<Expr*> new_verbosity;
    if (n->verbosity().has_value()) {
      new_verbosity = down_cast<Expr*>(old_to_new_.at(*n->verbosity()));
    }

    old_to_new_[n] = module_->Make<FormatMacro>(
        n->span(), n->macro(),
        std::vector<FormatStep>(n->format().begin(), n->format().end()),
        new_args, new_verbosity);
    return absl::OkStatus();
  }

  absl::Status HandleFunctionRef(const FunctionRef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<FunctionRef>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->callee())),
        CloneParametrics(n->explicit_parametrics()));
    return absl::OkStatus();
  }

  absl::Status HandleZeroMacro(const ZeroMacro* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    ExprOrType new_eot = ToExprOrType(old_to_new_.at(ToAstNode(n->type())));
    old_to_new_[n] =
        module_->Make<ZeroMacro>(n->span(), new_eot, n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleAllOnesMacro(const AllOnesMacro* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    ExprOrType new_eot = ToExprOrType(old_to_new_.at(ToAstNode(n->type())));
    old_to_new_[n] =
        module_->Make<AllOnesMacro>(n->span(), new_eot, n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const Function* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    XLS_ASSIGN_OR_RETURN(
        NameDef * new_name_def,
        CastIfNotVerbatim<NameDef*>(old_to_new_.at(n->name_def())));

    std::vector<ParametricBinding*> new_parametric_bindings;
    new_parametric_bindings.reserve(n->parametric_bindings().size());
    for (const ParametricBinding* pb : n->parametric_bindings()) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(pb));
      new_parametric_bindings.push_back(
          down_cast<ParametricBinding*>(old_to_new_.at(pb)));
    }

    std::vector<Param*> new_params;
    new_params.reserve(n->params().size());
    for (const auto* param : n->params()) {
      new_params.push_back(down_cast<Param*>(old_to_new_.at(param)));
    }

    TypeAnnotation* new_return_type = nullptr;
    if (n->return_type() != nullptr) {
      new_return_type =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->return_type()));
    }
    XLS_ASSIGN_OR_RETURN(
        StatementBlock * new_body,
        CastIfNotVerbatim<StatementBlock*>(old_to_new_.at(n->body())));
    auto new_function = module_->Make<Function>(
        n->span(), new_name_def, new_parametric_bindings, new_params,
        new_return_type, new_body, n->tag(), n->is_public(),
        n->is_test_utility());
    if (n->extern_verilog_module().has_value()) {
      new_function->set_extern_verilog_module(*n->extern_verilog_module());
    }
    new_function->set_disable_format(n->disable_format());
    old_to_new_[n] = new_function;
    new_name_def->set_definer(old_to_new_.at(n));
    if (n->impl().has_value()) {
      Impl* new_impl = down_cast<Impl*>(old_to_new_.at(*(n->impl())));
      new_function->set_impl(new_impl);
    }
    return absl::OkStatus();
  }

  absl::Status HandleImport(const Import* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    old_to_new_[n] = module_->Make<Import>(
        n->span(), n->subject(),
        *down_cast<NameDef*>(old_to_new_.at(&n->name_def())), n->alias());
    return absl::OkStatus();
  }

  absl::Status HandleUse(const Use* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    const UseTreeEntry* old_root = &n->root();
    UseTreeEntry& new_root =
        *down_cast<UseTreeEntry*>(old_to_new_.at(old_root));
    old_to_new_[n] = module_->Make<Use>(n->span(), new_root);
    return absl::OkStatus();
  }

  absl::Status HandleUseTreeEntry(const UseTreeEntry* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    using PayloadT = std::variant<UseInteriorEntry, NameDef*>;
    PayloadT new_payload = absl::visit(
        Visitor{[&](const NameDef* name_def) -> PayloadT {
                  return down_cast<NameDef*>(old_to_new_.at(name_def));
                },
                [&](const UseInteriorEntry& interior) -> PayloadT {
                  std::vector<UseTreeEntry*> new_subtrees;
                  new_subtrees.reserve(interior.subtrees().size());
                  for (UseTreeEntry* subtree : interior.subtrees()) {
                    new_subtrees.push_back(
                        down_cast<UseTreeEntry*>(old_to_new_.at(subtree)));
                  }
                  return UseInteriorEntry{std::string{interior.identifier()},
                                          std::move(new_subtrees)};
                }},
        n->payload());

    old_to_new_[n] =
        module_->Make<UseTreeEntry>(std::move(new_payload), n->span());
    return absl::OkStatus();
  }

  absl::Status HandleIndex(const Index* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    IndexRhs new_rhs =
        absl::visit(Visitor{[&](auto* expr) -> IndexRhs {
                      return down_cast<decltype(expr)>(old_to_new_.at(expr));
                    }},
                    n->rhs());

    old_to_new_[n] = module_->Make<Index>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->lhs())), new_rhs,
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> new_args;
    new_args.reserve(n->args().size());
    for (const Expr* arg : n->args()) {
      new_args.push_back(down_cast<Expr*>(old_to_new_.at(arg)));
    }
    std::optional<Invocation*> new_originator = std::nullopt;
    if (n->originating_invocation().has_value()) {
      if (!old_to_new_.contains(*n->originating_invocation())) {
        XLS_RETURN_IF_ERROR(ReplaceOrVisit(*n->originating_invocation()));
      }
      new_originator =
          down_cast<Invocation*>(old_to_new_.at(*n->originating_invocation()));
    }

    old_to_new_[n] = module_->Make<Invocation>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->callee())), new_args,
        CloneParametrics(n->explicit_parametrics()), n->in_parens(),
        new_originator);
    return absl::OkStatus();
  }

  absl::Status HandleLambda(const Lambda* n) override {
    return absl::UnimplementedError("lambdas not yet supported");
  }

  absl::Status HandleLet(const Let* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    TypeAnnotation* new_type = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }

    old_to_new_[n] = module_->Make<Let>(
        n->span(), down_cast<NameDefTree*>(old_to_new_.at(n->name_def_tree())),
        new_type, down_cast<Expr*>(old_to_new_.at(n->rhs())), n->is_const());
    return absl::OkStatus();
  }

  absl::Status HandleMatch(const Match* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<MatchArm*> new_arms;
    new_arms.reserve(n->arms().size());
    for (const MatchArm* arm : n->arms()) {
      new_arms.push_back(down_cast<MatchArm*>(old_to_new_.at(arm)));
    }

    old_to_new_[n] = module_->Make<Match>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->matched())), new_arms,
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleMatchArm(const MatchArm* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<NameDefTree*> new_patterns;
    new_patterns.reserve(n->patterns().size());
    for (const NameDefTree* pattern : n->patterns()) {
      new_patterns.push_back(down_cast<NameDefTree*>(old_to_new_.at(pattern)));
    }

    old_to_new_[n] = module_->Make<MatchArm>(
        n->span(), new_patterns, down_cast<Expr*>(old_to_new_.at(n->expr())));

    return absl::OkStatus();
  }

  // If `replacement` is a `VerbatimNode`, return it. Otherwise, return the
  // `replacement` cast to the given type.
  template <typename T>
  static ModuleMember CastOrVerbatim(AstNode* replacement) {
    if (replacement->kind() == AstNodeKind::kVerbatimNode) {
      return down_cast<VerbatimNode*>(replacement);
    }
    return down_cast<T>(replacement);
  }

  absl::Status HandleModule(const Module* n) override {
    for (const ModuleMember member : n->top()) {
      ModuleMember new_member;
      XLS_RETURN_IF_ERROR(absl::visit(
          Visitor{
              [&](auto* old_member) -> absl::Status {
                XLS_RETURN_IF_ERROR(ReplaceOrVisit(old_member));
                new_member = CastOrVerbatim<decltype(old_member)>(
                    old_to_new().at(old_member));
                return absl::OkStatus();
              },
          },
          member));
      XLS_RETURN_IF_ERROR(
          module_->AddTop(new_member, /*make_collision_error=*/nullptr));
    }

    return absl::OkStatus();
  }

  absl::Status HandleNameDef(const NameDef* n) override {
    if (!old_to_new_.contains(n)) {
      // We need to set the definer in the definer itself, not here, to avoid
      // looping (Function -> NameDef -> Function -> NameDef -> ...).
      old_to_new_[n] =
          module_->Make<NameDef>(n->span(), n->identifier(), nullptr);
    }

    return absl::OkStatus();
  }

  absl::Status HandleNameDefTree(const NameDefTree* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    if (n->is_leaf()) {
      NameDefTree::Leaf leaf;
      absl::visit(
          [&](auto* x) { leaf = down_cast<decltype(x)>(old_to_new_.at(x)); },
          n->leaf());
      old_to_new_[n] = module_->Make<NameDefTree>(n->span(), leaf);
      return absl::OkStatus();
    }

    NameDefTree::Nodes nodes;
    nodes.reserve(n->nodes().size());
    for (const auto& node : n->nodes()) {
      nodes.push_back(down_cast<NameDefTree*>(old_to_new_.at(node)));
    }
    old_to_new_[n] = module_->Make<NameDefTree>(n->span(), nodes);
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* n) override {
    // If it's a ref to a cloned def, then point it to the cloned def.
    // Otherwise, it may be a ref to a def that is outside the scope being
    // cloned.
    auto it = old_to_new_.end();
    if (std::holds_alternative<const NameDef*>(n->name_def())) {
      it = old_to_new_.find(std::get<const NameDef*>(n->name_def()));
    }
    old_to_new_[n] = module_->Make<NameRef>(
        n->span(), n->identifier(),
        it == old_to_new_.end() ? n->name_def()
                                : down_cast<NameDef*>(it->second),
        n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleNumber(const Number* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    TypeAnnotation* new_type = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }
    old_to_new_[n] = module_->Make<Number>(
        n->span(), n->text(), n->number_kind(), new_type, n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleParam(const Param* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<Param>(
        down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())));
    return absl::OkStatus();
  }

  absl::Status HandleProcMember(const ProcMember* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<ProcMember>(
        down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())));
    return absl::OkStatus();
  }

  absl::Status HandleParametricBinding(const ParametricBinding* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    Expr* new_expr = nullptr;
    if (n->expr() != nullptr) {
      new_expr = down_cast<Expr*>(old_to_new_.at(n->expr()));
    }
    old_to_new_[n] = module_->Make<ParametricBinding>(
        down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())),
        new_expr);
    return absl::OkStatus();
  }

  absl::Status HandleProc(const Proc* n) override {
    if (old_to_new_.contains(n)) {
      // If we've already cloned this proc, just return it.
      return absl::OkStatus();
    }
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<ParametricBinding*> new_parametric_bindings;
    new_parametric_bindings.reserve(n->parametric_bindings().size());
    for (const ParametricBinding* pb : n->parametric_bindings()) {
      new_parametric_bindings.push_back(
          down_cast<ParametricBinding*>(old_to_new_.at(pb)));
    }

    std::vector<ProcMember*> new_members;
    new_members.reserve(n->members().size());
    for (const ProcMember* member : n->members()) {
      new_members.push_back(down_cast<ProcMember*>(old_to_new_.at(member)));
    }

    std::vector<ProcStmt> new_stmts;
    new_stmts.reserve(n->stmts().size());
    for (const ProcStmt& stmt : n->stmts()) {
      XLS_ASSIGN_OR_RETURN(ProcStmt new_stmt,
                           ToProcStmt(old_to_new_.at(ToAstNode(stmt))));
      new_stmts.push_back(new_stmt);
    }

    NameDef* new_name_def = down_cast<NameDef*>(old_to_new_.at(n->name_def()));
    ProcLikeBody new_body = {
        .stmts = new_stmts,
        .config = down_cast<Function*>(old_to_new_.at(&n->config())),
        .next = down_cast<Function*>(old_to_new_.at(&n->next())),
        .init = down_cast<Function*>(old_to_new_.at(&n->init())),
        .members = new_members,
    };
    Proc* p =
        module_->Make<Proc>(n->span(), n->body_span(), new_name_def,
                            new_parametric_bindings, new_body, n->is_public());
    new_name_def->set_definer(p);
    old_to_new_[n] = p;
    return absl::OkStatus();
  }

  absl::Status HandleQuickCheck(const QuickCheck* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<QuickCheck>(
        n->GetSpan().value(), down_cast<Function*>(old_to_new_.at(n->fn())),
        n->test_cases());
    return absl::OkStatus();
  }

  absl::Status HandleRange(const Range* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<Range>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->start())),
        n->inclusive_end(), down_cast<Expr*>(old_to_new_.at(n->end())),
        n->in_parens(), n->has_pattern_semantics());
    return absl::OkStatus();
  }

  absl::Status HandleSlice(const Slice* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    Expr* new_start = n->start() == nullptr
                          ? nullptr
                          : down_cast<Expr*>(old_to_new_.at(n->start()));
    Expr* new_limit = n->limit() == nullptr
                          ? nullptr
                          : down_cast<Expr*>(old_to_new_.at(n->limit()));

    old_to_new_[n] =
        module_->Make<Slice>(n->GetSpan().value(), new_start, new_limit);
    return absl::OkStatus();
  }

  absl::Status HandleSpawn(const Spawn* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->callee()));
    old_to_new_[n] = module_->Make<Spawn>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->callee())),
        down_cast<Invocation*>(old_to_new_.at(n->config())),
        down_cast<Invocation*>(old_to_new_.at(n->next())),
        CloneParametrics(n->explicit_parametrics()));
    return absl::OkStatus();
  }

  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    // Have to explicitly visit struct ref, since it's not a child.
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->struct_ref()));
    TypeAnnotation* new_struct_ref =
        down_cast<TypeAnnotation*>(old_to_new_.at(n->struct_ref()));

    const std::vector<std::pair<std::string, Expr*>>& old_members =
        n->members();
    std::vector<std::pair<std::string, Expr*>> new_members;
    new_members.reserve(old_members.size());
    for (const auto& member : old_members) {
      new_members.push_back(std::make_pair(
          member.first, down_cast<Expr*>(old_to_new_.at(member.second))));
    }

    old_to_new_[n] = module_->Make<SplatStructInstance>(
        n->span(), new_struct_ref, new_members,
        down_cast<Expr*>(old_to_new_.at(n->splatted())));
    return absl::OkStatus();
  }

  absl::Status HandleString(const String* n) override {
    old_to_new_[n] =
        module_->Make<String>(n->span(), n->text(), n->in_parens());
    return absl::OkStatus();
  }

  // If the node is a VerbatimNode, returns an error; otherwise, casts to the
  // template type.
  template <typename T>
  static absl::StatusOr<T> CastIfNotVerbatim(AstNode* node) {
    if (node->kind() == AstNodeKind::kVerbatimNode) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot disable formatting here yet: ",
                       down_cast<VerbatimNode*>(node)->text()));
    }
    return down_cast<T>(node);
  }

  // Helper function to clone a node of type `T` where `T` is a concrete
  // subclass of `StructDefBase`.
  template <typename T>
  absl::Status HandleStructDefBaseInternal(const T* n) {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<ParametricBinding*> new_parametric_bindings;
    new_parametric_bindings.reserve(n->parametric_bindings().size());
    for (const auto* pb : n->parametric_bindings()) {
      XLS_ASSIGN_OR_RETURN(
          ParametricBinding * new_pb,
          CastIfNotVerbatim<ParametricBinding*>(old_to_new_.at(pb)));
      new_parametric_bindings.push_back(new_pb);
    }

    std::vector<StructMemberNode*> new_members;
    for (const StructMemberNode* member : n->members()) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(member->name_def()));
      XLS_ASSIGN_OR_RETURN(
          TypeAnnotation * new_type,
          CastIfNotVerbatim<TypeAnnotation*>(old_to_new_.at(member->type())));
      XLS_ASSIGN_OR_RETURN(
          NameDef * new_name,
          CastIfNotVerbatim<NameDef*>(old_to_new_.at(member->name_def())));
      new_members.push_back(module_->Make<StructMemberNode>(
          member->span(), new_name, member->colon_span(), new_type));
    }

    auto* new_name_def = down_cast<NameDef*>(old_to_new_.at(n->name_def()));
    auto* new_struct_def =
        module_->Make<T>(n->span(), new_name_def, new_parametric_bindings,
                         new_members, n->is_public());
    old_to_new_[n] = new_struct_def;

    if (n->impl().has_value()) {
      if (!old_to_new_.contains(n->impl().value())) {
        XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->impl().value()));
      }
      auto* new_impl = down_cast<Impl*>(old_to_new_.at(n->impl().value()));
      new_struct_def->set_impl(new_impl);
    }
    new_name_def->set_definer(new_struct_def);
    return absl::OkStatus();
  }

  absl::Status HandleStructDef(const StructDef* n) override {
    absl::Status status = HandleStructDefBaseInternal(n);
    if (status.ok()) {
      AstNode* new_node = old_to_new_.at(n);
      if (new_node->kind() == AstNodeKind::kStructDef) {
        auto new_struct_def = down_cast<StructDef*>(new_node);
        if (n->extern_type_name().has_value()) {
          new_struct_def->set_extern_type_name(*n->extern_type_name());
        }
      }
    }
    return status;
  }

  absl::Status HandleProcDef(const ProcDef* n) override {
    return HandleStructDefBaseInternal(n);
  }

  absl::Status HandleImpl(const Impl* n) override {
    // To avoid infinite loops between impl -> function -> struct -> impl, use a
    // placeholder for children (members and struct def) and add afterward.
    Impl* new_impl = module_->Make<Impl>(
        n->span(), nullptr, std::vector<ImplMember>{}, n->is_public());
    old_to_new_[n] = new_impl;
    if (!old_to_new_.contains(n->struct_ref())) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->struct_ref()));
    }
    TypeAnnotation* new_struct_ref =
        down_cast<TypeAnnotation*>(old_to_new_.at(n->struct_ref()));
    new_impl->set_struct_ref(new_struct_ref);
    std::vector<ImplMember> new_members;
    new_members.reserve(n->members().size());
    for (const auto& member : n->members()) {
      AstNode* member_node = ToAstNode(member);
      if (!old_to_new_.contains(member_node)) {
        XLS_RETURN_IF_ERROR(ReplaceOrVisit(member_node));
      }
      AstNode* new_node = old_to_new_.at(member_node);
      if (new_node->kind() == AstNodeKind::kConstantDef) {
        new_members.push_back(down_cast<ConstantDef*>(new_node));
      } else if (new_node->kind() == AstNodeKind::kFunction) {
        new_members.push_back(down_cast<Function*>(new_node));
      } else {
        new_members.push_back(down_cast<VerbatimNode*>(new_node));
      }
    }
    new_impl->set_members(new_members);
    return absl::OkStatus();
  }

  absl::Status HandleStructMemberNode(const StructMemberNode* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    StructMemberNode* new_struct_member = module_->Make<StructMemberNode>(
        n->span(), down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        n->colon_span(), down_cast<TypeAnnotation*>(old_to_new_.at(n->type())));
    old_to_new_[n] = new_struct_member;
    return absl::OkStatus();
  }

  absl::Status HandleStructInstance(const StructInstance* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    // Have to explicitly visit struct ref, since it's not a child.
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->struct_ref()));
    TypeAnnotation* new_struct_ref =
        down_cast<TypeAnnotation*>(old_to_new_.at(n->struct_ref()));

    absl::Span<const std::pair<std::string, Expr*>> old_members =
        n->GetUnorderedMembers();
    std::vector<std::pair<std::string, Expr*>> new_members;
    new_members.reserve(old_members.size());
    for (const auto& member : old_members) {
      new_members.push_back(std::make_pair(
          member.first, down_cast<Expr*>(old_to_new_.at(member.second))));
    }

    old_to_new_[n] = module_->Make<StructInstance>(n->span(), new_struct_ref,
                                                   new_members, n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleConditional(const Conditional* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::variant<StatementBlock*, Conditional*> new_alternate;
    AstNode* new_alternate_node = old_to_new_.at(ToAstNode(n->alternate()));
    if (new_alternate_node->kind() == AstNodeKind::kStatementBlock) {
      new_alternate = down_cast<StatementBlock*>(new_alternate_node);
    } else if (new_alternate_node->kind() == AstNodeKind::kConditional) {
      new_alternate = down_cast<Conditional*>(new_alternate_node);
    } else {
      return absl::InternalError("Unexpected Conditional alternate node type.");
    }

    old_to_new_[n] = module_->Make<Conditional>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->test())),
        down_cast<StatementBlock*>(old_to_new_.at(n->consequent())),
        new_alternate, n->in_parens(), n->HasElse());
    return absl::OkStatus();
  }

  absl::Status HandleTestFunction(const TestFunction* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    XLS_RETURN_IF_ERROR(ReplaceOrVisit(&n->fn()));
    old_to_new_[n] = module_->Make<TestFunction>(
        n->span(), *down_cast<Function*>(old_to_new_.at(&n->fn())));
    return absl::OkStatus();
  }

  absl::Status HandleTestProc(const TestProc* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    old_to_new_[n] = module_->Make<TestProc>(
        down_cast<Proc*>(old_to_new_.at(n->proc())), n->expected_fail_label());
    return absl::OkStatus();
  }

  absl::Status HandleTupleIndex(const TupleIndex* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<TupleIndex>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->lhs())),
        down_cast<Number*>(old_to_new_.at(n->index())), n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleTupleTypeAnnotation(
      const TupleTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<TypeAnnotation*> new_members;
    new_members.reserve(n->members().size());
    for (const auto* member : n->members()) {
      new_members.push_back(down_cast<TypeAnnotation*>(old_to_new_.at(member)));
    }
    old_to_new_[n] = module_->Make<TupleTypeAnnotation>(n->span(), new_members);
    return absl::OkStatus();
  }

  absl::Status HandleTypeAlias(const TypeAlias* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    NameDef* new_name_def = down_cast<NameDef*>(old_to_new_.at(&n->name_def()));
    TypeAlias* new_ta = module_->Make<TypeAlias>(
        n->span(), *new_name_def,
        *down_cast<TypeAnnotation*>(old_to_new_.at(&n->type_annotation())),
        n->is_public());
    if (n->extern_type_name().has_value()) {
      new_ta->set_extern_type_name(*n->extern_type_name());
    }
    new_name_def->set_definer(new_ta);
    old_to_new_[n] = new_ta;
    return absl::OkStatus();
  }

  absl::Status HandleTypeRef(const TypeRef* n) override {
    TypeDefinition new_type_definition = n->type_definition();

    // A TypeRef doesn't own its referenced type definition, so we have to
    // explicitly visit it.
    XLS_RETURN_IF_ERROR(absl::visit(
        Visitor{[&](auto* ref) -> absl::Status {
          XLS_RETURN_IF_ERROR(ReplaceOrVisit(ref));
          new_type_definition = down_cast<decltype(ref)>(old_to_new_.at(ref));
          return absl::OkStatus();
        }},
        n->type_definition()));

    old_to_new_[n] = module_->Make<TypeRef>(n->span(), new_type_definition);
    return absl::OkStatus();
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<TypeRefTypeAnnotation>(
        n->span(), down_cast<TypeRef*>(old_to_new_.at(n->type_ref())),
        CloneParametrics(n->parametrics()));
    return absl::OkStatus();
  }

  absl::Status HandleTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->type_variable()));
    old_to_new_[n] = module_->Make<TypeVariableTypeAnnotation>(
        down_cast<const NameRef*>(old_to_new_[n->type_variable()]));
    return absl::OkStatus();
  }

  absl::Status HandleMemberTypeAnnotation(
      const MemberTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->struct_type()));
    old_to_new_[n] = module_->Make<MemberTypeAnnotation>(
        down_cast<const TypeAnnotation*>(old_to_new_[n->struct_type()]),
        n->member_name());
    return absl::OkStatus();
  }

  absl::Status HandleElementTypeAnnotation(
      const ElementTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->container_type()));
    if (n->tuple_index().has_value()) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(*n->tuple_index()));
    }
    old_to_new_[n] = module_->Make<ElementTypeAnnotation>(
        down_cast<const TypeAnnotation*>(old_to_new_[n->container_type()]),
        n->tuple_index().has_value()
            ? std::make_optional(
                  down_cast<const Number*>(old_to_new_[*n->tuple_index()]))
            : std::nullopt);
    return absl::OkStatus();
  }

  absl::Status HandleSliceTypeAnnotation(
      const SliceTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->source_type()));
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(ToAstNode(n->slice())));
    AstNode* new_slice_node = old_to_new_[ToAstNode(n->slice())];
    std::variant<Slice*, WidthSlice*> new_slice;
    if (new_slice_node->kind() == AstNodeKind::kSlice) {
      new_slice = down_cast<Slice*>(new_slice_node);
    } else {
      new_slice = down_cast<WidthSlice*>(new_slice_node);
    }
    old_to_new_[n] = module_->Make<SliceTypeAnnotation>(
        n->span(), down_cast<TypeAnnotation*>(old_to_new_[n->source_type()]),
        new_slice);
    return absl::OkStatus();
  }

  absl::Status HandleFunctionTypeAnnotation(
      const FunctionTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->return_type()));
    std::vector<const TypeAnnotation*> param_types;
    param_types.reserve(n->param_types().size());
    for (const TypeAnnotation* argument : n->param_types()) {
      XLS_RETURN_IF_ERROR(ReplaceOrVisit(argument));
      param_types.push_back(
          down_cast<const TypeAnnotation*>(old_to_new_[argument]));
    }
    old_to_new_[n] = module_->Make<FunctionTypeAnnotation>(
        std::move(param_types),
        down_cast<TypeAnnotation*>(old_to_new_[n->return_type()]));
    return absl::OkStatus();
  }

  absl::Status HandleReturnTypeAnnotation(
      const ReturnTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->function_type()));
    old_to_new_[n] = module_->Make<ReturnTypeAnnotation>(
        down_cast<TypeAnnotation*>(old_to_new_[n->function_type()]));
    return absl::OkStatus();
  }

  absl::Status HandleParamTypeAnnotation(
      const ParamTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(ReplaceOrVisit(n->function_type()));
    old_to_new_[n] = module_->Make<ParamTypeAnnotation>(
        down_cast<TypeAnnotation*>(old_to_new_[n->function_type()]),
        n->param_index());
    return absl::OkStatus();
  }

  absl::Status HandleAnyTypeAnnotation(const AnyTypeAnnotation* n) override {
    old_to_new_[n] = module_->Make<AnyTypeAnnotation>(n->multiple());
    return absl::OkStatus();
  }

  absl::Status HandleUnop(const Unop* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] =
        module_->Make<Unop>(n->span(), n->unop_kind(),
                            down_cast<Expr*>(old_to_new_.at(n->operand())),
                            n->op_span(), n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleUnrollFor(const UnrollFor* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    auto new_type_annotation =
        n->type_annotation() == nullptr
            ? nullptr
            : down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));

    old_to_new_[n] = module_->Make<UnrollFor>(
        n->span(), down_cast<NameDefTree*>(old_to_new_.at(n->names())),
        new_type_annotation, down_cast<Expr*>(old_to_new_.at(n->iterable())),
        down_cast<StatementBlock*>(old_to_new_.at(n->body())),
        down_cast<Expr*>(old_to_new_.at(n->init())), n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleWidthSlice(const WidthSlice* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<WidthSlice>(
        n->GetSpan().value(), down_cast<Expr*>(old_to_new_.at(n->start())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->width())));
    return absl::OkStatus();
  }

  absl::Status HandleWildcardPattern(const WildcardPattern* n) override {
    old_to_new_[n] = module_->Make<WildcardPattern>(n->span());
    return absl::OkStatus();
  }

  absl::Status HandleRestOfTuple(const RestOfTuple* n) override {
    old_to_new_[n] = module_->Make<RestOfTuple>(n->span());
    return absl::OkStatus();
  }

  absl::Status HandleXlsTuple(const XlsTuple* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> members;
    members.reserve(n->members().size());
    for (const auto* member : n->members()) {
      members.push_back(down_cast<Expr*>(old_to_new_.at(member)));
    }

    old_to_new_[n] = module_->Make<XlsTuple>(
        n->span(), members, n->has_trailing_comma(), n->in_parens());
    return absl::OkStatus();
  }

  absl::Status HandleVerbatimNode(const VerbatimNode* n) override {
    old_to_new_[n] = module_->Make<VerbatimNode>(n->span(), n->text());
    return absl::OkStatus();
  }

  absl::Status HandleGenericTypeAnnotation(
      const GenericTypeAnnotation* n) override {
    old_to_new_[n] = module_->Make<GenericTypeAnnotation>(n->span());
    return absl::OkStatus();
  }

  const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new() {
    return old_to_new_;
  }

 private:
  // Visits all the children of the given node, skipping those that have
  // already been processed.
  absl::Status VisitChildren(const AstNode* node) {
    for (const auto& child : node->GetChildren(/*want_types=*/true)) {
      if (!old_to_new_.contains(child)) {
        XLS_RETURN_IF_ERROR(ReplaceOrVisit(child));
      }
    }
    return absl::OkStatus();
  }

  absl::Status ReplaceOrVisit(const AstNode* node) {
    if (node == nullptr) {
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(std::optional<AstNode*> replacement, replacer_(node));
    if (replacement.has_value()) {
      old_to_new_[node] = *replacement;
      return absl::OkStatus();
    }
    return node->Accept(this);
  }

  std::vector<ExprOrType> CloneParametrics(
      const std::vector<ExprOrType>& parametrics) {
    std::vector<ExprOrType> new_parametrics;
    new_parametrics.reserve(parametrics.size());
    for (const ExprOrType& parametric : parametrics) {
      AstNode* old_node = ToAstNode(parametric);
      if (ReplaceOrVisit(old_node).ok()) {
        AstNode* new_node = old_to_new_.at(old_node);
        if (std::holds_alternative<Expr*>(parametric)) {
          new_parametrics.push_back(down_cast<Expr*>(new_node));
        } else {
          new_parametrics.push_back(down_cast<TypeAnnotation*>(new_node));
        }
      }
    }
    return new_parametrics;
  }

  Module* const module_;
  CloneReplacer replacer_;
  absl::flat_hash_map<const AstNode*, AstNode*> old_to_new_;
};

}  // namespace

std::optional<AstNode*> PreserveTypeDefinitionsReplacer(const AstNode* node) {
  if (node->kind() == AstNodeKind::kTypeRef) {
    const auto* type_ref = down_cast<const TypeRef*>(node);
    return node->owner()->Make<TypeRef>(type_ref->span(),
                                        type_ref->type_definition());
  }
  return std::nullopt;
}

CloneReplacer NameRefReplacer(const NameDef* def, Expr* replacement) {
  return [=](const AstNode* node) -> std::optional<AstNode*> {
    if (node->kind() == AstNodeKind::kNameRef) {
      const auto* name_ref = down_cast<const NameRef*>(node);
      if (std::holds_alternative<const NameDef*>(name_ref->name_def()) &&
          std::get<const NameDef*>(name_ref->name_def()) == def) {
        return replacement;
      }
    }
    return std::nullopt;
  };
}

CloneReplacer NameRefReplacer(
    const absl::flat_hash_map<const NameDef*, NameDef*>* replacement_defs) {
  return [=](const AstNode* original_node) -> std::optional<AstNode*> {
    if (original_node->kind() == AstNodeKind::kNameRef) {
      const auto* original_ref = down_cast<const NameRef*>(original_node);
      const AstNode* def = ToAstNode(original_ref->name_def());
      if (def->kind() == AstNodeKind::kNameDef) {
        const auto it = replacement_defs->find(down_cast<const NameDef*>(def));
        if (it != replacement_defs->end()) {
          return original_node->owner()->Make<NameRef>(
              original_ref->span(), original_ref->identifier(), it->second);
        }
      }
    }
    return std::nullopt;
  };
}

absl::StatusOr<absl::flat_hash_map<const AstNode*, AstNode*>>
CloneAstAndGetAllPairs(const AstNode* root, CloneReplacer replacer) {
  if (root->kind() == AstNodeKind::kModule) {
    return absl::InvalidArgumentError("Clone a module via 'CloneModule'.");
  }
  XLS_ASSIGN_OR_RETURN(std::optional<AstNode*> root_replacement,
                       replacer(root));
  if (root_replacement.has_value()) {
    return absl::flat_hash_map<const AstNode*, AstNode*>{
        {root, *root_replacement}};
  }
  AstCloner cloner(root->owner(), std::move(replacer));
  XLS_RETURN_IF_ERROR(root->Accept(&cloner));
  return cloner.old_to_new();
}

absl::StatusOr<AstNode*> CloneAst(const AstNode* root, CloneReplacer replacer) {
  absl::flat_hash_map<const AstNode*, AstNode*> all_pairs;
  XLS_ASSIGN_OR_RETURN(all_pairs,
                       CloneAstAndGetAllPairs(root, std::move(replacer)));
  return all_pairs.at(root);
}

absl::StatusOr<std::unique_ptr<Module>> CloneModule(const Module& module,
                                                    CloneReplacer replacer) {
  auto new_module = std::make_unique<Module>(module.name(), module.fs_path(),
                                             *module.file_table());
  std::optional<Span> attribute_span = module.GetAttributeSpan();
  for (const ModuleAttribute& dir : module.attributes()) {
    new_module->AddAttribute(dir, attribute_span);
  }
  AstCloner cloner(new_module.get(), std::move(replacer));
  XLS_RETURN_IF_ERROR(module.Accept(&cloner));
  return new_module;
}

CloneReplacer ChainCloneReplacers(CloneReplacer first, CloneReplacer second) {
  return [first = std::move(first),
          second = std::move(second)](const AstNode* node) mutable
             -> absl::StatusOr<std::optional<AstNode*>> {
    XLS_ASSIGN_OR_RETURN(std::optional<AstNode*> first_result, first(node));
    XLS_ASSIGN_OR_RETURN(
        std::optional<AstNode*> second_result,
        second(first_result.has_value() ? *first_result : node));
    return second_result.has_value() ? second_result : first_result;
  };
}

// Verifies that `node` consists solely of "new" AST nodes and none that are
// in the "old" set.
absl::Status VerifyClone(const AstNode* old_root, const AstNode* new_root,
                         const FileTable& file_table) {
  absl::flat_hash_set<const AstNode*> old_nodes = FlattenToSet(old_root);
  absl::flat_hash_set<const AstNode*> new_nodes = FlattenToSet(new_root);
  for (const AstNode* new_node : new_nodes) {
    if (old_nodes.contains(new_node)) {
      std::optional<Span> span = new_node->GetSpan();
      return absl::InvalidArgumentError(absl::StrFormat(
          "Node \"%s\" (%s; %s) was found in both the old set and "
          "new translation!",
          new_node->ToString(),
          span.has_value() ? span.value().ToString(file_table) : "<no span>",
          new_node->GetNodeTypeName()));
    }
  }
  return absl::OkStatus();
}

}  // namespace xls::dslx
