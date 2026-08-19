// Copyright 2026 The XLS Authors
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

#include "xls/dslx/fmt/legacy_proc_converter.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {

namespace {

// Returns true if `stmt` is pointless as the last statement in a stateless
// legacy proc next function. Note that a "stateless legacy proc" may be one
// with a nominal empty-tuple state parameter, in which case next() may validly
// return that empty tuple by name.
bool IsRedundantStatelessNextReturn(const Statement* stmt,
                                    std::string_view state_param_name) {
  if (!std::holds_alternative<Expr*>(stmt->wrapped())) {
    return false;
  }
  const Expr* expr = std::get<Expr*>(stmt->wrapped());
  if (auto* tuple = dynamic_cast<const XlsTuple*>(expr); tuple != nullptr) {
    return tuple->empty();
  }
  if (auto* name_ref = dynamic_cast<const NameRef*>(expr);
      name_ref != nullptr) {
    return name_ref->identifier() == state_param_name;
  }
  return false;
}

absl::StatusOr<bool> HasReferenceToAnyName(
    const AstNode* node, const absl::flat_hash_set<std::string_view>& names) {
  if (node == nullptr) {
    return false;
  }
  if (auto* name_ref = dynamic_cast<const NameRef*>(node)) {
    if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
      const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
      if (names.contains(def->identifier())) {
        return true;
      }
    }
  }
  if (auto* type_ref_annot = dynamic_cast<const TypeRefTypeAnnotation*>(node)) {
    const TypeRef* type_ref = type_ref_annot->type_ref();
    if (type_ref != nullptr) {
      TypeDefinition type_def = type_ref->type_definition();
      if (std::holds_alternative<TypeAlias*>(type_def)) {
        const TypeAlias* alias = std::get<TypeAlias*>(type_def);
        if (names.contains(alias->identifier())) {
          return true;
        }
      }
    }
  }
  for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
    XLS_ASSIGN_OR_RETURN(bool has_ref, HasReferenceToAnyName(child, names));
    if (has_ref) {
      return true;
    }
  }
  return false;
}

void GatherReferencedLocalTypeAliases(
    const AstNode* node,
    const absl::flat_hash_set<std::string_view>& local_type_alias_names,
    absl::flat_hash_set<std::string_view>& gathered) {
  if (node == nullptr) {
    return;
  }
  if (auto* name_ref = dynamic_cast<const NameRef*>(node);
      name_ref != nullptr &&
      std::holds_alternative<const NameDef*>(name_ref->name_def())) {
    const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
    if (local_type_alias_names.contains(def->identifier())) {
      gathered.insert(def->identifier());
    }
  }
  if (auto* type_ref_annot = dynamic_cast<const TypeRefTypeAnnotation*>(node)) {
    const TypeRef* type_ref = type_ref_annot->type_ref();
    if (type_ref != nullptr) {
      TypeDefinition type_def = type_ref_annot->type_ref()->type_definition();
      if (std::optional<std::string_view> id = GetIdentifier(type_def);
          id.has_value() && local_type_alias_names.contains(*id)) {
        gathered.insert(*id);
      }
    }
  }
  for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
    GatherReferencedLocalTypeAliases(child, local_type_alias_names, gathered);
  }
}

bool HasExplicitStateAccess(const AstNode* node) {
  if (auto* invocation = dynamic_cast<const Invocation*>(node)) {
    if (auto* name_ref = dynamic_cast<const NameRef*>(invocation->callee());
        name_ref != nullptr && (name_ref->identifier() == "read" ||
                                name_ref->identifier() == "write")) {
      return true;
    }
  }
  for (AstNode* child : node->GetChildren(/*want_types=*/false)) {
    if (HasExplicitStateAccess(child)) {
      return true;
    }
  }
  return false;
}

bool IsLiteralEmptyTuple(const Statement* stmt) {
  if (std::holds_alternative<Expr*>(stmt->wrapped())) {
    const Expr* expr = std::get<Expr*>(stmt->wrapped());
    if (auto* tuple = dynamic_cast<const XlsTuple*>(expr)) {
      return tuple->empty();
    }
  }
  return false;
}

bool FunctionDoesAnything(const Function& fn) {
  const auto& stmts = fn.body()->statements();
  if (stmts.empty()) {
    return false;
  }
  if (stmts.size() == 1) {
    if (std::holds_alternative<Expr*>(stmts[0]->wrapped())) {
      const Expr* expr = std::get<Expr*>(stmts[0]->wrapped());
      if (auto* tuple = dynamic_cast<const XlsTuple*>(expr);
          tuple != nullptr && tuple->empty()) {
        return false;
      }
    }
  }
  return true;
}

class LegacyProcConverter : public Formatter {
 public:
  using Formatter::Formatter;

  // Adds the explicit state access and generics feature flags to the module
  // if not already present, because impl-style procs require them.
  absl::StatusOr<DocRef> FormatModule(const Module& n) override {
    Module& mutable_n = const_cast<Module&>(n);
    if (!n.attributes().contains(ModuleAttribute::kExplicitStateAccess)) {
      mutable_n.AddAttribute(ModuleAttribute::kExplicitStateAccess,
                             std::nullopt);
    }
    if (!n.attributes().contains(ModuleAttribute::kGenerics)) {
      mutable_n.AddAttribute(ModuleAttribute::kGenerics, std::nullopt);
    }

    XLS_ASSIGN_OR_RETURN(DocRef doc, Formatter::FormatModule(n));
    if (!status_.ok()) {
      return status_;
    }
    return doc;
  }

 protected:
  bool IsBlockedExprWithLeader(const Expr& e) override {
    if (e.kind() == AstNodeKind::kSpawn) {
      return false;
    }
    return Formatter::IsBlockedExprWithLeader(e);
  }

  // Formats a legacy proc structure into an impl-style proc structure.
  //
  // Before:
  //   proc Foo {
  //       x: chan<u32> in;
  //       config(x: chan<u32> in) { (x,) }
  //       init { () }
  //       next(state: ()) { ... }
  //   }
  //
  // After:
  //   proc Foo {
  //       x: chan<u32> in,
  //   }
  //   impl Foo {
  //       fn new(x: chan<u32> in) -> Self { Foo { x } }
  //       fn next(self) { ... }
  //   }
  DocRef FormatProc(const Proc& n, bool is_test = false) override {
    if (!status_.ok()) {
      return arena_.empty();
    }

    local_constants_.clear();
    local_type_aliases_.clear();

    std::vector<const AstNode*> proc_level_decls;
    proc_level_decls.reserve(n.stmts().size());
    std::vector<const ProcMember*> members;
    members.reserve(n.stmts().size());
    absl::Status status =
        AnalyzeAndSplitProcStatements(n, proc_level_decls, members);
    if (!status.ok()) {
      status_ = status;
      return arena_.empty();
    }

    std::vector<const Param*> state_params;
    state_params.reserve(n.next().params().size());
    for (const Param* param : n.next().params()) {
      if (auto* tuple_type = dynamic_cast<const TupleTypeAnnotation*>(
              param->type_annotation());
          !tuple_type || !tuple_type->empty()) {
        state_params.push_back(param);
      }
    }

    absl::flat_hash_set<std::string_view> local_constant_names;
    for (const auto& [name, _] : local_constants_) {
      local_constant_names.insert(name);
    }
    for (const Param* state_param : state_params) {
      auto has_ref = HasReferenceToAnyName(state_param->type_annotation(),
                                           local_constant_names);
      if (!has_ref.ok()) {
        status_ = has_ref.status();
        return arena_.empty();
      }
      if (has_ref.value()) {
        status_ = absl::InvalidArgumentError(absl::StrFormat(
            "Proc state parameter `%s` references a constant declared inside "
            "the proc, which is not allowed in impl-style procs.",
            state_param->identifier()));
        return arena_.empty();
      }
    }
    for (const ProcMember* member : members) {
      auto has_ref = HasReferenceToAnyName(member->type_annotation(),
                                           local_constant_names);
      if (!has_ref.ok()) {
        status_ = has_ref.status();
        return arena_.empty();
      }
      if (has_ref.value()) {
        status_ = absl::InvalidArgumentError(absl::StrFormat(
            "Proc member `%s` references a constant declared inside "
            "the proc",
            member->identifier()));
        return arena_.empty();
      }
    }

    // Find needed type aliases.
    absl::flat_hash_set<std::string_view> local_type_alias_names;
    for (auto const& [name, _] : local_type_aliases_) {
      local_type_alias_names.insert(name);
    }
    absl::flat_hash_set<std::string_view> needed_type_aliases;
    for (const ProcMember* member : members) {
      GatherReferencedLocalTypeAliases(member->type_annotation(),
                                       local_type_alias_names,
                                       needed_type_aliases);
    }
    for (const Param* param : state_params) {
      GatherReferencedLocalTypeAliases(param->type_annotation(),
                                       local_type_alias_names,
                                       needed_type_aliases);
    }

    // Transitive closure for needed type aliases.
    std::vector<std::string_view> to_process(needed_type_aliases.begin(),
                                             needed_type_aliases.end());
    while (!to_process.empty()) {
      std::string_view current = to_process.back();
      to_process.pop_back();
      const TypeAlias* alias = local_type_aliases_.at(current);
      absl::flat_hash_set<std::string_view> deps;
      GatherReferencedLocalTypeAliases(&alias->type_annotation(),
                                       local_type_alias_names, deps);
      for (std::string_view dep : deps) {
        if (needed_type_aliases.insert(dep).second) {
          to_process.push_back(dep);
        }
      }
    }

    // Process needed type aliases and filter proc_level_decls.
    std::vector<ParametricBinding*> additional_parametrics;
    std::vector<const AstNode*> remaining_proc_level_decls;
    additional_parametrics.reserve(proc_level_decls.size());
    remaining_proc_level_decls.reserve(proc_level_decls.size());

    for (const AstNode* node : proc_level_decls) {
      if (auto* t = dynamic_cast<const TypeAlias*>(node)) {
        if (needed_type_aliases.contains(t->identifier())) {
          auto param_status = ProcessNeededTypeAlias(*t, n.owner());
          if (!param_status.ok()) {
            status_ = param_status.status();
            return arena_.empty();
          }
          additional_parametrics.push_back(param_status.value());
          continue;
        }
      }
      remaining_proc_level_decls.push_back(node);
    }

    bool already_has_explicit_state_access =
        !state_params.empty() && HasExplicitStateAccess(n.next().body());

    Pos last_stmt_limit = members.empty() ? n.body_span().start()
                                          : members.back()->span().limit();
    ProcDef* proc_def = CreateSyntheticProcDef(n, is_test, state_params,
                                               members, additional_parametrics);
    Function* new_fn =
        CreateSyntheticNewFunction(n, proc_def, state_params, members);
    Function* next_fn = CreateSyntheticNextFunction(
        n, proc_def, already_has_explicit_state_access, state_params);
    Impl* impl = CreateSyntheticImpl(
        n, proc_def, new_fn, next_fn, remaining_proc_level_decls,
        proc_def->parametric_bindings(), last_stmt_limit);

    std::optional<DocRef> init_comments;
    if (n.config().span().start() < n.init().span().start()) {
      init_comments = FormatCommentsBetween(n.config().span().limit(),
                                            n.init().span().limit());
    }
    current_init_comments_ = init_comments;

    DocRef proc_decl_doc = Formatter::FormatProcDef(*proc_def);
    DocRef impl_block_doc = Formatter::FormatImpl(*impl);

    std::vector<DocRef> final_pieces{proc_decl_doc, arena_.hard_line(),
                                     arena_.hard_line(), impl_block_doc};

    current_proc_member_names_ = std::nullopt;
    local_constants_.clear();
    local_type_aliases_.clear();
    return ConcatN(arena_, final_pieces);
  }

  // Formats a test proc structure. Test procs preserve the `#[test]` attribute
  // on their proc block and have their config/init/next converted.
  //
  // Before:
  //   #[test]
  //   proc MyTest {
  //       ...
  //   }
  //
  // After:
  //   #[test]
  //   proc MyTest {
  //       ...
  //   }
  //   impl MyTest {
  //       ...
  //   }
  DocRef FormatTestProc(const TestProc& n) override {
    if (!status_.ok()) {
      return arena_.empty();
    }
    std::vector<DocRef> pieces;
    pieces.reserve(3);
    if (n.expected_fail_label().has_value()) {
      pieces.push_back(
          arena_.MakeText(absl::StrFormat("#[test(expected_fail_label=\"%s\")]",
                                          n.expected_fail_label().value())));
    } else {
      pieces.push_back(arena_.MakeText("#[test]"));
    }
    pieces.push_back(arena_.hard_line());
    pieces.push_back(FormatProc(*n.proc(), /*is_test=*/true));
    return ConcatN(arena_, pieces);
  }

  // Formats a legacy `spawn` statement by creating a synthetic AST Invocation
  // chain `Callee::new(...).spawn()` and delegating to FormatInvocation.
  //
  // Before:
  //   spawn MyProc(a, b)
  //
  // After:
  //   MyProc::new(a, b).spawn()
  DocRef FormatSpawn(const Spawn& n) override {
    Module* owner = n.owner();
    Span span = n.span();

    ColonRef::Subject subject;
    if (!n.explicit_parametrics().empty()) {
      TypeRef* type_ref = nullptr;
      if (auto* name_ref = dynamic_cast<const NameRef*>(n.callee())) {
        auto* name_def =
            owner->Make<NameDef>(span, name_ref->identifier(), nullptr);
        auto* proc_def = owner->Make<ProcDef>(
            span, name_def, std::vector<ParametricBinding*>{},
            std::vector<StructMemberNode*>{}, /*is_public=*/false);
        type_ref = owner->Make<TypeRef>(span, proc_def);
      } else if (auto* colon_ref = dynamic_cast<const ColonRef*>(n.callee())) {
        type_ref = owner->Make<TypeRef>(span, const_cast<ColonRef*>(colon_ref));
      }
      if (type_ref != nullptr) {
        subject = owner->Make<TypeRefTypeAnnotation>(span, type_ref,
                                                     n.explicit_parametrics());
      } else {
        subject = owner->Make<NameRef>(span, n.callee()->ToString(),
                                       static_cast<const NameDef*>(nullptr));
      }
    } else {
      if (auto* name_ref = dynamic_cast<const NameRef*>(n.callee())) {
        subject = const_cast<NameRef*>(name_ref);
      } else if (auto* colon_ref = dynamic_cast<const ColonRef*>(n.callee())) {
        subject = const_cast<ColonRef*>(colon_ref);
      } else {
        subject = owner->Make<NameRef>(span, n.callee()->ToString(),
                                       static_cast<const NameDef*>(nullptr));
      }
    }

    auto* new_colon_ref = owner->Make<ColonRef>(span, subject, "new");
    std::vector<Expr*> config_args(n.config()->args().begin(),
                                   n.config()->args().end());
    auto* new_invocation = owner->Make<Invocation>(
        span, new_colon_ref, std::move(config_args),
        /*explicit_parametrics=*/std::vector<ExprOrType>{});
    auto* dot_spawn = owner->Make<Attr>(span, new_invocation, "spawn");
    auto* spawn_invocation = owner->Make<Invocation>(
        span, dot_spawn, std::vector<Expr*>{}, std::vector<ExprOrType>{});
    return Formatter::FormatInvocation(*spawn_invocation);
  }

 private:
  absl::Status AnalyzeAndSplitProcStatements(
      const Proc& n, std::vector<const AstNode*>& proc_level_decls,
      std::vector<const ProcMember*>& members) {
    for (const ProcStmt& stmt : n.stmts()) {
      absl::Status visit_status = std::visit(
          Visitor{
              [&](const Function* f) { return absl::OkStatus(); },
              [&](const ProcMember* m) {
                members.push_back(m);
                return absl::OkStatus();
              },
              [&](const ConstantDef* c) {
                proc_level_decls.push_back(c);
                local_constants_[c->identifier()] = c;
                return absl::OkStatus();
              },
              [&](const TypeAlias* t) {
                proc_level_decls.push_back(t);
                local_type_aliases_[t->identifier()] = t;
                return absl::OkStatus();
              },
              [&](const ConstAssert* ca) {
                return absl::InvalidArgumentError(
                    "Const asserts inside a proc are not supported in "
                    "impl-style procs.");
              },
          },
          stmt);
      if (!visit_status.ok()) {
        return visit_status;
      }
    }

    absl::flat_hash_set<std::string> member_names;
    member_names.reserve(members.size());
    for (const ProcMember* m : members) {
      member_names.insert(m->identifier());
    }
    current_proc_member_names_ = std::move(member_names);

    return absl::OkStatus();
  }

  absl::StatusOr<ParametricBinding*> ProcessNeededTypeAlias(const TypeAlias& t,
                                                            Module* owner) {
    std::function<absl::StatusOr<std::optional<AstNode*>>(
        const AstNode*, Module*,
        const absl::flat_hash_map<const AstNode*, AstNode*>&)>
        replacer;
    replacer =
        [&](const AstNode* node, Module* module,
            const absl::flat_hash_map<const AstNode*, AstNode*>& mappings)
        -> absl::StatusOr<std::optional<AstNode*>> {
      if (auto* name_ref = dynamic_cast<const NameRef*>(node)) {
        if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
          const NameDef* def = std::get<const NameDef*>(name_ref->name_def());
          auto it = local_constants_.find(def->identifier());
          if (it != local_constants_.end()) {
            XLS_ASSIGN_OR_RETURN(
                Expr * cloned_val,
                CloneNode<Expr>(const_cast<Expr*>(it->second->value()),
                                replacer));
            return cloned_val;
          }
        }
      }
      return std::nullopt;
    };

    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * substituted_rhs,
        CloneNode<TypeAnnotation>(
            const_cast<TypeAnnotation*>(&t.type_annotation()), replacer));

    NameDef* name_def = owner->Make<NameDef>(t.span(), t.identifier(), nullptr);
    GenericTypeAnnotation* type_annot =
        owner->Make<GenericTypeAnnotation>(t.span());
    return owner->Make<ParametricBinding>(name_def, type_annot,
                                          ExprOrType(substituted_rhs));
  }

  ProcDef* CreateSyntheticProcDef(
      const Proc& n, bool is_test, absl::Span<const Param* const> state_params,
      absl::Span<const ProcMember* const> members,
      absl::Span<ParametricBinding* const> additional_parametrics) {
    Module* owner = n.owner();
    Pos members_limit = members.empty() ? n.body_span().start()
                                        : members.back()->span().limit();
    Span synthetic_span(n.span().start(), members_limit);

    std::vector<StructMemberNode*> struct_members;
    struct_members.reserve(members.size() + state_params.size());
    for (const ProcMember* member : members) {
      auto* name_def =
          owner->Make<NameDef>(member->span(), member->identifier(), nullptr);
      struct_members.push_back(owner->Make<StructMemberNode>(
          member->span(), name_def, member->span(),
          const_cast<TypeAnnotation*>(member->type_annotation())));
    }
    for (const Param* state_param : state_params) {
      Span state_span(members_limit, members_limit);
      auto* name_def =
          owner->Make<NameDef>(state_span, state_param->identifier(), nullptr);
      struct_members.push_back(owner->Make<StructMemberNode>(
          state_span, name_def, state_span,
          const_cast<TypeAnnotation*>(state_param->type_annotation())));
    }

    std::vector<ParametricBinding*> all_parametric_bindings;
    all_parametric_bindings.reserve(n.parametric_bindings().size() +
                                    additional_parametrics.size());
    for (const ParametricBinding* pb : n.parametric_bindings()) {
      all_parametric_bindings.push_back(const_cast<ParametricBinding*>(pb));
    }
    for (ParametricBinding* pb : additional_parametrics) {
      all_parametric_bindings.push_back(pb);
    }

    auto* name_def = owner->Make<NameDef>(n.span(), n.identifier(), nullptr);
    auto* proc_def =
        owner->Make<ProcDef>(synthetic_span, name_def, all_parametric_bindings,
                             struct_members, n.is_public());
    if (n.is_test_utility() && !is_test) {
      proc_def->AddAttribute(owner->Make<Attribute>(
          n.span(), std::nullopt,
          AttributeData(AttributeKind::kCfg, {std::string("test")})));
    }
    return proc_def;
  }

  Function* CreateSyntheticNewFunction(
      const Proc& n, ProcDef* proc_def,
      absl::Span<const Param* const> state_params,
      absl::Span<const ProcMember* const> members) {
    Module* owner = n.owner();
    Span span = n.config().span();

    const XlsTuple* config_tuple = nullptr;
    if (!n.config().body()->statements().empty()) {
      const Statement* last_config_stmt =
          n.config().body()->statements().back();
      if (std::holds_alternative<Expr*>(last_config_stmt->wrapped())) {
        const Expr* config_expr = std::get<Expr*>(last_config_stmt->wrapped());
        config_tuple = dynamic_cast<const XlsTuple*>(config_expr);
      }
    }
    if (!members.empty()) {
      CHECK(config_tuple != nullptr);
      CHECK_EQ(config_tuple->members().size(), members.size());
    }

    const Expr* init_expr = nullptr;
    const StatementBlock* init_body = n.init().body();
    const auto& init_stmts = init_body->statements();
    if (init_stmts.size() == 1) {
      init_expr = std::get<Expr*>(init_stmts[0]->wrapped());
    } else if (init_body != nullptr && !init_body->empty() &&
               !init_body->trailing_semi()) {
      const Statement* last_stmt = init_body->statements().back();
      if (std::holds_alternative<Expr*>(last_stmt->wrapped())) {
        init_expr = std::get<Expr*>(last_stmt->wrapped());
      }
    }

    const XlsTuple* init_tuple = nullptr;
    if (init_expr != nullptr) {
      init_tuple = dynamic_cast<const XlsTuple*>(init_expr);
    }

    bool init_yields_tuple_per_state_param =
        !state_params.empty() && state_params.size() > 1 &&
        init_tuple != nullptr &&
        init_tuple->members().size() == state_params.size();

    std::vector<Statement*> new_statements;
    const auto& config_stmts = n.config().body()->statements();
    int config_end_idx;
    if (members.empty()) {
      if (!config_stmts.empty() && IsLiteralEmptyTuple(config_stmts.back())) {
        config_end_idx = config_stmts.size() - 1;
      } else {
        config_end_idx = config_stmts.size();
      }
    } else {
      config_end_idx = config_stmts.empty() ? 0 : config_stmts.size() - 1;
    }
    new_statements.reserve(config_end_idx + init_stmts.size() + 1);
    for (int i = 0; i < config_end_idx; ++i) {
      new_statements.push_back(config_stmts[i]);
    }

    Pos body_start = (config_end_idx == 0 && init_stmts.size() > 1)
                         ? n.init().body()->span().start().BumpCol()
                         : n.config().body()->span().start().BumpCol();
    Pos prev_pos = new_statements.empty()
                       ? body_start
                       : new_statements.back()->GetSpan()->limit();

    if (init_stmts.size() > 1) {
      for (size_t i = 0; i < init_stmts.size() - 1; ++i) {
        new_statements.push_back(init_stmts[i]);
      }
      prev_pos = new_statements.back()->GetSpan()->limit();
    }

    if (!state_params.empty() && state_params.size() > 1) {
      if (!init_yields_tuple_per_state_param) {
        auto* name_def = owner->Make<NameDef>(Span(prev_pos, prev_pos),
                                              "init_state", nullptr);
        auto* let_stmt =
            owner->Make<Let>(Span(prev_pos, prev_pos), name_def, nullptr,
                             const_cast<Expr*>(init_expr), /*is_const=*/false);
        new_statements.push_back(owner->Make<Statement>(let_stmt));
        prev_pos = let_stmt->GetSpan()->limit();
      }
    }

    std::vector<std::pair<std::string, Expr*>> struct_members;
    struct_members.reserve(members.size() + state_params.size());
    if (config_tuple != nullptr) {
      for (int i = 0; i < members.size(); ++i) {
        struct_members.push_back({std::string(members[i]->identifier()),
                                  config_tuple->members()[i]});
      }
    }
    if (!state_params.empty()) {
      if (state_params.size() == 1) {
        struct_members.push_back({std::string(state_params[0]->identifier()),
                                  const_cast<Expr*>(init_expr)});
      } else if (init_yields_tuple_per_state_param) {
        for (int i = 0; i < state_params.size(); ++i) {
          struct_members.push_back({std::string(state_params[i]->identifier()),
                                    init_tuple->members()[i]});
        }
      } else {
        for (int i = 0; i < state_params.size(); ++i) {
          Span tuple_span(prev_pos, prev_pos);
          auto* name_ref = owner->Make<NameRef>(
              tuple_span, "init_state", static_cast<const NameDef*>(nullptr));
          auto* num = owner->Make<Number>(tuple_span, std::to_string(i),
                                          NumberKind::kOther, nullptr);
          auto* tuple_idx = owner->Make<TupleIndex>(tuple_span, name_ref, num);
          struct_members.push_back(
              {std::string(state_params[i]->identifier()), tuple_idx});
        }
      }
    }

    Span struct_span(prev_pos, prev_pos);
    auto* struct_type_ref = owner->Make<TypeRef>(struct_span, proc_def);
    auto* struct_type_annot = owner->Make<TypeRefTypeAnnotation>(
        struct_span, struct_type_ref, std::vector<ExprOrType>{});
    auto* struct_instance = owner->Make<StructInstance>(
        struct_span, struct_type_annot, struct_members);
    new_statements.push_back(owner->Make<Statement>(struct_instance));

    Pos body_start_pos = (config_end_idx == 0 && init_stmts.size() > 1)
                             ? n.init().body()->span().start()
                             : n.config().body()->span().start();
    Pos body_limit_pos = n.config().body()->span().limit();
    if (body_start_pos > body_limit_pos) {
      body_limit_pos = body_start_pos;
    }
    Span body_span(body_start_pos, body_limit_pos);
    auto* new_body = owner->Make<StatementBlock>(body_span, new_statements,
                                                 /*trailing_semi=*/false);
    auto* return_type = owner->Make<SelfTypeAnnotation>(span, false, nullptr);
    auto* fn_name_def = owner->Make<NameDef>(span, "new", nullptr);
    return owner->Make<Function>(
        span, fn_name_def, std::vector<ParametricBinding*>{},
        n.config().params(), return_type, new_body, FunctionTag::kNormal,
        /*is_public=*/false, /*is_stub=*/false);
  }

  Function* CreateSyntheticNextFunction(
      const Proc& n, ProcDef* proc_def, bool already_has_explicit_state_access,
      absl::Span<const Param* const> state_params) {
    const Function& next_fn = n.next();
    Pos last_before_next_limit =
        std::max(n.config().span().limit(), n.init().span().limit());
    if (!(!state_params.empty() || FunctionDoesAnything(next_fn) ||
          (last_before_next_limit <= next_fn.span().limit() &&
           comments_.HasComments(
               Span(last_before_next_limit, next_fn.span().limit()))))) {
      return nullptr;
    }

    Module* owner = n.owner();
    Span span = next_fn.span();
    Pos body_start = next_fn.body()->span().start().BumpCol();
    Span start_span(body_start, body_start);

    auto replacer = [&](const AstNode* node, Module* module,
                        const absl::flat_hash_map<const AstNode*, AstNode*>&)
        -> absl::StatusOr<std::optional<AstNode*>> {
      if (auto* name_ref = dynamic_cast<const NameRef*>(node);
          name_ref != nullptr) {
        bool is_member =
            current_proc_member_names_.has_value() &&
            current_proc_member_names_->contains(name_ref->identifier());
        bool is_state_param = false;
        if (already_has_explicit_state_access) {
          for (const Param* state_param : state_params) {
            if (name_ref->identifier() == state_param->identifier()) {
              is_state_param = true;
              break;
            }
          }
        }
        if (is_member || is_state_param) {
          auto* self_ref = module->Make<NameRef>(
              name_ref->span(), "self", static_cast<const NameDef*>(nullptr));
          return module->Make<Attr>(name_ref->span(), self_ref,
                                    name_ref->identifier());
        }
      }
      return std::nullopt;
    };

    absl::StatusOr<StatementBlock*> cloned_next_body_status =
        CloneNode<StatementBlock>(const_cast<StatementBlock*>(next_fn.body()),
                                  replacer);
    CHECK_OK(cloned_next_body_status.status());
    StatementBlock* cloned_next_body = cloned_next_body_status.value();

    const auto& next_stmts = cloned_next_body->statements();
    std::string_view state_param_name =
        next_fn.params().empty()
            ? std::string_view("")
            : std::string_view(next_fn.params()[0]->identifier());
    const bool has_redundant_stateless_return =
        state_params.empty() && !next_stmts.empty() &&
        IsRedundantStatelessNextReturn(next_stmts.back(), state_param_name);

    bool should_drop_last = false;
    if (!already_has_explicit_state_access && !next_stmts.empty()) {
      should_drop_last =
          !state_params.empty() || has_redundant_stateless_return;
    }
    int end_idx = should_drop_last ? next_stmts.size() - 1 : next_stmts.size();

    bool is_empty_body = false;
    if (state_params.empty()) {
      if (next_stmts.empty() ||
          (next_stmts.size() == 1 && has_redundant_stateless_return)) {
        is_empty_body = true;
      }
    }

    NameDef* self_name_def = owner->Make<NameDef>(span, "self", nullptr);
    TypeAnnotation* self_type =
        owner->Make<SelfTypeAnnotation>(span, false, nullptr);
    Param* self_param = owner->Make<Param>(self_name_def, self_type);

    std::vector<Statement*> statements;
    size_t state_stmt_count =
        already_has_explicit_state_access || state_params.empty()
            ? 0
            : (state_params.size() +
               (state_params.size() > 1 ? 1 + state_params.size() : 1));
    statements.reserve(state_stmt_count + end_idx);
    if (!state_params.empty() && !already_has_explicit_state_access) {
      for (const Param* state_param : state_params) {
        auto* self_ref = owner->Make<NameRef>(
            start_span, "self", static_cast<const NameDef*>(self_name_def));
        auto* self_attr =
            owner->Make<Attr>(start_span, self_ref, state_param->identifier());
        auto* read_ref = owner->Make<NameRef>(
            start_span, "read", static_cast<const NameDef*>(nullptr));
        auto* read_invoc = owner->Make<Invocation>(
            start_span, read_ref, std::vector<Expr*>{self_attr});
        auto* let_name = owner->Make<NameDef>(
            start_span, state_param->identifier(), nullptr);
        auto* let_stmt =
            owner->Make<Let>(start_span, let_name, nullptr, read_invoc, false);
        statements.push_back(owner->Make<Statement>(let_stmt));
      }
    }

    if (!is_empty_body) {
      for (int i = 0; i < end_idx; ++i) {
        statements.push_back(next_stmts[i]);
      }
      if (!state_params.empty() && !already_has_explicit_state_access) {
        CHECK(!next_stmts.empty());
        const Expr* final_expr = std::get<Expr*>(next_stmts.back()->wrapped());
        Pos prev_pos = statements.empty()
                           ? body_start
                           : statements.back()->GetSpan()->limit();
        if (state_params.size() == 1) {
          Span write_span(prev_pos, prev_pos);
          auto* self_ref = owner->Make<NameRef>(
              write_span, "self", static_cast<const NameDef*>(self_name_def));
          auto* self_attr = owner->Make<Attr>(write_span, self_ref,
                                              state_params[0]->identifier());
          auto* write_ref = owner->Make<NameRef>(
              write_span, "write", static_cast<const NameDef*>(nullptr));
          auto* write_invoc = owner->Make<Invocation>(
              write_span, write_ref,
              std::vector<Expr*>{self_attr, const_cast<Expr*>(final_expr)});
          statements.push_back(owner->Make<Statement>(write_invoc));
        } else {
          Span let_span(prev_pos, prev_pos);
          auto* next_state_name =
              owner->Make<NameDef>(let_span, "next_state", nullptr);
          auto* let_stmt =
              owner->Make<Let>(let_span, next_state_name, nullptr,
                               const_cast<Expr*>(final_expr), false);
          statements.push_back(owner->Make<Statement>(let_stmt));
          prev_pos = let_stmt->GetSpan()->limit();
          Span end_span(prev_pos, prev_pos);
          for (int i = 0; i < state_params.size(); ++i) {
            auto* self_ref = owner->Make<NameRef>(
                end_span, "self", static_cast<const NameDef*>(self_name_def));
            auto* self_attr = owner->Make<Attr>(end_span, self_ref,
                                                state_params[i]->identifier());
            auto* next_state_ref = owner->Make<NameRef>(
                end_span, "next_state",
                static_cast<const NameDef*>(next_state_name));
            auto* num = owner->Make<Number>(end_span, std::to_string(i),
                                            NumberKind::kOther, nullptr);
            auto* tuple_idx =
                owner->Make<TupleIndex>(end_span, next_state_ref, num);
            auto* write_ref = owner->Make<NameRef>(
                end_span, "write", static_cast<const NameDef*>(nullptr));
            auto* write_invoc = owner->Make<Invocation>(
                end_span, write_ref, std::vector<Expr*>{self_attr, tuple_idx});
            statements.push_back(owner->Make<Statement>(write_invoc));
          }
        }
      }
    }

    auto* next_body =
        owner->Make<StatementBlock>(span, statements, /*trailing_semi=*/true);
    auto* next_name_def = owner->Make<NameDef>(span, "next", nullptr);
    return owner->Make<Function>(
        span, next_name_def, std::vector<ParametricBinding*>{},
        std::vector<Param*>{self_param}, nullptr, next_body,
        FunctionTag::kNormal, /*is_public=*/false, /*is_stub=*/false);
  }

  DocRef FormatFunction(const Function& n, bool is_test = false) override {
    if (current_init_comments_.has_value() && n.identifier() == "new") {
      DocRef comments = *current_init_comments_;
      current_init_comments_ = std::nullopt;
      return ConcatN(arena_, {comments, arena_.hard_line(),
                              Formatter::FormatFunction(n, is_test)});
    }
    return Formatter::FormatFunction(n, is_test);
  }

  Impl* CreateSyntheticImpl(
      const Proc& n, ProcDef* proc_def, Function* new_fn, Function* next_fn,
      absl::Span<const AstNode* const> remaining_proc_level_decls,
      absl::Span<ParametricBinding* const> all_parametric_bindings,
      Pos last_stmt_limit) {
    Module* owner = n.owner();
    Span span(last_stmt_limit, n.span().limit());

    TypeRef* type_ref = owner->Make<TypeRef>(span, proc_def);
    std::vector<ExprOrType> impl_parametrics;
    impl_parametrics.reserve(all_parametric_bindings.size());
    for (const ParametricBinding* pb : all_parametric_bindings) {
      impl_parametrics.push_back(ExprOrType(
          owner->Make<NameRef>(span, pb->identifier(),
                               static_cast<const NameDef*>(pb->name_def()))));
    }
    TypeRefTypeAnnotation* struct_ref =
        owner->Make<TypeRefTypeAnnotation>(span, type_ref, impl_parametrics);

    std::vector<ImplMember> members;
    members.reserve(remaining_proc_level_decls.size() + 2);
    for (const AstNode* node : remaining_proc_level_decls) {
      if (auto* c = dynamic_cast<const ConstantDef*>(node)) {
        members.push_back(const_cast<ConstantDef*>(c));
      } else if (auto* t = dynamic_cast<const TypeAlias*>(node)) {
        members.push_back(const_cast<TypeAlias*>(t));
      }
    }
    members.push_back(new_fn);
    if (next_fn != nullptr) {
      members.push_back(next_fn);
    }

    auto* impl =
        owner->Make<Impl>(span, struct_ref, members, /*is_public=*/false);
    new_fn->set_impl(impl);
    if (next_fn != nullptr) {
      next_fn->set_impl(impl);
    }
    return impl;
  }

  absl::flat_hash_map<std::string_view, const ConstantDef*> local_constants_;
  absl::flat_hash_map<std::string_view, const TypeAlias*> local_type_aliases_;
  std::optional<absl::flat_hash_set<std::string>> current_proc_member_names_;
  std::optional<DocRef> current_init_comments_;
  absl::Status status_ = absl::OkStatus();
};

}  // namespace

std::unique_ptr<Formatter> CreateLegacyProcConverter(Comments& comments,
                                                     DocArena& arena) {
  return std::make_unique<LegacyProcConverter>(comments, arena);
}

}  // namespace xls::dslx
