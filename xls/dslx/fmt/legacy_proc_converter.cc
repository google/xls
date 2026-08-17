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

  // Adds the explicit state access feature flag to the module if not already
  // present, because impl-style procs require it.
  absl::StatusOr<DocRef> FormatModule(const Module& n) override {
    Module& mutable_n = const_cast<Module&>(n);
    bool has_explicit_state_access =
        n.attributes().contains(ModuleAttribute::kExplicitStateAccess);
    bool has_generics = n.attributes().contains(ModuleAttribute::kGenerics);

    std::optional<Span> attr_span = n.GetAttributeSpan();
    if (!attr_span.has_value() && comments_.last_data_limit().has_value()) {
      Fileno fileno = comments_.last_data_limit()->fileno();
      std::vector<const CommentData*> unplaced = comments_.GetUnplacedComments(
          Span(Pos(fileno, 0, 0),
               Pos(fileno, std::numeric_limits<int64_t>::max(), 0)));
      if (!unplaced.empty() &&
          absl::StrContains(absl::AsciiStrToLower(unplaced[0]->text),
                            "copyright")) {
        Pos limit = unplaced[0]->span.limit();
        for (size_t k = 1; k < unplaced.size(); ++k) {
          if (unplaced[k]->span.start().lineno() ==
              unplaced[k - 1]->span.start().lineno() + 1) {
            limit = unplaced[k]->span.limit();
          } else {
            break;
          }
        }
        attr_span = Span(limit, limit);
      } else {
        attr_span = Span(Pos(fileno, 0, 0), Pos(fileno, 0, 0));
      }
    }

    if (!has_explicit_state_access) {
      mutable_n.AddAttribute(ModuleAttribute::kExplicitStateAccess, attr_span);
    }
    if (!has_generics) {
      mutable_n.AddAttribute(ModuleAttribute::kGenerics, attr_span);
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
    for (const Param* param : n.next().params()) {
      if (auto* tuple_type = dynamic_cast<const TupleTypeAnnotation*>(
              param->type_annotation());
          !tuple_type || !tuple_type->empty()) {
        state_params.push_back(param);
      }
    }

    // Perform checks for direct constant references in members and state
    // params.
    absl::flat_hash_set<std::string_view> local_constant_names;
    for (auto const& [name, _] : local_constants_) {
      local_constant_names.insert(name);
    }
    for (const ProcMember* member : members) {
      auto has_ref_status = HasReferenceToAnyName(member->type_annotation(),
                                                  local_constant_names);
      if (!has_ref_status.ok()) {
        status_ = has_ref_status.status();
        return arena_.empty();
      }
      if (has_ref_status.value()) {
        status_ = absl::InvalidArgumentError(absl::StrFormat(
            "Proc member `%s` references a constant declared "
            "inside the proc, which is not allowed in impl-style procs.",
            member->identifier()));
        return arena_.empty();
      }
    }
    for (const Param* param : state_params) {
      auto has_ref_status =
          HasReferenceToAnyName(param->type_annotation(), local_constant_names);
      if (!has_ref_status.ok()) {
        status_ = has_ref_status.status();
        return arena_.empty();
      }
      if (has_ref_status.value()) {
        status_ = absl::InvalidArgumentError(absl::StrFormat(
            "Proc state parameter `%s` references a constant declared "
            "inside the proc, which is not allowed in impl-style procs.",
            param->identifier()));
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
    std::vector<DocRef> additional_parametric_docs;
    std::vector<DocRef> additional_parametric_names;
    std::vector<const AstNode*> remaining_proc_level_decls;
    additional_parametric_docs.reserve(proc_level_decls.size());
    additional_parametric_names.reserve(proc_level_decls.size());
    remaining_proc_level_decls.reserve(proc_level_decls.size());

    for (const AstNode* node : proc_level_decls) {
      if (auto* t = dynamic_cast<const TypeAlias*>(node)) {
        if (needed_type_aliases.contains(t->identifier())) {
          auto param_doc_status = ProcessNeededTypeAlias(*t);
          if (!param_doc_status.ok()) {
            status_ = param_doc_status.status();
            return arena_.empty();
          }
          additional_parametric_docs.push_back(param_doc_status.value());
          additional_parametric_names.push_back(
              arena_.MakeText(t->identifier()));
          continue;
        }
      }
      remaining_proc_level_decls.push_back(node);
    }

    bool already_has_explicit_state_access =
        !state_params.empty() && HasExplicitStateAccess(n.next().body());

    std::vector<DocRef> impl_decl_docs;
    impl_decl_docs.reserve(remaining_proc_level_decls.size());
    for (const AstNode* node : remaining_proc_level_decls) {
      if (auto* c = dynamic_cast<const ConstantDef*>(node)) {
        impl_decl_docs.push_back(FormatConstantDef(*c));
      } else if (auto* t = dynamic_cast<const TypeAlias*>(node)) {
        impl_decl_docs.push_back(
            arena_.MakeConcat(FormatTypeAlias(*t), arena_.semi()));
      }
    }

    DocRef proc_decl_doc = FormatProcBlock(n, is_test, state_params, members,
                                           additional_parametric_docs);

    DocRef impl_block_doc =
        FormatImplBlock(n, already_has_explicit_state_access, state_params,
                        members, impl_decl_docs, additional_parametric_names);

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

  // Formats a legacy `spawn` statement into an impl-style instantiation and
  // spawn.
  //
  // Before:
  //   spawn MyProc(a, b)
  //
  // After:
  //   MyProc::new(a, b).spawn()
  DocRef FormatSpawn(const Spawn& n) override {
    std::vector<DocRef> callee_pieces = {FormatExpr(*n.callee())};

    if (!n.explicit_parametrics().empty()) {
      std::vector<DocRef> parametric_docs;
      parametric_docs.reserve(n.explicit_parametrics().size());
      for (const ExprOrType& et : n.explicit_parametrics()) {
        if (std::holds_alternative<Expr*>(et)) {
          parametric_docs.push_back(FormatExpr(*std::get<Expr*>(et)));
        } else {
          parametric_docs.push_back(
              FormatTypeAnnotation(*std::get<TypeAnnotation*>(et)));
        }
      }
      callee_pieces.push_back(
          ConcatNGroup(arena_, {arena_.oangle(),
                                FormatJoin(parametric_docs, Joiner::kCommaSpace,
                                           /*group=*/false),
                                arena_.cangle()}));
    }

    callee_pieces.push_back(arena_.MakeText("::new"));
    DocRef callee_doc = ConcatNGroup(arena_, callee_pieces);

    DocRef args_doc_internal = FormatJoin<const Expr*>(
        n.config()->args(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
        [this](const Expr* e) { return Format(e); });

    std::vector<DocRef> arg_pieces = {
        arena_.MakeNestIfFlatFits(
            /*on_nested_flat_ref=*/args_doc_internal,
            /*on_other_ref=*/arena_.MakeAlign(args_doc_internal)),
        arena_.cparen()};
    DocRef args_doc = ConcatNGroup(arena_, arg_pieces);
    DocRef args_doc_nested = arena_.MakeNest(args_doc);

    DocRef new_flat =
        ConcatN(arena_, {callee_doc, arena_.oparen(), args_doc});
    DocRef new_leader_flat =
        ConcatN(arena_, {callee_doc, arena_.oparen(), arena_.break0(),
                         args_doc_nested});
    DocRef new_call_doc = arena_.MakeGroup(
        arena_.MakeFlatChoice(/*on_flat=*/new_flat,
                              /*on_break=*/new_leader_flat));

    DocRef dot_spawn =
        ConcatN(arena_, {new_call_doc, arena_.dot(), arena_.MakeText("spawn")});
    DocRef spawn_flat =
        ConcatN(arena_, {dot_spawn, arena_.oparen(), arena_.cparen()});
    DocRef spawn_leader_flat =
        ConcatN(arena_, {dot_spawn, arena_.oparen(), arena_.break0(),
                         arena_.MakeNest(arena_.cparen())});
    return arena_.MakeGroup(
        arena_.MakeFlatChoice(/*on_flat=*/spawn_flat,
                              /*on_break=*/spawn_leader_flat));
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

  absl::StatusOr<DocRef> ProcessNeededTypeAlias(const TypeAlias& t) {
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

    DocRef rhs_doc = FormatTypeAnnotation(*substituted_rhs);

    DocRef param_doc = ConcatNGroup(
        arena_, {arena_.MakeText(t.identifier()), arena_.colon(),
                 arena_.space(), arena_.Make(Keyword::kType), arena_.space(),
                 arena_.equals(), arena_.space(), rhs_doc});
    return param_doc;
  }

  // Formats the new `proc` block containing member fields and channels.
  DocRef FormatProcBlock(const Proc& n, bool is_test,
                         absl::Span<const Param* const> state_params,
                         absl::Span<const ProcMember* const> members,
                         absl::Span<const DocRef> additional_parametrics) {
    std::vector<DocRef> attribute_pieces;
    if (n.is_test_utility() && !is_test) {
      attribute_pieces.push_back(
          ConcatN(arena_, {
                              arena_.MakeText("#"),
                              arena_.obracket(),
                              arena_.MakeText(std::string(kCfgTestAttr)),
                              arena_.cbracket(),
                              arena_.hard_line(),
                          }));
    }

    std::vector<DocRef> signature_pieces;
    if (n.is_public()) {
      signature_pieces.push_back(arena_.Make(Keyword::kPub));
      signature_pieces.push_back(arena_.space());
    }
    signature_pieces.push_back(arena_.Make(Keyword::kProc));
    signature_pieces.push_back(arena_.space());
    signature_pieces.push_back(arena_.MakeText(n.identifier()));

    if (!n.parametric_bindings().empty() && additional_parametrics.empty()) {
      Pos final_parametric_limit;
      if (!members.empty()) {
        final_parametric_limit = members.front()->span().start();
      } else if (!state_params.empty()) {
        final_parametric_limit = state_params.front()->span().start();
      } else {
        final_parametric_limit = n.span().limit();
      }
      signature_pieces.push_back(FormatParametricBindings(
          n.parametric_bindings(), final_parametric_limit,
          /*break_before_angle=*/false));
    } else if (n.IsParametric() || !additional_parametrics.empty()) {
      std::vector<DocRef> parametric_docs;
      parametric_docs.reserve(n.parametric_bindings().size() +
                              additional_parametrics.size());
      for (const ParametricBinding* pb : n.parametric_bindings()) {
        parametric_docs.push_back(FormatParametricBindingPtr(pb));
      }
      for (const DocRef& doc : additional_parametrics) {
        parametric_docs.push_back(doc);
      }
      DocRef flat_parametrics =
          ConcatNGroup(arena_, {arena_.oangle(),
                                FormatJoin(parametric_docs, Joiner::kCommaSpace,
                                           /*group=*/false),
                                arena_.cangle()});
      DocRef bindings_joined =
          FormatJoin(parametric_docs, Joiner::kCommaBreak1, /*group=*/false);
      DocRef parametric_guts = ConcatN(
          arena_,
          {arena_.oangle(), arena_.MakeAlign(bindings_joined), arena_.cangle()});
      DocRef break_parametrics = ConcatNGroup(
          arena_, {arena_.MakeFlatChoice(parametric_guts,
                                         arena_.MakeNest(parametric_guts))});
      signature_pieces.push_back(
          arena_.MakeFlatChoice(flat_parametrics, break_parametrics));
    }
    signature_pieces.push_back(arena_.space());
    signature_pieces.push_back(arena_.ocurl());

    int num_members = members.size() + state_params.size();
    std::vector<DocRef> body_pieces;
    body_pieces.reserve(num_members * 2);

    Pos last_stmt_limit = n.body_span().start();
    for (int i = 0; i < num_members; ++i) {
      std::string_view identifier;
      const TypeAnnotation* type_annotation = nullptr;
      std::optional<DocRef> comments;

      if (i < members.size()) {
        const ProcMember* member = members[i];
        identifier = member->identifier();
        type_annotation = member->type_annotation();
        comments =
            FormatCommentsBetween(last_stmt_limit, member->span().start());
        last_stmt_limit = member->span().limit();
      } else {
        const Param* state_param = state_params[i - members.size()];
        identifier = state_param->identifier();
        type_annotation = state_param->type_annotation();
      }

      std::vector<DocRef> line_pieces;
      if (comments.has_value()) {
        line_pieces.push_back(*comments);
        line_pieces.push_back(arena_.hard_line());
      }
      bool is_last = (i + 1 == num_members);
      DocRef comma_doc =
          is_last ? arena_.MakeFlatChoice(arena_.empty(), arena_.comma())
                  : arena_.comma();
      line_pieces.push_back(ConcatN(
          arena_,
          {arena_.MakeText(std::string(identifier)), arena_.colon(),
           arena_.space(), FormatTypeAnnotation(*type_annotation), comma_doc}));
      body_pieces.push_back(ConcatN(arena_, line_pieces));
      if (!is_last) {
        body_pieces.push_back(arena_.hard_line());
      }
    }

    DocRef proc_decl_doc;
    if (num_members == 0) {
      proc_decl_doc =
          ConcatNGroup(arena_, {
                                   ConcatNGroup(arena_, attribute_pieces),
                                   ConcatNGroup(arena_, signature_pieces),
                                   arena_.break0(),
                                   arena_.ccurl(),
                               });
    } else {
      proc_decl_doc = ConcatNGroup(
          arena_, {
                      ConcatNGroup(arena_, attribute_pieces),
                      ConcatNGroup(arena_, signature_pieces),
                      arena_.hard_line(),
                      arena_.MakeNest(ConcatN(arena_, body_pieces)),
                      arena_.hard_line(),
                      arena_.ccurl(),
                  });
    }
    return proc_decl_doc;
  }

  // Formats the legacy `config` and `init` functions into the constructor
  // `fn new`.
  //
  // Before:
  //   config(x: chan<u32> in) { (x,) }
  //   init { u32:42 }
  //
  // After:
  //   fn new(x: chan<u32> in) -> Self {
  //       Foo { x, state: u32:42 }
  //   }
  DocRef FormatNewFunction(const Proc& n,
                           absl::Span<const Param* const> state_params,
                           absl::Span<const ProcMember* const> members) {
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

    DocRef init_val_doc;
    const Expr* init_expr = nullptr;
    const StatementBlock* init_body = n.init().body();
    const auto& init_stmts = init_body->statements();
    if (init_stmts.size() == 1) {
      init_expr = std::get<Expr*>(init_stmts[0]->wrapped());
      init_val_doc = FormatExpr(*init_expr);
    } else {
      init_val_doc = FormatBlock(*init_body);
      if (init_body != nullptr && !init_body->empty() &&
          !init_body->trailing_semi()) {
        const Statement* last_stmt = init_body->statements().back();
        if (std::holds_alternative<Expr*>(last_stmt->wrapped())) {
          init_expr = std::get<Expr*>(last_stmt->wrapped());
        }
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

    std::vector<DocRef> struct_init_pieces =
        FormatStructInitPieces(config_tuple, state_params, members, init_expr,
                               init_val_doc, init_tuple);

    DocRef struct_init_doc;
    if (struct_init_pieces.empty()) {
      struct_init_doc =
          ConcatNGroup(arena_, {
                                   arena_.MakeText(n.identifier()),
                                   arena_.space(),
                                   arena_.ocurl(),
                                   arena_.ccurl(),
                               });
    } else {
      DocRef members_flat =
          FormatJoin(struct_init_pieces, Joiner::kCommaBreak1, /*group=*/false);
      DocRef on_flat = ConcatN(arena_, {arena_.space(), members_flat,
                                        arena_.space(), arena_.ccurl()});
      DocRef on_break = ConcatN(
          arena_,
          {
              arena_.hard_line(),
              arena_.MakeNest(FormatJoin(
                  struct_init_pieces, Joiner::kCommaHardlineTrailingCommaAlways,
                  /*group=*/false)),
              arena_.hard_line(),
              arena_.ccurl(),
          });
      struct_init_doc =
          ConcatNGroup(arena_, {
                                   arena_.MakeText(n.identifier()),
                                   arena_.space(),
                                   arena_.ocurl(),
                                   arena_.MakeFlatChoice(on_flat, on_break),
                               });
    }

    DocRef params_doc = FormatParams(n.config().params());

    std::vector<DocRef> new_sig_pieces = {
        arena_.Make(Keyword::kFn),
        arena_.space(),
        arena_.MakeText("new"),
    };
    std::vector<DocRef> params_pieces;
    params_pieces.push_back(arena_.break0());
    params_pieces.push_back(params_doc);

    std::vector<DocRef> return_pieces;
    return_pieces.push_back(arena_.break1());
    return_pieces.push_back(arena_.arrow());
    return_pieces.push_back(arena_.space());
    return_pieces.push_back(arena_.MakeText("Self"));
    return_pieces.push_back(arena_.space());
    return_pieces.push_back(arena_.ocurl());
    params_pieces.push_back(ConcatNGroup(arena_, return_pieces));

    new_sig_pieces.push_back(
        arena_.MakeNest(ConcatNGroup(arena_, params_pieces)));

    const auto& config_stmts = n.config().body()->statements();
    std::vector<DocRef> append_statements;
    append_statements.reserve(2);
    if (!state_params.empty() && state_params.size() > 1) {
      if (init_yields_tuple_per_state_param) {
        if (init_stmts.size() > 1) {
          DocRef prefix_doc = FormatBlock(
              *init_body, FormatBlockOptions{.start_idx = 0,
                                             .end_idx = static_cast<int>(
                                                 init_stmts.size() - 1),
                                             .force_trailing_semi = true,
                                             .add_curls = false,
                                             .add_nest = false});
          append_statements.push_back(prefix_doc);
        }
      } else {
        DocRef let_init =
            ConcatNGroup(arena_, {arena_.MakeText("let init_state ="),
                                  arena_.space(), init_val_doc, arena_.semi()});
        append_statements.push_back(let_init);
      }
    }
    append_statements.push_back(struct_init_doc);

    int end_idx;
    if (members.empty()) {
      if (!config_stmts.empty() && IsLiteralEmptyTuple(config_stmts.back())) {
        end_idx = config_stmts.size() - 1;
      } else {
        end_idx = config_stmts.size();
      }
    } else {
      end_idx = config_stmts.empty() ? 0 : config_stmts.size() - 1;
    }
    DocRef body_doc =
        FormatBlock(*n.config().body(),
                    FormatBlockOptions{.start_idx = 0,
                                       .end_idx = end_idx,
                                       .append_statements = append_statements,
                                       .add_curls = false,
                                       .force_multiline = true});
    std::vector<DocRef> new_body = {
        arena_.hard_line(),
        body_doc,
        arena_.hard_line(),
        arena_.ccurl(),
    };

    DocRef new_fn_doc = ConcatNGroup(
        arena_,
        {ConcatNGroup(arena_, new_sig_pieces), ConcatN(arena_, new_body)});

    Pos last_stmt_limit = n.body_span().start();
    if (!members.empty()) {
      last_stmt_limit = members.back()->span().limit();
    }

    std::optional<DocRef> new_comments =
        FormatCommentsBetween(last_stmt_limit, n.config().span().start());
    std::optional<DocRef> init_comments = FormatCommentsBetween(
        n.config().span().limit(), n.init().span().start());

    std::vector<DocRef> new_fn_pieces;
    if (new_comments.has_value()) {
      new_fn_pieces.push_back(*new_comments);
      new_fn_pieces.push_back(arena_.hard_line());
    }
    if (init_comments.has_value()) {
      new_fn_pieces.push_back(*init_comments);
      new_fn_pieces.push_back(arena_.hard_line());
    }
    new_fn_pieces.push_back(new_fn_doc);
    return ConcatN(arena_, new_fn_pieces);
  }

  // Helper for `FormatNewFunction` which creates the pieces of the struct
  // initializer returned by the generated `new` function. These are sourced
  // from the legacy `init` function and the legacy `config` function.
  std::vector<DocRef> FormatStructInitPieces(
      const XlsTuple* config_tuple, absl::Span<const Param* const> state_params,
      absl::Span<const ProcMember* const> members, const Expr* init_expr,
      DocRef init_val_doc, const XlsTuple* init_tuple) {
    std::vector<DocRef> struct_init_pieces;
    struct_init_pieces.reserve(members.size() + state_params.size());
    for (int i = 0; i < members.size(); ++i) {
      const ProcMember* member = members[i];
      const Expr* member_init_expr = config_tuple->members()[i];
      bool is_shorthand = false;
      if (auto* name_ref = dynamic_cast<const NameRef*>(member_init_expr);
          name_ref != nullptr &&
          name_ref->identifier() == member->identifier()) {
        is_shorthand = true;
      }

      if (is_shorthand) {
        struct_init_pieces.push_back(arena_.MakeText(member->identifier()));
      } else {
        DocRef val_doc = FormatExpr(*member_init_expr);
        struct_init_pieces.push_back(
            ConcatNGroup(arena_, {arena_.MakeText(member->identifier()),
                                  arena_.colon(), arena_.space(), val_doc}));
      }
    }
    if (!state_params.empty()) {
      if (state_params.size() == 1) {
        bool is_state_shorthand = false;
        if (init_expr != nullptr) {
          if (auto* name_ref = dynamic_cast<const NameRef*>(init_expr);
              name_ref != nullptr &&
              name_ref->identifier() == state_params[0]->identifier()) {
            is_state_shorthand = true;
          }
        }

        if (is_state_shorthand) {
          struct_init_pieces.push_back(
              arena_.MakeText(state_params[0]->identifier()));
        } else {
          struct_init_pieces.push_back(ConcatNGroup(
              arena_, {arena_.MakeText(state_params[0]->identifier()),
                       arena_.colon(), arena_.space(), init_val_doc}));
        }
      } else {
        if (init_tuple != nullptr &&
            init_tuple->members().size() == state_params.size()) {
          for (int i = 0; i < state_params.size(); ++i) {
            DocRef init_val_i_doc = FormatExpr(*init_tuple->members()[i]);
            struct_init_pieces.push_back(ConcatNGroup(
                arena_, {arena_.MakeText(state_params[i]->identifier()),
                         arena_.colon(), arena_.space(), init_val_i_doc}));
          }
        } else {
          for (int i = 0; i < state_params.size(); ++i) {
            struct_init_pieces.push_back(ConcatNGroup(
                arena_,
                {arena_.MakeText(state_params[i]->identifier()), arena_.colon(),
                 arena_.space(),
                 arena_.MakeText(absl::StrFormat("init_state.%d", i))}));
          }
        }
      }
    }
    return struct_init_pieces;
  }

  // Formats the legacy `next` function to read and write state member
  // variables.
  //
  // Before:
  //   next(state: u32) {
  //       state + u32:1
  //   }
  //
  // After:
  //   fn next(self) {
  //       let state = read(self.state);
  //       let next_state = state + u32:1;
  //       write(self.state, next_state);
  //   }
  std::optional<DocRef> FormatNextFunction(
      const Proc& n, bool already_has_explicit_state_access,
      absl::Span<const Param* const> state_params) {
    const Function& next_fn = n.next();

    if (!(!state_params.empty() || FunctionDoesAnything(next_fn) ||
          comments_.HasComments(
              Span(n.init().span().limit(), next_fn.span().limit())))) {
      return std::nullopt;
    }

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
          NameDef* self_def = next_fn.params().empty()
                                  ? nullptr
                                  : next_fn.params()[0]->name_def();
          auto* self_ref =
              module->Make<NameRef>(name_ref->span(), "self", self_def);
          auto* attr_node = module->Make<Attr>(name_ref->span(), self_ref,
                                               name_ref->identifier());
          return attr_node;
        }
      }
      return std::nullopt;
    };

    absl::StatusOr<StatementBlock*> cloned_next_body_status =
        CloneNode<StatementBlock>(const_cast<StatementBlock*>(next_fn.body()),
                                  replacer);
    CHECK_OK(cloned_next_body_status.status());
    StatementBlock* cloned_next_body = cloned_next_body_status.value();

    std::vector<DocRef> prepend_statements;
    if (!state_params.empty() && !already_has_explicit_state_access) {
      prepend_statements.reserve(state_params.size());
      for (const Param* state_param : state_params) {
        DocRef read_stmt = ConcatNGroup(
            arena_, {arena_.MakeText("let"), arena_.space(),
                     arena_.MakeText(state_param->identifier()), arena_.space(),
                     arena_.equals(), arena_.space(),
                     arena_.MakeText(absl::StrFormat(
                         "read(self.%s)", state_param->identifier())),
                     arena_.semi()});
        prepend_statements.push_back(read_stmt);
      }
    }

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

    std::vector<DocRef> append_statements;
    if (!state_params.empty() && !already_has_explicit_state_access) {
      append_statements.reserve(state_params.size());
      CHECK(!next_stmts.empty());
      const Expr* final_expr = std::get<Expr*>(next_stmts.back()->wrapped());
      if (state_params.size() == 1) {
        DocRef write_stmt = ConcatNGroup(
            arena_, {arena_.MakeText(absl::StrFormat(
                         "write(self.%s,", state_params[0]->identifier())),
                     arena_.space(), FormatExpr(*final_expr),
                     arena_.MakeText(")"), arena_.semi()});
        append_statements.push_back(write_stmt);
      } else {
        DocRef let_stmt = ConcatNGroup(
            arena_, {arena_.MakeText("let next_state ="), arena_.space(),
                     FormatExpr(*final_expr), arena_.semi()});
        append_statements.push_back(let_stmt);
        for (int i = 0; i < state_params.size(); ++i) {
          DocRef write_stmt =
              ConcatNGroup(arena_, {arena_.MakeText(absl::StrFormat(
                                        "write(self.%s, next_state.%d)",
                                        state_params[i]->identifier(), i)),
                                    arena_.semi()});
          append_statements.push_back(write_stmt);
        }
      }
    }

    DocRef next_fn_doc;
    bool is_empty_body = false;
    if (state_params.empty()) {
      if (next_stmts.empty()) {
        is_empty_body = true;
      } else if (next_stmts.size() == 1 && has_redundant_stateless_return) {
        is_empty_body = true;
      }
    }
    if (is_empty_body) {
      next_fn_doc = ConcatNGroup(arena_, {
                                             arena_.Make(Keyword::kFn),
                                             arena_.space(),
                                             arena_.MakeText("next"),
                                             arena_.oparen(),
                                             arena_.MakeText("self"),
                                             arena_.cparen(),
                                             arena_.space(),
                                             arena_.ocurl(),
                                             arena_.break0(),
                                             arena_.ccurl(),
                                         });
    } else {
      DocRef body_doc = FormatBlock(
          *cloned_next_body,
          FormatBlockOptions{.start_idx = 0,
                             .end_idx = end_idx,
                             .prepend_statements = prepend_statements,
                             .append_statements = append_statements,
                             .force_trailing_semi = true,
                             .add_curls = false,
                             .force_multiline = true});
      std::vector<DocRef> next_body = {
          arena_.hard_line(),
          body_doc,
          arena_.hard_line(),
          arena_.ccurl(),
      };
      next_fn_doc = ConcatNGroup(arena_, {
                                             arena_.Make(Keyword::kFn),
                                             arena_.space(),
                                             arena_.MakeText("next"),
                                             arena_.oparen(),
                                             arena_.MakeText("self"),
                                             arena_.cparen(),
                                             arena_.space(),
                                             arena_.ocurl(),
                                             ConcatN(arena_, next_body),
                                         });
    }

    std::optional<DocRef> next_comments =
        FormatCommentsBetween(n.init().span().limit(), n.next().span().start());
    std::vector<DocRef> next_fn_pieces;
    if (next_comments.has_value()) {
      next_fn_pieces.push_back(*next_comments);
      next_fn_pieces.push_back(arena_.hard_line());
    }
    next_fn_pieces.push_back(next_fn_doc);
    return ConcatN(arena_, next_fn_pieces);
  }

  // Formats the `impl` block enclosing the `new` and `next` member functions.
  //
  // Before: init(), config(), and next() were inside the proc.
  //
  // After:
  //   impl Foo {
  //       fn new(...) -> Self { ... }
  //       fn next(self) { ... }
  //   }
  DocRef FormatImplBlock(const Proc& n, bool already_has_explicit_state_access,
                         absl::Span<const Param* const> state_params,
                         absl::Span<const ProcMember* const> members,
                         absl::Span<const DocRef> module_decl_docs,
                         absl::Span<const DocRef> additional_parametric_names) {
    DocRef final_new_fn = FormatNewFunction(n, state_params, members);
    std::optional<DocRef> final_next_fn =
        FormatNextFunction(n, already_has_explicit_state_access, state_params);

    DocRef impl_target = arena_.MakeText(n.identifier());
    if (n.IsParametric() || !additional_parametric_names.empty()) {
      std::vector<DocRef> parametric_names;
      parametric_names.reserve(n.parametric_bindings().size() +
                               additional_parametric_names.size());
      for (const ParametricBinding* pb : n.parametric_bindings()) {
        parametric_names.push_back(arena_.MakeText(pb->identifier()));
      }
      for (const DocRef& doc : additional_parametric_names) {
        parametric_names.push_back(doc);
      }
      impl_target = ConcatNGroup(
          arena_, {impl_target, arena_.oangle(),
                   FormatJoin(parametric_names, Joiner::kCommaSpace,
                              /*group=*/false),
                   arena_.cangle()});
    }

    std::vector<DocRef> impl_guts;
    impl_guts.reserve(module_decl_docs.size() * 3 + 4);
    for (const DocRef& doc : module_decl_docs) {
      impl_guts.push_back(arena_.MakeNest(doc));
      impl_guts.push_back(arena_.hard_line());
      impl_guts.push_back(arena_.hard_line());
    }
    impl_guts.push_back(arena_.MakeNest(final_new_fn));
    if (final_next_fn.has_value()) {
      impl_guts.push_back(arena_.hard_line());
      impl_guts.push_back(arena_.hard_line());
      impl_guts.push_back(arena_.MakeNest(*final_next_fn));
    }

    DocRef impl_block_doc =
        ConcatNGroup(arena_, {
                                 arena_.Make(Keyword::kImpl),
                                 arena_.space(),
                                 impl_target,
                                 arena_.space(),
                                 arena_.ocurl(),
                                 arena_.hard_line(),
                                 ConcatN(arena_, impl_guts),
                                 arena_.hard_line(),
                                 arena_.ccurl(),
                             });
    return impl_block_doc;
  }

  absl::flat_hash_map<std::string_view, const ConstantDef*> local_constants_;
  absl::flat_hash_map<std::string_view, const TypeAlias*> local_type_aliases_;
  std::optional<absl::flat_hash_set<std::string>> current_proc_member_names_;
  absl::Status status_ = absl::OkStatus();
};

}  // namespace

std::unique_ptr<Formatter> CreateLegacyProcConverter(Comments& comments,
                                                     DocArena& arena) {
  return std::make_unique<LegacyProcConverter>(comments, arena);
}

}  // namespace xls::dslx
