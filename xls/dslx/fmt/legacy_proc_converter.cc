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

absl::StatusOr<bool> HasReferenceToAnyName(
    const AstNode* node, const absl::flat_hash_set<std::string_view>& names) {
  XLS_ASSIGN_OR_RETURN(
      (std::vector<std::pair<const NameRef*, const NameDef*>> defs),
      CollectReferencedUnder(node, /*want_types=*/true));
  for (const auto& [_, def] : defs) {
    if (names.contains(def->identifier())) {
      return true;
    }
  }
  return false;
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
    XLS_ASSIGN_OR_RETURN(DocRef doc, Formatter::FormatModule(n));
    if (!status_.ok()) {
      return status_;
    }
    bool has_feature =
        n.attributes().contains(ModuleAttribute::kExplicitStateAccess);
    if (!has_feature) {
      DocRef attr_doc = ConcatNGroup(
          arena_, {arena_.MakeText("#![feature(explicit_state_access)]"),
                   arena_.hard_line(), arena_.hard_line()});
      doc = arena_.MakeConcat(attr_doc, doc);
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

    std::vector<const ConstantDef*> proc_constant_defs;
    proc_constant_defs.reserve(n.stmts().size());
    std::vector<const ProcMember*> members;
    members.reserve(n.stmts().size());
    absl::Status status =
        AnalyzeAndSplitProcStatements(n, proc_constant_defs, members);
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

    bool already_has_explicit_state_access =
        !state_params.empty() && HasExplicitStateAccess(n.next().body());

    std::vector<DocRef> impl_constant_defs =
        FormatProcConstantDefs(proc_constant_defs);

    DocRef proc_decl_doc = FormatProcBlock(n, is_test, state_params, members);

    DocRef impl_block_doc =
        FormatImplBlock(n, already_has_explicit_state_access, state_params,
                        members, impl_constant_defs);

    std::vector<DocRef> final_pieces{proc_decl_doc, arena_.hard_line(),
                                     arena_.hard_line(), impl_block_doc};

    current_proc_member_names_ = std::nullopt;
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
    std::vector<DocRef> pieces;
    pieces.push_back(FormatExpr(*n.callee()));

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
      pieces.push_back(ConcatNGroup(
          arena_, {arena_.oangle(),
                   FormatJoin(parametric_docs, Joiner::kCommaBreak1,
                              /*group=*/false),
                   arena_.cangle()}));
    }

    pieces.push_back(arena_.MakeText("::new"));

    std::vector<DocRef> arg_docs;
    arg_docs.reserve(n.config()->args().size());
    for (const Expr* arg : n.config()->args()) {
      arg_docs.push_back(FormatExpr(*arg));
    }

    DocRef args_guts =
        FormatJoin(arg_docs, Joiner::kCommaBreak1, /*group=*/false);
    pieces.push_back(arena_.oparen());
    pieces.push_back(arena_.MakeNestIfFlatFits(
        /*on_nested_flat_ref=*/args_guts,
        /*on_other_ref=*/arena_.MakeAlign(args_guts)));
    pieces.push_back(arena_.cparen());
    pieces.push_back(arena_.MakeText(".spawn()"));

    return ConcatNGroup(arena_, pieces);
  }

 private:
  absl::Status AnalyzeAndSplitProcStatements(
      const Proc& n, std::vector<const ConstantDef*>& constant_defs,
      std::vector<const ProcMember*>& members) {
    absl::flat_hash_set<std::string_view> constant_names;
    for (const ProcStmt& stmt : n.stmts()) {
      absl::Status visit_status = std::visit(
          Visitor{
              [&](const Function* f) { return absl::OkStatus(); },
              [&](const ProcMember* m) {
                members.push_back(m);
                return absl::OkStatus();
              },
              [&](const ConstantDef* c) {
                constant_defs.push_back(c);
                constant_names.insert(c->identifier());
                return absl::OkStatus();
              },
              [&](const TypeAlias* t) {
                return absl::InvalidArgumentError(
                    "Type aliases inside a proc are not supported in "
                    "impl-style procs.");
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

    // Check if any member references proc constants.
    for (const ProcMember* member : members) {
      XLS_ASSIGN_OR_RETURN(
          bool has_reference,
          HasReferenceToAnyName(member->type_annotation(), constant_names));
      if (has_reference) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc member `%s` references a constant declared inside "
            "the proc, which is not allowed in impl-style procs.",
            member->identifier()));
      }
    }

    // Check if any state param references proc constants.
    for (const Param* param : n.next().params()) {
      XLS_ASSIGN_OR_RETURN(
          bool has_reference,
          HasReferenceToAnyName(param->type_annotation(), constant_names));
      if (has_reference) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc state parameter `%s` references a constant declared "
            "inside the proc, which is not allowed in impl-style procs.",
            param->identifier()));
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

  // Hoists constants declared inside a legacy proc so that they can be
  // formatted as part of the impl block.
  std::vector<DocRef> FormatProcConstantDefs(
      absl::Span<const ConstantDef* const> nodes) {
    std::vector<DocRef> result;
    result.reserve(nodes.size());
    for (const ConstantDef* c : nodes) {
      result.push_back(FormatConstantDef(*c));
    }
    return result;
  }

  // Formats the new `proc` block containing member fields and channels.
  DocRef FormatProcBlock(const Proc& n, bool is_test,
                         absl::Span<const Param* const> state_params,
                         absl::Span<const ProcMember* const> members) {
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

    if (n.IsParametric()) {
      std::vector<DocRef> parametric_docs;
      parametric_docs.reserve(n.parametric_bindings().size());
      for (const ParametricBinding* pb : n.parametric_bindings()) {
        parametric_docs.push_back(FormatParametricBindingPtr(pb));
      }
      signature_pieces.push_back(
          ConcatNGroup(arena_, {arena_.oangle(),
                                FormatJoin(parametric_docs, Joiner::kCommaSpace,
                                           /*group=*/false),
                                arena_.cangle()}));
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
        params_doc,
        arena_.space(),
        arena_.arrow(),
        arena_.space(),
        arena_.MakeText("Self"),
        arena_.space(),
        arena_.ocurl(),
    };

    const auto& config_stmts = n.config().body()->statements();
    std::vector<DocRef> append_statements;
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
                                       .add_curls = false});
    std::vector<DocRef> new_body = {
        arena_.break1(),
        body_doc,
        arena_.break1(),
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
      if (auto* name_ref = dynamic_cast<const NameRef*>(node)) {
        if (current_proc_member_names_.has_value() &&
            current_proc_member_names_->contains(name_ref->identifier())) {
          NameDef* self_def = next_fn.params().empty()
                                  ? nullptr
                                  : next_fn.params()[0]->name_def();
          auto* self_ref =
              module->Make<NameRef>(name_ref->span(), "self", self_def);
          auto* attr_node = module->Make<Attr>(name_ref->span(), self_ref,
                                               name_ref->identifier());
          return attr_node;
        }
        if (already_has_explicit_state_access) {
          for (const Param* state_param : state_params) {
            if (name_ref->identifier() == state_param->identifier()) {
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

    std::vector<DocRef> append_statements;
    const auto& next_stmts = cloned_next_body->statements();
    int end_idx = (!already_has_explicit_state_access && !next_stmts.empty())
                      ? next_stmts.size() - 1
                      : next_stmts.size();
    if (!state_params.empty() && !already_has_explicit_state_access) {
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
    if (next_stmts.size() <= 1 && state_params.empty()) {
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
                             .add_curls = false});
      std::vector<DocRef> next_body = {
          arena_.break1(),
          body_doc,
          arena_.break1(),
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
                         absl::Span<const DocRef> module_decl_docs) {
    DocRef final_new_fn = FormatNewFunction(n, state_params, members);
    std::optional<DocRef> final_next_fn =
        FormatNextFunction(n, already_has_explicit_state_access, state_params);

    DocRef impl_target = arena_.MakeText(n.identifier());
    if (n.IsParametric()) {
      std::vector<DocRef> parametric_names;
      parametric_names.reserve(n.parametric_bindings().size());
      for (const ParametricBinding* pb : n.parametric_bindings()) {
        parametric_names.push_back(arena_.MakeText(pb->identifier()));
      }
      impl_target = ConcatNGroup(
          arena_, {impl_target, arena_.oangle(),
                   FormatJoin(parametric_names, Joiner::kCommaBreak1,
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

  std::optional<absl::flat_hash_set<std::string>> current_proc_member_names_;
  absl::Status status_ = absl::OkStatus();
};

}  // namespace

std::unique_ptr<Formatter> CreateLegacyProcConverter(Comments& comments,
                                                     DocArena& arena) {
  return std::make_unique<LegacyProcConverter>(comments, arena);
}

}  // namespace xls::dslx
