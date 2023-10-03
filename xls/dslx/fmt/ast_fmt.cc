// Copyright 2023 The XLS Authors
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

#include "xls/dslx/fmt/ast_fmt.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"

namespace xls::dslx {
namespace {

// Forward decl.
DocRef Fmt(const TypeAnnotation& n, const Comments& comments, DocArena& arena);

DocRef Fmt(const BuiltinTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  return arena.MakeText(BuiltinTypeToString(n.builtin_type()));
}

DocRef Fmt(const TupleTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  std::vector<DocRef> pieces = {arena.oparen()};
  for (size_t i = 0; i < n.size(); ++i) {
    pieces.push_back(Fmt(*n.members()[i], comments, arena));
    if (i + 1 != n.size()) {
      pieces.push_back(arena.comma());
      pieces.push_back(arena.break1());
    }
  }
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const TypeAnnotation& n, const Comments& comments, DocArena& arena) {
  if (auto* t = dynamic_cast<const BuiltinTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }
  if (auto* t = dynamic_cast<const TupleTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }

  XLS_LOG(FATAL) << "handle type annotation: " << n.ToString();
}

DocRef Fmt(const TypeAlias& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle type alias: " << n.ToString();
}

DocRef Fmt(const NameDef& n, const Comments& comments, DocArena& arena) {
  return arena.MakeText(n.identifier());
}

DocRef Fmt(const NameRef& n, const Comments& comments, DocArena& arena) {
  return arena.MakeText(n.identifier());
}

DocRef Fmt(const Number& n, const Comments& comments, DocArena& arena) {
  DocRef num_text = arena.MakeText(n.text());
  if (const TypeAnnotation* type = n.type_annotation()) {
    return ConcatNGroup(arena, {Fmt(*type, comments, arena), arena.colon(),
                                arena.break0(), num_text});
  }
  return num_text;
}

DocRef Fmt(const WildcardPattern& n, const Comments& comments,
           DocArena& arena) {
  return arena.underscore();
}

DocRef Fmt(const Array& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle array: " << n.ToString();
}

DocRef Fmt(const Attr& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle attr: " << n.ToString();
}

DocRef Fmt(const Binop& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle binop: " << n.ToString();
}

DocRef Fmt(const Block& n, const Comments& comments, DocArena& arena) {
  if (n.statements().empty()) {
    return ConcatNGroup(arena, {arena.ocurl(), arena.break0(), arena.ccurl()});
  }

  // We only want to flatten single-statement blocks -- multi-statement blocks
  // we always make line breaks between the statements.
  if (n.statements().size() == 1) {
    std::vector<DocRef> pieces = {arena.ocurl(), arena.break1(),
                                  Fmt(*n.statements()[0], comments, arena)};
    if (n.trailing_semi()) {
      pieces.push_back(arena.semi());
    }
    pieces.push_back(arena.break1());
    pieces.push_back(arena.ccurl());
    return ConcatNGroup(arena, pieces);
  }

  // Emit a '{' then nest to emit statements with semis, then emit a '}' outside
  // the nesting.
  std::vector<DocRef> top = {
      arena.ocurl(),
  };

  std::vector<DocRef> nested = {arena.hard_line()};
  for (size_t i = 0; i < n.statements().size(); ++i) {
    const Statement* stmt = n.statements()[i];
    nested.push_back(Fmt(*stmt, comments, arena));
    bool last_stmt = i + 1 == n.statements().size();
    if (!last_stmt || n.trailing_semi()) {
      nested.push_back(arena.semi());
    }
    if (!last_stmt) {
      nested.push_back(arena.hard_line());
    }
  }

  top.push_back(arena.MakeNest(ConcatN(arena, nested)));
  top.push_back(arena.hard_line());
  top.push_back(arena.ccurl());

  return ConcatNGroup(arena, top);
}

DocRef Fmt(const Cast& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle cast: " << n.ToString();
}

DocRef Fmt(const ChannelDecl& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle channel decl: " << n.ToString();
}

DocRef Fmt(const ColonRef& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle colon ref: " << n.ToString();
}

DocRef Fmt(const For& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle for: " << n.ToString();
}

DocRef Fmt(const FormatMacro& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle format macro: " << n.ToString();
}

DocRef Fmt(const Index& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle index: " << n.ToString();
}

DocRef Fmt(const Invocation& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle invocation: " << n.ToString();
}

DocRef Fmt(const Match& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle match: " << n.ToString();
}

DocRef Fmt(const Spawn& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle spawn: " << n.ToString();
}

DocRef Fmt(const XlsTuple& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle tuple: " << n.ToString();
}

DocRef Fmt(const SplatStructInstance& n, const Comments& comments,
           DocArena& arena) {
  XLS_LOG(FATAL) << "handle splat struct instance: " << n.ToString();
}

DocRef Fmt(const StructInstance& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle struct instance: " << n.ToString();
}

DocRef Fmt(const String& n, const Comments& comments, DocArena& arena) {
  return arena.MakeText(n.ToString());
}

DocRef Fmt(const Conditional& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle conditional: " << n.ToString();
}

DocRef Fmt(const ConstAssert& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle const assert: " << n.ToString();
}

DocRef Fmt(const TupleIndex& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle tuple index: " << n.ToString();
}

DocRef Fmt(const UnrollFor& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle unroll for: " << n.ToString();
}

DocRef Fmt(const ZeroMacro& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle zero macro: " << n.ToString();
}

DocRef Fmt(const Unop& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle unop: " << n.ToString();
}

// Forward decl.
DocRef Fmt(const Range& n, const Comments& comments, DocArena& arena);
DocRef Fmt(const Let& n, const Comments& comments, DocArena& arena);

class FmtExprVisitor : public ExprVisitor {
 public:
  FmtExprVisitor(DocArena& arena, const Comments& comments)
      : arena_(arena), comments_(comments) {}

  ~FmtExprVisitor() override = default;

#define DEFINE_HANDLER(__type)                               \
  absl::Status Handle##__type(const __type* expr) override { \
    result_ = Fmt(*expr, comments_, arena_);                 \
    return absl::OkStatus();                                 \
  }

  XLS_DSLX_EXPR_NODE_EACH(DEFINE_HANDLER)

#undef DEFINE_HANDLER

  DocRef result() const { return result_.value(); }

 private:
  DocArena& arena_;
  const Comments& comments_;
  std::optional<DocRef> result_;
};

DocRef Fmt(const Expr& n, const Comments& comments, DocArena& arena) {
  FmtExprVisitor v(arena, comments);
  XLS_CHECK_OK(n.AcceptExpr(&v));
  return v.result();
}

DocRef Fmt(const Range& n, const Comments& comments, DocArena& arena) {
  return ConcatNGroup(
      arena, {Fmt(*n.start(), comments, arena), arena.break0(), arena.dotdot(),
              arena.break0(), Fmt(*n.end(), comments, arena)});
}

DocRef Fmt(const NameDefTree::Leaf& n, const Comments& comments,
           DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const NameDef* n) { return Fmt(*n, comments, arena); },
          [&](const NameRef* n) { return Fmt(*n, comments, arena); },
          [&](const WildcardPattern* n) { return Fmt(*n, comments, arena); },
          [&](const Number* n) { return Fmt(*n, comments, arena); },
          [&](const ColonRef* n) { return Fmt(*n, comments, arena); },
          [&](const Range* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef Fmt(const NameDefTree& n, const Comments& comments, DocArena& arena) {
  if (n.is_leaf()) {
    return Fmt(n.leaf(), comments, arena);
  }
  std::vector<DocRef> pieces = {arena.oparen()};
  std::vector<std::variant<NameDefTree::Leaf, NameDefTree*>> flattened =
      n.Flatten1();
  for (size_t i = 0; i < flattened.size(); ++i) {
    const auto& item = flattened[i];
    absl::visit(Visitor{
                    [&](const NameDefTree::Leaf& leaf) {
                      pieces.push_back(Fmt(leaf, comments, arena));
                    },
                    [&](const NameDefTree* subtree) {
                      pieces.push_back(Fmt(*subtree, comments, arena));
                    },
                },
                item);
    if (i + 1 != flattened.size()) {
      pieces.push_back(arena.comma());
      pieces.push_back(arena.break1());
    }
  }
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Let& n, const Comments& comments, DocArena& arena) {
  DocRef break1 = arena.break1();

  std::vector<DocRef> guts = {arena.MakeText(n.is_const() ? "const" : "let"),
                              break1, Fmt(*n.name_def_tree(), comments, arena)};
  if (const TypeAnnotation* t = n.type_annotation()) {
    guts.push_back(arena.colon());
    guts.push_back(break1);
    guts.push_back(Fmt(*t, comments, arena));
  }

  guts.push_back(break1);
  guts.push_back(arena.equals());
  guts.push_back(break1);
  guts.push_back(Fmt(*n.rhs(), comments, arena));

  DocRef syntax = ConcatNGroup(arena, guts);

  std::vector<const CommentData*> comment_data = comments.GetComments(n.span());
  if (comment_data.size() == 1) {
    std::string comment_text = comment_data[0]->text;
    if (!comment_text.empty() && comment_text.back() == '\n') {
      comment_text.pop_back();
    }

    DocRef comment_text_ref = arena.MakeText(comment_text);

    // If it's a single line comment we create a FlatChoice between:
    //    let ... // comment text
    //
    // and:
    //
    //    // comment text reflowed with // prefix
    //    let ...
    DocRef flat = ConcatN(
        arena, {syntax, arena.space(), arena.slash_slash(), comment_text_ref});

    // TODO(leary): 2023-09-30 Make this so it reflows overlong lines in the
    // comment text with the // prefix inserted at the indentation level.
    DocRef line_prefixed = ConcatN(
        arena,
        {arena.slash_slash(), comment_text_ref, arena.hard_line(), syntax});
    return arena.MakeGroup(arena.MakeFlatChoice(flat, line_prefixed));
  }

  if (!comment_data.empty()) {
    XLS_LOG(FATAL) << "let: multiple inline comments";
  }

  return syntax;
}

}  // namespace

/* static */ Comments Comments::Create(absl::Span<const CommentData> comments) {
  absl::flat_hash_map<int64_t, CommentData> line_to_comment;
  for (const CommentData& cd : comments) {
    XLS_VLOG(3) << "comment on line: " << cd.span.start().lineno();
    // Note: we don't have multi-line comments for now, so we just note the
    // start line number for the comment.
    line_to_comment[cd.span.start().lineno()] = cd;
  }
  return Comments{std::move(line_to_comment)};
}

std::vector<const CommentData*> Comments::GetComments(
    const Span& node_span) const {
  XLS_VLOG(3) << "GetComments; node_span: " << node_span;

  // Implementation note: this will typically be a single access (as most things
  // will be on a single line), so we prefer a flat hash map to a btree map.
  std::vector<const CommentData*> results;
  for (int64_t i = node_span.start().lineno(); i <= node_span.limit().lineno();
       ++i) {
    if (auto it = line_to_comment_.find(i); it != line_to_comment_.end()) {
      results.push_back(&it->second);
    }
  }
  return results;
}

DocRef Fmt(const Statement& n, const Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return Fmt(*n, comments, arena); },
          [&](const TypeAlias* n) { return Fmt(*n, comments, arena); },
          [&](const Let* n) { return Fmt(*n, comments, arena); },
          [&](const ConstAssert* n) { return Fmt(*n, comments, arena); },
      },
      n.wrapped());
}

static DocRef FmtParams(absl::Span<const Param* const> params,
                        const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {arena.oparen()};
  for (const Param* param : params) {
    DocRef type = Fmt(*param->type_annotation(), comments, arena);
    pieces.push_back(ConcatNGroup(
        arena, {arena.MakeText(param->identifier()), arena.break0(),
                arena.colon(), arena.break1(), type}));
  }
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Function& n, const Comments& comments, DocArena& arena) {
  DocRef fn = arena.MakeText("fn");
  DocRef name = arena.MakeText(n.identifier());

  DocRef params = FmtParams(n.params(), comments, arena);

  std::vector<DocRef> pieces = {fn,     arena.break1(), name, arena.break0(),
                                params, arena.break1()};
  if (n.return_type() != nullptr) {
    pieces.push_back(arena.arrow());
    pieces.push_back(arena.break1());
    pieces.push_back(Fmt(*n.return_type(), comments, arena));
    pieces.push_back(arena.break1());
  }

  return ConcatNGroup(arena, {
                                 ConcatNGroup(arena, pieces),
                                 Fmt(*n.body(), comments, arena),
                             });
}

}  // namespace xls::dslx
