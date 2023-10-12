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
#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/token.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {
namespace {

// Forward decls.
DocRef Fmt(const TypeAnnotation& n, const Comments& comments, DocArena& arena);
DocRef Fmt(const ColonRef& n, const Comments& comments, DocArena& arena);
DocRef FmtExprOrType(const ExprOrType& n, const Comments& comments,
                     DocArena& arena);
DocRef Fmt(const NameDefTree& n, const Comments& comments, DocArena& arena);

static DocRef FmtExprPtr(const Expr* n, const Comments& comments,
                         DocArena& arena) {
  XLS_CHECK(n != nullptr);
  return Fmt(*n, comments, arena);
}

enum class Joiner : uint8_t {
  kCommaSpace,
  kCommaBreak1,
  kSpaceBarBreak,
  kHardLine,
};

// Helper for doing a "join via comma space" pattern with doc refs.
//
// This elides the "joiner" being present after the last item.
template <typename T>
DocRef FmtJoin(
    absl::Span<const T> items, Joiner joiner,
    const std::function<DocRef(const T&, const Comments&, DocArena&)>& fmt,
    const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  for (size_t i = 0; i < items.size(); ++i) {
    const T& item = items[i];
    pieces.push_back(fmt(item, comments, arena));
    if (i + 1 != items.size()) {
      switch (joiner) {
        case Joiner::kCommaSpace:
          pieces.push_back(arena.comma());
          pieces.push_back(arena.space());
          break;
        case Joiner::kCommaBreak1:
          pieces.push_back(arena.comma());
          pieces.push_back(arena.break1());
          break;
        case Joiner::kSpaceBarBreak:
          pieces.push_back(arena.space());
          pieces.push_back(arena.bar());
          pieces.push_back(arena.break1());
          break;
        case Joiner::kHardLine:
          pieces.push_back(arena.hard_line());
          break;
      }
    }
  }
  return ConcatN(arena, pieces);
}

DocRef Fmt(const BuiltinTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  return arena.MakeText(BuiltinTypeToString(n.builtin_type()));
}

DocRef Fmt(const ArrayTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  DocRef elem = Fmt(*n.element_type(), comments, arena);
  DocRef dim = Fmt(*n.dim(), comments, arena);
  return ConcatNGroup(arena, {elem, arena.obracket(), dim, arena.cbracket()});
}

static DocRef FmtTypeAnnotationPtr(const TypeAnnotation* n,
                                   const Comments& comments, DocArena& arena) {
  XLS_CHECK(n != nullptr);
  return Fmt(*n, comments, arena);
}

DocRef Fmt(const TupleTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  std::vector<DocRef> pieces = {arena.oparen()};
  pieces.push_back(FmtJoin<const TypeAnnotation*>(
      n.members(), Joiner::kCommaSpace, FmtTypeAnnotationPtr, comments, arena));
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const TypeRef& n, const Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const TypeAlias* n) { return arena.MakeText(n->identifier()); },
          [&](const StructDef* n) { return arena.MakeText(n->identifier()); },
          [&](const EnumDef* n) { return arena.MakeText(n->identifier()); },
          [&](const ColonRef* n) { return Fmt(*n, comments, arena); },
      },
      n.type_definition());
}

DocRef Fmt(const TypeRefTypeAnnotation& n, const Comments& comments,
           DocArena& arena) {
  std::vector<DocRef> pieces = {Fmt(*n.type_ref(), comments, arena)};
  if (!n.parametrics().empty()) {
    pieces.push_back(arena.oangle());
    pieces.push_back(FmtJoin<ExprOrType>(absl::MakeConstSpan(n.parametrics()),
                                         Joiner::kCommaSpace, FmtExprOrType,
                                         comments, arena));
    pieces.push_back(arena.cangle());
  }

  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const TypeAnnotation& n, const Comments& comments, DocArena& arena) {
  if (auto* t = dynamic_cast<const BuiltinTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }
  if (auto* t = dynamic_cast<const TupleTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }
  if (auto* t = dynamic_cast<const ArrayTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }
  if (auto* t = dynamic_cast<const TypeRefTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }

  XLS_LOG(FATAL) << "handle type annotation: " << n.ToString()
                 << " type: " << n.GetNodeTypeName();
}

DocRef Fmt(const TypeAlias& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  if (n.is_public()) {
    pieces.push_back(arena.Make(Keyword::kPub));
    pieces.push_back(arena.space());
  }
  pieces.push_back(arena.Make(Keyword::kType));
  pieces.push_back(arena.space());
  pieces.push_back(arena.MakeText(n.identifier()));
  pieces.push_back(arena.space());
  pieces.push_back(arena.equals());
  pieces.push_back(arena.break1());
  pieces.push_back(Fmt(*n.type_annotation(), comments, arena));
  pieces.push_back(arena.semi());
  return ConcatNGroup(arena, pieces);
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
  std::vector<DocRef> pieces;
  if (TypeAnnotation* t = n.type_annotation()) {
    pieces.push_back(Fmt(*t, comments, arena));
    pieces.push_back(arena.colon());
  }
  pieces.push_back(arena.obracket());
  pieces.push_back(arena.break0());
  for (const Expr* member : n.members()) {
    pieces.push_back(Fmt(*member, comments, arena));
    pieces.push_back(arena.comma());
    pieces.push_back(arena.break1());
  }
  if (n.has_ellipsis()) {
    pieces.push_back(arena.MakeText("..."));
  } else {
    pieces.pop_back();
    pieces.pop_back();
  }
  pieces.push_back(arena.cbracket());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Attr& n, const Comments& comments, DocArena& arena) {
  Precedence op_precedence = n.GetPrecedence();
  const Expr& lhs = *n.lhs();
  Precedence lhs_precedence = lhs.GetPrecedence();
  std::vector<DocRef> pieces;
  if (WeakerThan(lhs_precedence, op_precedence)) {
    pieces.push_back(arena.oparen());
    pieces.push_back(Fmt(lhs, comments, arena));
    pieces.push_back(arena.cparen());
  } else {
    pieces.push_back(Fmt(lhs, comments, arena));
  }
  pieces.push_back(arena.dot());
  pieces.push_back(arena.MakeText(std::string{n.attr()}));
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Binop& n, const Comments& comments, DocArena& arena) {
  Precedence op_precedence = n.GetPrecedence();
  const Expr& lhs = *n.lhs();
  const Expr& rhs = *n.rhs();
  Precedence lhs_precedence = lhs.GetPrecedence();

  std::vector<DocRef> pieces;

  auto emit = [&](const Expr& e, bool parens) {
    if (parens) {
      pieces.push_back(arena.oparen());
      pieces.push_back(Fmt(e, comments, arena));
      pieces.push_back(arena.cparen());
    } else {
      pieces.push_back(Fmt(e, comments, arena));
    }
  };

  if (WeakerThan(lhs_precedence, op_precedence)) {
    // We have to parenthesize the LHS.
    emit(lhs, /*parens=*/true);
  } else if (n.binop_kind() == BinopKind::kLt &&
             lhs.kind() == AstNodeKind::kCast && !lhs.in_parens()) {
    // If there is an open angle bracket, and the LHS is suffixed with a type,
    // we parenthesize it to avoid ambiguity; e.g.
    //
    //    foo as bar < baz
    //           ^~~~~~~~^
    //
    // We don't know whether `bar<baz` is the start of a parametric type
    // instantiation, so we force conservative parenthesization:
    //
    //    (foo as bar) < baz
    emit(lhs, /*parens=*/true);
  } else {
    emit(lhs, /*parens=*/false);
  }

  pieces.push_back(arena.break1());
  pieces.push_back(arena.MakeText(BinopKindFormat(n.binop_kind())));
  pieces.push_back(arena.break1());

  if (WeakerThan(rhs.GetPrecedence(), op_precedence)) {
    emit(rhs, /*parens=*/true);
  } else {
    emit(rhs, /*parens=*/false);
  }

  return ConcatNGroup(arena, pieces);
}

// Note: we only add leading/trailing spaces in the block if add_curls is true.
static DocRef FmtBlock(const Block& n, const Comments& comments,
                       DocArena& arena, bool add_curls,
                       bool force_multiline = false) {
  if (n.statements().empty()) {
    if (add_curls) {
      return ConcatNGroup(arena,
                          {arena.ocurl(), arena.break0(), arena.ccurl()});
    }
    return arena.break0();
  }

  // We only want to flatten single-statement blocks -- multi-statement blocks
  // we always make line breaks between the statements.
  if (n.statements().size() == 1 && !force_multiline) {
    std::vector<DocRef> pieces;
    if (add_curls) {
      pieces = {arena.ocurl(), arena.break1()};
    }

    pieces.push_back(Fmt(*n.statements()[0], comments, arena));

    if (n.trailing_semi()) {
      pieces.push_back(arena.semi());
    }
    if (add_curls) {
      pieces.push_back(arena.break1());
      pieces.push_back(arena.ccurl());
    }
    return arena.MakeNest(ConcatNGroup(arena, pieces));
  }

  // Emit a '{' then nest to emit statements with semis, then emit a '}' outside
  // the nesting.
  std::vector<DocRef> top;

  if (add_curls) {
    top.push_back(arena.ocurl());
    top.push_back(arena.hard_line());
  }

  std::vector<DocRef> nested;
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
  if (add_curls) {
    top.push_back(arena.hard_line());
    top.push_back(arena.ccurl());
  }

  return ConcatNGroup(arena, top);
}

DocRef Fmt(const Block& n, const Comments& comments, DocArena& arena) {
  return FmtBlock(n, comments, arena, /*add_curls=*/true);
}

DocRef Fmt(const Cast& n, const Comments& comments, DocArena& arena) {
  DocRef lhs = Fmt(*n.expr(), comments, arena);

  Precedence arg_precedence = n.expr()->GetPrecedence();
  if (WeakerThan(arg_precedence, Precedence::kAs)) {
    lhs = ConcatN(arena, {arena.oparen(), lhs, arena.cparen()});
  }

  return ConcatNGroup(
      arena, {lhs, arena.space(), arena.Make(Keyword::kAs), arena.break1(),
              Fmt(*n.type_annotation(), comments, arena)});
}

DocRef Fmt(const ChannelDecl& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle channel decl: " << n.ToString();
}

DocRef Fmt(const ColonRef& n, const Comments& comments, DocArena& arena) {
  DocRef subject = absl::visit(
      Visitor{[&](const NameRef* n) { return Fmt(*n, comments, arena); },
              [&](const ColonRef* n) { return Fmt(*n, comments, arena); }},
      n.subject());

  return ConcatNGroup(arena,
                      {subject, arena.colon_colon(), arena.MakeText(n.attr())});
}

DocRef Fmt(const For& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.Make(Keyword::kFor),
      arena.space(),
      Fmt(*n.names(), comments, arena),
  };

  if (n.type_annotation() != nullptr) {
    pieces.push_back(arena.colon());
    pieces.push_back(arena.space());
    pieces.push_back(Fmt(*n.type_annotation(), comments, arena));
  }

  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(Keyword::kIn));
  pieces.push_back(arena.space());
  pieces.push_back(Fmt(*n.iterable(), comments, arena));
  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());

  std::vector<DocRef> body_pieces;
  body_pieces.push_back(arena.hard_line());
  body_pieces.push_back(FmtBlock(*n.body(), comments, arena,
                                 /*add_curls=*/false,
                                 /*force_multiline=*/true));
  body_pieces.push_back(arena.hard_line());
  body_pieces.push_back(arena.ccurl());
  body_pieces.push_back(arena.oparen());
  body_pieces.push_back(Fmt(*n.init(), comments, arena));
  body_pieces.push_back(arena.cparen());

  return arena.MakeConcat(ConcatNGroup(arena, pieces),
                          ConcatN(arena, body_pieces));
}

DocRef Fmt(const FormatMacro& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.MakeText(n.macro()),
      arena.oparen(),
      arena.MakeText(
          absl::StrCat("\"", StepsToXlsFormatString(n.format()), "\"")),
      arena.comma(),
      arena.break1(),
  };
  pieces.push_back(FmtJoin<const Expr*>(n.args(), Joiner::kCommaSpace,
                                        FmtExprPtr, comments, arena));
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Slice& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;

  if (n.start() != nullptr) {
    pieces.push_back(Fmt(*n.start(), comments, arena));
  }
  pieces.push_back(arena.colon());
  if (n.limit() != nullptr) {
    pieces.push_back(Fmt(*n.limit(), comments, arena));
  }
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const WidthSlice& n, const Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 Fmt(*n.start(), comments, arena),
                                 arena.break0(),
                                 arena.plus_colon(),
                                 arena.break0(),
                                 Fmt(*n.width(), comments, arena),
                             });
}

static DocRef Fmt(const IndexRhs& n, const Comments& comments,
                  DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return Fmt(*n, comments, arena); },
          [&](const Slice* n) { return Fmt(*n, comments, arena); },
          [&](const WidthSlice* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef Fmt(const Index& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  if (WeakerThan(n.lhs()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena.oparen());
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
    pieces.push_back(arena.cparen());
  } else {
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
  }
  pieces.push_back(arena.obracket());
  pieces.push_back(Fmt(n.rhs(), comments, arena));
  pieces.push_back(arena.cbracket());
  return ConcatNGroup(arena, pieces);
}

DocRef FmtExprOrType(const ExprOrType& n, const Comments& comments,
                     DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return Fmt(*n, comments, arena); },
          [&](const TypeAnnotation* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef Fmt(const Invocation& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {
      Fmt(*n.callee(), comments, arena),
  };
  if (!n.explicit_parametrics().empty()) {
    pieces.push_back(arena.oangle());
    pieces.push_back(FmtJoin<ExprOrType>(
        absl::MakeConstSpan(n.explicit_parametrics()), Joiner::kCommaSpace,
        FmtExprOrType, comments, arena));
    pieces.push_back(arena.cangle());
  }
  pieces.push_back(arena.oparen());
  pieces.push_back(FmtJoin<const Expr*>(n.args(), Joiner::kCommaSpace,
                                        FmtExprPtr, comments, arena));
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

static DocRef FmtNameDefTreePtr(const NameDefTree* n, const Comments& comments,
                                DocArena& arena) {
  return Fmt(*n, comments, arena);
}

static DocRef Fmt(const MatchArm& n, const Comments& comments,
                  DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(
      FmtJoin<const NameDefTree*>(n.patterns(), Joiner::kSpaceBarBreak,
                                  FmtNameDefTreePtr, comments, arena));
  pieces.push_back(arena.space());
  pieces.push_back(arena.fat_arrow());
  pieces.push_back(arena.break1());
  pieces.push_back(Fmt(*n.expr(), comments, arena));
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Match& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(ConcatNGroup(
      arena,
      {arena.Make(Keyword::kMatch), arena.space(),
       Fmt(n.matched(), comments, arena), arena.space(), arena.ocurl()}));

  pieces.push_back(arena.hard_line());

  for (const MatchArm* arm : n.arms()) {
    pieces.push_back(arena.MakeNest(Fmt(*arm, comments, arena)));
    pieces.push_back(arena.comma());
    pieces.push_back(arena.hard_line());
  }

  pieces.push_back(arena.ccurl());
  return ConcatN(arena, pieces);
}

DocRef Fmt(const Spawn& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle spawn: " << n.ToString();
}

DocRef Fmt(const XlsTuple& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;

  for (size_t i = 0; i < n.members().size(); ++i) {
    const Expr* member = n.members()[i];
    DocRef member_doc = Fmt(*member, comments, arena);
    if (i + 1 == n.members().size()) {
      pieces.push_back(arena.MakeGroup(member_doc));
      pieces.push_back(arena.break0());
    } else {
      pieces.push_back(ConcatNGroup(arena, {member_doc, arena.comma()}));
      pieces.push_back(arena.break1());
    }
  }

  std::vector<DocRef> top = {arena.oparen()};
  top.push_back(arena.MakeAlign(ConcatNGroup(arena, pieces)));
  top.push_back(arena.cparen());
  return ConcatNGroup(arena, top);
}

static DocRef Fmt(const StructRef& n, const Comments& comments,
                  DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const StructDef* n) { return arena.MakeText(n->identifier()); },
          [&](const ColonRef* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

// Note: this does not put any spacing characters after the '{' so we can
// appropriately handle the case of an empty struct having no spacing in its
// `S {}` style construct.
static DocRef FmtStructLeader(const StructRef& struct_ref,
                              const Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 Fmt(struct_ref, comments, arena),
                                 arena.break1(),
                                 arena.ocurl(),
                             });
}

static DocRef FmtStructMembers(
    absl::Span<const std::pair<std::string, Expr*>> members,
    const Comments& comments, DocArena& arena) {
  return FmtJoin<std::pair<std::string, Expr*>>(
      members, Joiner::kCommaBreak1,
      [](const auto& member, const Comments& comments, DocArena& arena) {
        const auto& [name, expr] = member;
        return ConcatNGroup(
            arena, {arena.MakeText(name), arena.colon(), arena.break1(),
                    Fmt(*expr, comments, arena)});
      },
      comments, arena);
}

DocRef Fmt(const StructInstance& n, const Comments& comments, DocArena& arena) {
  DocRef leader = FmtStructLeader(n.struct_def(), comments, arena);

  if (n.GetUnorderedMembers().empty()) {  // empty struct instance
    return arena.MakeConcat(leader, arena.ccurl());
  }

  DocRef body_pieces =
      FmtStructMembers(n.GetUnorderedMembers(), comments, arena);

  return ConcatNGroup(arena,
                      {leader, arena.break1(), arena.MakeNest(body_pieces),
                       arena.break1(), arena.ccurl()});
}

DocRef Fmt(const SplatStructInstance& n, const Comments& comments,
           DocArena& arena) {
  DocRef leader = FmtStructLeader(n.struct_ref(), comments, arena);
  if (n.members().empty()) {
    return ConcatNGroup(arena, {leader, arena.break1(), arena.dot_dot(),
                                Fmt(*n.splatted(), comments, arena),
                                arena.break1(), arena.ccurl()});
  }

  DocRef body_pieces = FmtStructMembers(n.members(), comments, arena);

  return ConcatNGroup(
      arena,
      {leader, arena.break1(), arena.MakeNest(body_pieces), arena.comma(),
       arena.break1(), arena.dot_dot(), Fmt(*n.splatted(), comments, arena),
       arena.break1(), arena.ccurl()});

  XLS_LOG(FATAL) << "handle splat struct instance: " << n.ToString();
}

DocRef Fmt(const String& n, const Comments& comments, DocArena& arena) {
  return arena.MakeText(n.ToString());
}

// Creates a group that has the "test portion" of the conditional; i.e.
//
//  if <break1> $test_expr <break1> {
static DocRef MakeConditionalTestGroup(const Conditional& n,
                                       const Comments& comments,
                                       DocArena& arena) {
  return ConcatNGroup(arena, {
                                 arena.Make(Keyword::kIf),
                                 arena.break1(),
                                 Fmt(*n.test(), comments, arena),
                                 arena.break1(),
                                 arena.ocurl(),
                             });
}

// When there's an else-if, or multiple statements inside of the blocks, we
// force the formatting to be multi-line.
static DocRef FmtConditionalMultiline(const Conditional& n,
                                      const Comments& comments,
                                      DocArena& arena) {
  std::vector<DocRef> pieces = {
      MakeConditionalTestGroup(n, comments, arena), arena.hard_line(),
      FmtBlock(*n.consequent(), comments, arena, /*add_curls=*/false),
      arena.hard_line()};

  std::variant<Block*, Conditional*> alternate = n.alternate();
  while (std::holds_alternative<Conditional*>(alternate)) {
    Conditional* elseif = std::get<Conditional*>(alternate);
    alternate = elseif->alternate();
    pieces.push_back(arena.ccurl());
    pieces.push_back(arena.space());
    pieces.push_back(arena.Make(Keyword::kElse));
    pieces.push_back(arena.space());
    pieces.push_back(MakeConditionalTestGroup(*elseif, comments, arena));
    pieces.push_back(arena.hard_line());
    pieces.push_back(
        FmtBlock(*elseif->consequent(), comments, arena, /*add_curls=*/false));
    pieces.push_back(arena.hard_line());
  }

  XLS_CHECK(std::holds_alternative<Block*>(alternate));

  Block* else_block = std::get<Block*>(alternate);
  pieces.push_back(arena.ccurl());
  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(Keyword::kElse));
  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());
  pieces.push_back(arena.hard_line());
  pieces.push_back(FmtBlock(*else_block, comments, arena, /*add_curls=*/false));
  pieces.push_back(arena.hard_line());
  pieces.push_back(arena.ccurl());

  return ConcatN(arena, pieces);
}

DocRef Fmt(const Conditional& n, const Comments& comments, DocArena& arena) {
  // If there's an else-if clause or multi-statement blocks we force it to be
  // multi-line.
  if (n.HasElseIf() || n.HasMultiStatementBlocks()) {
    return FmtConditionalMultiline(n, comments, arena);
  }

  std::vector<DocRef> pieces = {
      MakeConditionalTestGroup(n, comments, arena),
      arena.break1(),
      FmtBlock(*n.consequent(), comments, arena, /*add_curls=*/false),
      arena.break1(),
  };

  XLS_CHECK(std::holds_alternative<Block*>(n.alternate()));
  const Block* else_block = std::get<Block*>(n.alternate());
  pieces.push_back(ConcatNGroup(
      arena, {arena.ccurl(), arena.break1(), arena.Make(Keyword::kElse),
              arena.break1(), arena.ocurl(), arena.break1()}));
  pieces.push_back(FmtBlock(*else_block, comments, arena, /*add_curls=*/false));
  pieces.push_back(arena.break1());
  pieces.push_back(arena.ccurl());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const ConstAssert& n, const Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 arena.MakeText("const_assert!("),
                                 Fmt(*n.arg(), comments, arena),
                                 arena.cparen(),
                             });
}

DocRef Fmt(const TupleIndex& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  if (WeakerThan(n.lhs()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena.oparen());
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
    pieces.push_back(arena.cparen());
  } else {
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
  }

  pieces.push_back(arena.dot());
  pieces.push_back(Fmt(*n.index(), comments, arena));
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const UnrollFor& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "handle unroll for: " << n.ToString();
}

DocRef Fmt(const ZeroMacro& n, const Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 arena.MakeText("zero!"),
                                 arena.oangle(),
                                 FmtExprOrType(n.type(), comments, arena),
                                 arena.cangle(),
                                 arena.oparen(),
                                 arena.cparen(),
                             });
}

DocRef Fmt(const Unop& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.MakeText(UnopKindToString(n.unop_kind()))};
  if (WeakerThan(n.operand()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena.oparen());
    pieces.push_back(Fmt(*n.operand(), comments, arena));
    pieces.push_back(arena.cparen());
  } else {
    pieces.push_back(Fmt(n.operand(), comments, arena));
  }
  return ConcatNGroup(arena, pieces);
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

DocRef Fmt(const Range& n, const Comments& comments, DocArena& arena) {
  return ConcatNGroup(
      arena, {Fmt(*n.start(), comments, arena), arena.break0(), arena.dot_dot(),
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

  std::vector<DocRef> leader_pieces = {
      arena.MakeText(n.is_const() ? "const" : "let"), break1,
      Fmt(*n.name_def_tree(), comments, arena)};
  if (const TypeAnnotation* t = n.type_annotation()) {
    leader_pieces.push_back(arena.colon());
    leader_pieces.push_back(break1);
    leader_pieces.push_back(Fmt(*t, comments, arena));
  }

  leader_pieces.push_back(break1);
  leader_pieces.push_back(arena.equals());
  leader_pieces.push_back(break1);

  DocRef leader = ConcatNGroup(arena, leader_pieces);
  DocRef body = Fmt(*n.rhs(), comments, arena);

  DocRef syntax = arena.MakeConcat(leader, body);

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
  for (size_t i = 0; i < params.size(); ++i) {
    const Param* param = params[i];
    DocRef type = Fmt(*param->type_annotation(), comments, arena);
    std::vector<DocRef> param_pieces = {arena.MakeText(param->identifier()),
                                        arena.break0(), arena.colon(),
                                        arena.break1(), type};
    if (i + 1 != params.size()) {
      param_pieces.push_back(arena.comma());
      param_pieces.push_back(arena.break1());
    }
    pieces.push_back(ConcatNGroup(arena, param_pieces));
  }
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

static DocRef Fmt(const ParametricBinding& n, const Comments& comments,
                  DocArena& arena) {
  return ConcatNGroup(
      arena, {arena.MakeText(n.identifier()), arena.colon(), arena.break1(),
              Fmt(*n.type_annotation(), comments, arena)});
}

static DocRef FmtParametricBindingPtr(const ParametricBinding* n,
                                      const Comments& comments,
                                      DocArena& arena) {
  XLS_CHECK(n != nullptr);
  return Fmt(*n, comments, arena);
}

DocRef Fmt(const Function& n, const Comments& comments, DocArena& arena) {
  DocRef fn = arena.MakeText("fn");
  DocRef name = arena.MakeText(n.identifier());

  DocRef params = FmtParams(n.params(), comments, arena);

  std::vector<DocRef> signature_pieces = {fn, arena.break1(), name};

  if (n.IsParametric()) {
    signature_pieces.push_back(
        ConcatNGroup(arena, {arena.oangle(),
                             FmtJoin<const ParametricBinding*>(
                                 n.parametric_bindings(), Joiner::kCommaSpace,
                                 FmtParametricBindingPtr, comments, arena),
                             arena.cangle()}));
  }

  signature_pieces.push_back(arena.break0());
  signature_pieces.push_back(params);
  signature_pieces.push_back(arena.break1());

  if (n.return_type() != nullptr) {
    signature_pieces.push_back(arena.arrow());
    signature_pieces.push_back(arena.break1());
    signature_pieces.push_back(Fmt(*n.return_type(), comments, arena));
    signature_pieces.push_back(arena.break1());
  }

  signature_pieces.push_back(arena.ocurl());

  // For empty function we don't put spaces between the curls.
  if (n.body()->empty()) {
    std::vector<DocRef> fn_pieces = {
        ConcatNGroup(arena, signature_pieces),
        FmtBlock(*n.body(), comments, arena, /*add_curls=*/false),
        arena.ccurl(),
    };

    return ConcatNGroup(arena, fn_pieces);
  }

  std::vector<DocRef> fn_pieces = {
      ConcatNGroup(arena, signature_pieces),
      arena.break1(),
      FmtBlock(*n.body(), comments, arena, /*add_curls=*/false),
      arena.break1(),
      arena.ccurl(),
  };

  return ConcatNGroup(arena, fn_pieces);
}

static DocRef Fmt(const Proc& n, const Comments& comments, DocArena& arena) {
  XLS_LOG(FATAL) << "proc: " << n.ToString();
}

static DocRef Fmt(const TestFunction& n, const Comments& comments,
                  DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena.MakeText("#[test]"));
  pieces.push_back(arena.hard_line());
  pieces.push_back(Fmt(*n.fn(), comments, arena));
  return ConcatN(arena, pieces);
}

static DocRef Fmt(const TestProc& n, const Comments& comments,
                  DocArena& arena) {
  XLS_LOG(FATAL) << "test proc: " << n.ToString();
}

static DocRef Fmt(const QuickCheck& n, const Comments& comments,
                  DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena.MakeText("#[quickcheck]"));
  pieces.push_back(arena.hard_line());
  pieces.push_back(Fmt(*n.f(), comments, arena));
  return ConcatN(arena, pieces);
}

static DocRef Fmt(const StructDef& n, const Comments& comments,
                  DocArena& arena) {
  std::vector<DocRef> pieces;
  if (n.is_public()) {
    pieces.push_back(arena.Make(Keyword::kPub));
    pieces.push_back(arena.space());
  }
  pieces.push_back(arena.Make(Keyword::kStruct));
  pieces.push_back(arena.space());
  pieces.push_back(arena.MakeText(n.identifier()));

  if (!n.parametric_bindings().empty()) {
    pieces.push_back(arena.oangle());
    pieces.push_back(FmtJoin<const ParametricBinding*>(
        n.parametric_bindings(), Joiner::kCommaSpace, FmtParametricBindingPtr,
        comments, arena));
    pieces.push_back(arena.cangle());
  }

  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());

  if (!n.members().empty()) {
    pieces.push_back(arena.break1());

    std::vector<DocRef> body_pieces;
    for (size_t i = 0; i < n.members().size(); ++i) {
      const auto& [name_def, type] = n.members()[i];
      body_pieces.push_back(arena.MakeText(name_def->identifier()));
      body_pieces.push_back(arena.colon());
      body_pieces.push_back(arena.space());
      body_pieces.push_back(Fmt(*type, comments, arena));
      if (i + 1 == n.members().size()) {
        body_pieces.push_back(arena.MakeFlatChoice(/*on_flat=*/arena.empty(),
                                                   /*on_break=*/arena.comma()));
      } else {
        body_pieces.push_back(arena.comma());
        body_pieces.push_back(arena.break1());
      }
    }

    pieces.push_back(arena.MakeNest(ConcatN(arena, body_pieces)));
    pieces.push_back(arena.break1());
  }

  pieces.push_back(arena.ccurl());
  return ConcatNGroup(arena, pieces);
}

static DocRef Fmt(const ConstantDef& n, const Comments& comments,
                  DocArena& arena) {
  std::vector<DocRef> pieces;
  if (n.is_public()) {
    pieces.push_back(arena.Make(Keyword::kPub));
    pieces.push_back(arena.break1());
  }
  pieces.push_back(arena.Make(Keyword::kConst));
  pieces.push_back(arena.break1());
  pieces.push_back(arena.MakeText(n.identifier()));
  pieces.push_back(arena.break1());
  pieces.push_back(arena.equals());
  pieces.push_back(arena.break1());
  pieces.push_back(Fmt(*n.value(), comments, arena));
  pieces.push_back(arena.semi());
  return ConcatNGroup(arena, pieces);
}

static DocRef FmtEnumMember(const EnumMember& n, const Comments& comments,
                            DocArena& arena) {
  return ConcatNGroup(
      arena, {Fmt(*n.name_def, comments, arena), arena.space(), arena.equals(),
              arena.break1(), Fmt(*n.value, comments, arena), arena.comma()});
}

static DocRef Fmt(const EnumDef& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  if (n.is_public()) {
    pieces.push_back(arena.Make(Keyword::kPub));
    pieces.push_back(arena.space());
  }
  pieces.push_back(arena.Make(Keyword::kEnum));
  pieces.push_back(arena.space());
  pieces.push_back(arena.MakeText(n.identifier()));

  pieces.push_back(arena.space());
  if (n.type_annotation() != nullptr) {
    pieces.push_back(arena.colon());
    pieces.push_back(arena.space());
    pieces.push_back(Fmt(*n.type_annotation(), comments, arena));
    pieces.push_back(arena.space());
  }

  pieces.push_back(arena.ocurl());
  pieces.push_back(arena.hard_line());

  DocRef nested = FmtJoin<EnumMember>(n.values(), Joiner::kHardLine,
                                      FmtEnumMember, comments, arena);

  pieces.push_back(arena.MakeNest(nested));
  pieces.push_back(arena.hard_line());
  pieces.push_back(arena.ccurl());
  return ConcatN(arena, pieces);
}

static DocRef Fmt(const Import& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> dotted_pieces;
  for (size_t i = 0; i < n.subject().size(); ++i) {
    const std::string& subject_part = n.subject()[i];
    DocRef this_doc_ref;
    if (i + 1 == n.subject().size()) {
      this_doc_ref = ConcatNGroup(arena, {arena.MakeText(subject_part)});
    } else {
      this_doc_ref = ConcatNGroup(
          arena, {arena.MakeText(subject_part), arena.dot(), arena.break0()});
    }
    dotted_pieces.push_back(this_doc_ref);
  }
  return ConcatNGroup(arena,
                      {arena.MakeText("import "),
                       arena.MakeAlign(ConcatNGroup(arena, dotted_pieces))});
}

static DocRef Fmt(const ModuleMember& n, const Comments& comments,
                  DocArena& arena) {
  return absl::visit(
      Visitor{[&](const Function* n) { return Fmt(*n, comments, arena); },
              [&](const Proc* n) { return Fmt(*n, comments, arena); },
              [&](const TestFunction* n) { return Fmt(*n, comments, arena); },
              [&](const TestProc* n) { return Fmt(*n, comments, arena); },
              [&](const QuickCheck* n) { return Fmt(*n, comments, arena); },
              [&](const TypeAlias* n) { return Fmt(*n, comments, arena); },
              [&](const StructDef* n) { return Fmt(*n, comments, arena); },
              [&](const ConstantDef* n) { return Fmt(*n, comments, arena); },
              [&](const EnumDef* n) { return Fmt(*n, comments, arena); },
              [&](const Import* n) { return Fmt(*n, comments, arena); },
              [&](const ConstAssert* n) {
                return arena.MakeConcat(Fmt(*n, comments, arena), arena.semi());
              }},
      n);
}

DocRef Fmt(const Expr& n, const Comments& comments, DocArena& arena) {
  FmtExprVisitor v(arena, comments);
  XLS_CHECK_OK(n.AcceptExpr(&v));
  DocRef result = v.result();
  if (n.in_parens()) {
    return ConcatNGroup(arena, {arena.oparen(), result, arena.cparen()});
  }
  return result;
}

DocRef Fmt(const Module& n, const Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  for (size_t i = 0; i < n.top().size(); ++i) {
    const auto& member = n.top()[i];
    pieces.push_back(Fmt(member, comments, arena));
    if (i + 1 == n.top().size()) {
      pieces.push_back(arena.hard_line());
    } else {
      pieces.push_back(arena.hard_line());
      pieces.push_back(arena.hard_line());
    }
  }

  return ConcatN(arena, pieces);
}

std::string AutoFmt(const Module& m, const Comments& comments,
                    int64_t text_width) {
  DocArena arena;
  DocRef ref = Fmt(m, comments, arena);
  return PrettyPrint(arena, ref, text_width);
}

}  // namespace xls::dslx
