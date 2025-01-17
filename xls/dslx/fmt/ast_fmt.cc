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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/format_disabler.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {
namespace {

// Note: if a comment doc is emitted (i.e. return value has_value()) it does not
// have a trailing hard-line. This is for consistency with other emission
// routines which generally don't emit any whitespace afterwards, just their
// doc.
std::optional<DocRef> EmitCommentsBetween(
    std::optional<Pos> start_pos, const Pos& limit_pos, Comments& comments,
    DocArena& arena, std::optional<Span>* last_comment_span) {
  if (!start_pos.has_value()) {
    start_pos = Pos(limit_pos.fileno(), 0, 0);
  }
  // Due to the hack in AdjustCommentLimit, we can end up looking for a comment
  // between the fictitious end of a comment and the end of a block. Just don't
  // return anything in that case.
  if (start_pos >= limit_pos) {
    return std::nullopt;
  }
  const Span span(start_pos.value(), limit_pos);

  const FileTable& file_table = arena.file_table();
  VLOG(3) << "Looking for comments in span: " << span.ToString(file_table);

  std::vector<DocRef> pieces;

  std::vector<const CommentData*> items = comments.GetComments(span);
  VLOG(3) << "Found " << items.size() << " comment data items";
  std::optional<Span> previous_comment_span;
  for (size_t i = 0; i < items.size(); ++i) {
    const CommentData* comment_data = items[i];
    comments.PlaceComment(comment_data);

    // If the previous comment line and this comment line are abutted (i.e.
    // contiguous lines with comments), we don't put a newline between them.
    if (previous_comment_span.has_value() &&
        previous_comment_span->start().lineno() + 1 !=
            comment_data->span.start().lineno()) {
      VLOG(3) << "previous comment span: "
              << previous_comment_span.value().ToString(file_table)
              << " this comment span: "
              << comment_data->span.ToString(file_table)
              << " -- inserting hard line";
      pieces.push_back(arena.hard_line());
    }

    pieces.push_back(arena.MakePrefixedReflow(
        "//",
        std::string{absl::StripTrailingAsciiWhitespace(comment_data->text)}));

    if (i + 1 != items.size()) {
      pieces.push_back(arena.hard_line());
    }

    previous_comment_span = comment_data->span;
    if (last_comment_span != nullptr) {
      *last_comment_span = comment_data->span;
    }
  }

  if (pieces.empty()) {
    return std::nullopt;
  }

  return ConcatN(arena, pieces);
}

// If there is a '.' modifier in the attribute given by s (e.g. if it's of the
// form "ProcName.config") strips off the trailing modifier and returns the
// stem.
//
// TODO(https://github.com/google/xls/issues/1029): 2023-12-05 Ideally we'd have
// proc references and avoid strange modifiers on identifiers.
std::string StripAnyDotModifier(std::string_view s) {
  CHECK_LE(std::count(s.begin(), s.end(), '.'), 1);
  // Check for special identifier for proc config, which is ProcName.config
  // internally, but in spawns we just want to say ProcName.
  if (auto pos = s.rfind('.'); pos != std::string::npos) {
    return std::string(s.substr(0, pos));
  }

  return std::string(s);
}

// Forward decls.
// keep-sorted start
DocRef Fmt(const ColonRef& n, Comments& comments, DocArena& arena);
DocRef Fmt(const Expr& n, Comments& comments, DocArena& arena);
DocRef Fmt(const NameDefTree& n, Comments& comments, DocArena& arena);
DocRef Fmt(const TypeAnnotation& n, Comments& comments, DocArena& arena);
DocRef FmtBlockedExprLeader(const Expr& e, Comments& comments, DocArena& arena);
DocRef FmtExpr(const Expr& n, Comments& comments, DocArena& arena,
               bool suppress_parens);
DocRef FmtExprOrType(const ExprOrType& n, Comments& comments, DocArena& arena);
// keep-sorted end

// A parametric argument, as in parametric instantiation:
//
//    f<FOO as u32>()
//      ^^^^^^^^^^~~~ parametric argument, expr or types can go here generally
DocRef FmtParametricArg(const ExprOrType& n, Comments& comments,
                        DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) {
            DocRef guts = Fmt(*n, comments, arena);
            if (dynamic_cast<const NameRef*>(n) != nullptr ||
                dynamic_cast<const ColonRef*>(n) != nullptr ||
                dynamic_cast<const Number*>(n) != nullptr) {
              return guts;  // No need for enclosing curlies.
            }
            return ConcatN(arena, {arena.ocurl(), guts, arena.ccurl()});
          },
          [&](const TypeAnnotation* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef FmtExprPtr(const Expr* n, Comments& comments, DocArena& arena) {
  CHECK(n != nullptr);
  return Fmt(*n, comments, arena);
}

enum class Joiner : uint8_t {
  kCommaSpace,
  kCommaBreak1,

  kCommaHardlineTrailingCommaAlways,

  // Separates via a comma and break1, but groups the element with its
  // delimiter. This is useful when we're packing member elements that we want
  // to be reflowed across lines.
  //
  // Note that, in this mode, if we span multiple lines, we'll put a trailing
  // comma as well.
  kCommaBreak1AsGroupTrailingCommaOnBreak,
  kCommaBreak1AsGroupTrailingCommaAlways,
  kCommaBreak1AsGroupNoTrailingComma,

  kSpaceBarBreak,
  kHardLine,
};

// Helper for doing a "join via comma space" pattern with doc refs.
//
// This elides the "joiner" being present after the last item.
template <typename T>
DocRef FmtJoin(absl::Span<const T> items, Joiner joiner,
               const std::function<DocRef(const T&, Comments&, DocArena&)>& fmt,
               Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  for (size_t i = 0; i < items.size(); ++i) {
    const T& item = items[i];

    // First we format the member into a doc and then decide the best way to put
    // it into the sequence.
    DocRef member = fmt(item, comments, arena);

    if (i + 1 != items.size()) {  // Not the last item.
      switch (joiner) {
        case Joiner::kCommaSpace:
          pieces.push_back(member);
          pieces.push_back(arena.comma());
          pieces.push_back(arena.space());
          break;
        case Joiner::kCommaHardlineTrailingCommaAlways:
          pieces.push_back(member);
          pieces.push_back(arena.comma());
          pieces.push_back(arena.hard_line());
          break;
        case Joiner::kCommaBreak1:
          pieces.push_back(member);
          pieces.push_back(arena.comma());
          pieces.push_back(arena.break1());
          break;
        case Joiner::kCommaBreak1AsGroupNoTrailingComma:
        case Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak:
        case Joiner::kCommaBreak1AsGroupTrailingCommaAlways: {
          std::vector<DocRef> this_pieces;
          if (i != 0) {  // If it's the first item we don't put a leading space.
            this_pieces.push_back(arena.break1());
          }
          this_pieces.push_back(member);
          this_pieces.push_back(arena.comma());
          pieces.push_back(ConcatNGroup(arena, this_pieces));
          break;
        }
        case Joiner::kSpaceBarBreak:
          pieces.push_back(member);
          pieces.push_back(arena.space());
          pieces.push_back(arena.bar());
          pieces.push_back(arena.break1());
          break;
        case Joiner::kHardLine:
          pieces.push_back(member);
          pieces.push_back(arena.hard_line());
          break;
      }
    } else {  // Last item, generally "no trailing delimiter".
      switch (joiner) {
        case Joiner::kCommaBreak1AsGroupNoTrailingComma:
        case Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak:
        case Joiner::kCommaBreak1AsGroupTrailingCommaAlways: {
          // Note: we only want to put a leading space in front of the last
          // element if the last element is not also the first element.
          if (i == 0) {
            pieces.push_back(member);
          } else {
            pieces.push_back(ConcatNGroup(arena, {arena.break1(), member}));
          }

          if (joiner == Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak) {
            // With this pattern if we're in break mode (implying we spanned
            // multiple lines), we allow a trailing comma.
            pieces.push_back(
                arena.MakeFlatChoice(arena.empty(), arena.comma()));
          } else if (joiner == Joiner::kCommaBreak1AsGroupTrailingCommaAlways) {
            pieces.push_back(arena.comma());
          }
          break;
        }
        case Joiner::kCommaHardlineTrailingCommaAlways:
          pieces.push_back(member);
          pieces.push_back(arena.comma());
          break;
        default:
          pieces.push_back(member);
          break;
      }
    }
  }
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const BuiltinTypeAnnotation& n, Comments& comments,
           DocArena& arena) {
  return arena.MakeText(BuiltinTypeToString(n.builtin_type()));
}

DocRef Fmt(const VerbatimNode& n, Comments& comments, DocArena& arena) {
  return Formatter(comments, arena).Format(n);
}

DocRef Fmt(const ArrayTypeAnnotation& n, Comments& comments, DocArena& arena) {
  DocRef elem = Fmt(*n.element_type(), comments, arena);
  DocRef dim = Fmt(*n.dim(), comments, arena);

  return ConcatNGroup(
      arena, {elem, arena.obracket(), arena.MakeAlign(dim), arena.cbracket()});
}

DocRef FmtTypeAnnotationPtr(const TypeAnnotation* n, Comments& comments,
                            DocArena& arena) {
  CHECK(n != nullptr);
  return Fmt(*n, comments, arena);
}

DocRef Fmt(const TupleTypeAnnotation& n, Comments& comments, DocArena& arena) {
  DocRef guts = FmtJoin<const TypeAnnotation*>(
      n.members(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
      FmtTypeAnnotationPtr, comments, arena);

  return ConcatNGroup(
      arena, {
                 arena.oparen(),
                 arena.MakeFlatChoice(
                     /*on_flat=*/guts,
                     /*on_break=*/ConcatNGroup(arena,
                                               {
                                                   arena.hard_line(),
                                                   arena.MakeNest(guts),
                                                   arena.hard_line(),
                                               })),
                 arena.cparen(),
             });
}

DocRef Fmt(const TypeRef& n, Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const ColonRef* n) { return Fmt(*n, comments, arena); },
          [&](const auto* n) { return arena.MakeText(n->identifier()); },
      },
      n.type_definition());
}

DocRef Fmt(const TypeRefTypeAnnotation& n, Comments& comments,
           DocArena& arena) {
  std::vector<DocRef> pieces = {Fmt(*n.type_ref(), comments, arena)};
  if (!n.parametrics().empty()) {
    pieces.push_back(arena.oangle());
    pieces.push_back(FmtJoin<ExprOrType>(absl::MakeConstSpan(n.parametrics()),
                                         Joiner::kCommaSpace, FmtParametricArg,
                                         comments, arena));
    pieces.push_back(arena.cangle());
  }

  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const ChannelTypeAnnotation& n, Comments& comments,
           DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.Make(Keyword::kChan),
      arena.oangle(),
      Fmt(*n.payload(), comments, arena),
      arena.cangle(),
  };
  if (n.dims().has_value()) {
    for (const Expr* dim : *n.dims()) {
      pieces.push_back(arena.obracket());
      pieces.push_back(Fmt(*dim, comments, arena));
      pieces.push_back(arena.cbracket());
    }
  }

  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(
      n.direction() == ChannelDirection::kIn ? Keyword::kIn : Keyword::kOut));
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const TypeAnnotation& n, Comments& comments, DocArena& arena) {
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
  if (auto* t = dynamic_cast<const ChannelTypeAnnotation*>(&n)) {
    return Fmt(*t, comments, arena);
  }
  if (auto* t = dynamic_cast<const SelfTypeAnnotation*>(&n)) {
    return arena.Make(Keyword::kSelfType);
  }

  LOG(FATAL) << "handle type annotation: " << n.ToString()
             << " type: " << n.GetNodeTypeName();
}

DocRef JoinWithAttr(std::optional<DocRef> attr, DocRef rest, DocArena& arena) {
  if (!attr) {
    return rest;
  }
  return arena.MakeConcat(*attr, rest);
}

DocRef FmtSvTypeAttribute(std::string_view name, DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena.MakeText("#"));
  pieces.push_back(arena.obracket());
  pieces.push_back(arena.MakeText("sv_type"));
  pieces.push_back(arena.oparen());
  pieces.push_back(arena.MakeText(absl::StrFormat("\"%s\"", name)));
  pieces.push_back(arena.cparen());
  pieces.push_back(arena.cbracket());
  pieces.push_back(arena.hard_line());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const NameDef& n, Comments& comments, DocArena& arena) {
  return arena.MakeText(n.identifier());
}

DocRef Fmt(const NameRef& n, Comments& comments, DocArena& arena) {
  return arena.MakeText(StripAnyDotModifier(n.identifier()));
}

DocRef Fmt(const Number& n, Comments& comments, DocArena& arena) {
  DocRef num_text;
  if (n.number_kind() == NumberKind::kCharacter) {
    // Note: we don't need to escape double quote because this is going to end
    // up in single quotes.
    std::string guts;
    if (n.text() == "\"") {
      guts = "\"";  // Double quotes don't need to be escaped.
    } else {
      guts = Escape(n.text());  // Everything else we do normal C-string escape.
    }
    num_text = arena.MakeText(absl::StrFormat("'%s'", guts));
  } else {
    num_text = arena.MakeText(n.text());
  }
  if (const TypeAnnotation* type = n.type_annotation()) {
    return ConcatNGroup(arena,
                        {Fmt(*type, comments, arena), arena.colon(), num_text});
  }
  return num_text;
}

DocRef Fmt(const WildcardPattern& n, Comments& comments, DocArena& arena) {
  return arena.underscore();
}

DocRef Fmt(const RestOfTuple& n, Comments& comments, DocArena& arena) {
  return arena.dot_dot();
}

DocRef MakeArrayLeader(const Array& n, Comments& comments, DocArena& arena) {
  const TypeAnnotation* t = n.type_annotation();
  if (t == nullptr) {
    return arena.obracket();
  }
  std::vector<DocRef> pieces;
  pieces.push_back(Fmt(*t, comments, arena));
  pieces.push_back(arena.colon());
  pieces.push_back(arena.obracket());
  return ConcatN(arena, pieces);
}

DocRef FmtFlatBody(const Array& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> flat_pieces;
  flat_pieces.push_back(FmtJoin<const Expr*>(n.members(), Joiner::kCommaSpace,
                                             FmtExprPtr, comments, arena));
  if (n.has_ellipsis()) {
    // Note: while zero members with ellipsis is invalid at type checking, we
    // may choose not to flag it as a parse-time error, in which case we could
    // have it in the AST.
    if (!n.members().empty()) {
      flat_pieces.push_back(arena.comma());
    }

    flat_pieces.push_back(arena.space());
    flat_pieces.push_back(arena.MakeText("..."));
  }
  flat_pieces.push_back(arena.cbracket());
  return ConcatN(arena, flat_pieces);
}

DocRef FmtBreakBody(const Array& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> rest;
  rest.push_back(arena.break0());

  std::vector<DocRef> member_pieces;
  member_pieces.push_back(FmtJoin<const Expr*>(
      n.members(), Joiner::kCommaBreak1AsGroupTrailingCommaAlways, FmtExprPtr,
      comments, arena));

  if (n.has_ellipsis()) {
    member_pieces.push_back(
        ConcatNGroup(arena, {arena.break1(), arena.MakeText("...")}));
  }

  DocRef inner = ConcatNGroup(arena, member_pieces);
  rest.push_back(arena.MakeFlatChoice(inner, arena.MakeNest(inner)));
  rest.push_back(arena.break0());
  rest.push_back(arena.cbracket());

  return ConcatNGroup(arena, rest);
}

DocRef Fmt(const Array& n, Comments& comments, DocArena& arena) {
  DocRef on_break_body = FmtBreakBody(n, comments, arena);
  DocRef on_flat_body = FmtFlatBody(n, comments, arena);

  DocRef body = arena.MakeGroup(arena.MakeFlatChoice(
      /*on_flat=*/on_flat_body, /*on_break=*/on_break_body));
  return arena.MakeConcat(MakeArrayLeader(n, comments, arena), body);
}

DocRef Fmt(const Attr& n, Comments& comments, DocArena& arena) {
  Precedence op_precedence = n.GetPrecedenceWithoutParens();
  const Expr& lhs = *n.lhs();
  Precedence lhs_precedence = lhs.GetPrecedence();
  std::vector<DocRef> pieces;
  if (WeakerThan(lhs_precedence, op_precedence) && IsInfix(lhs_precedence)) {
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

std::optional<DocRef> EmitCommentsNested(const Pos start, const Pos limit,
                                         Comments& comments, DocArena& arena) {
  std::vector<const CommentData*> items =
      comments.GetComments(Span(start, limit));
  if (items.empty()) {
    return std::nullopt;
  }

  std::vector<DocRef> pieces;
  // Add the first comment "in line"
  auto first =
      EmitCommentsBetween(start, items[0]->span.limit(), comments, arena,
                          /*last_comment_span=*/nullptr);
  pieces.push_back(*first);
  pieces.push_back(arena.hard_line());
  if (items.size() > 1) {
    // Add the nth through last comment as a new nested document.
    auto nested_comments =
        EmitCommentsBetween(items[1]->span.start(), limit, comments, arena,
                            /*last_comment_span=*/nullptr);
    // EmitCommentsBetween doesn't indent (or nest) the 2nd through Nth
    // comments, so we have to do it manually here.
    pieces.push_back(arena.MakeNest(*nested_comments));
    pieces.push_back(arena.hard_line());
  }
  return ConcatN(arena, pieces);
}

DocRef Fmt(const Binop& n, Comments& comments, DocArena& arena) {
  Precedence op_precedence = n.GetPrecedenceWithoutParens();
  const Expr& lhs = *n.lhs();
  const Expr& rhs = *n.rhs();
  Precedence lhs_precedence = lhs.GetPrecedence();

  auto emit = [&](const Expr& e, bool parens, std::vector<DocRef>& pieces) {
    if (parens) {
      pieces.push_back(arena.oparen());
      pieces.push_back(Fmt(e, comments, arena));
      pieces.push_back(arena.cparen());
    } else {
      pieces.push_back(Fmt(e, comments, arena));
    }
  };

  std::vector<DocRef> lhs_pieces;
  if (WeakerThan(lhs_precedence, op_precedence)) {
    // We have to parenthesize the LHS.
    emit(lhs, /*parens=*/true, lhs_pieces);
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
    emit(lhs, /*parens=*/true, lhs_pieces);
  } else {
    emit(lhs, /*parens=*/false, lhs_pieces);
  }

  DocRef lhs_ref = ConcatN(arena, lhs_pieces);
  bool nest_rhs = false;

  // If there are comments between the LHS and the operator, we want to emit
  // them before the operator.
  if (std::optional<DocRef> comments_doc = EmitCommentsNested(
          lhs.span().limit(), n.op_span().start(), comments, arena)) {
    lhs_ref = ConcatN(arena, {lhs_ref, arena.space(), *comments_doc});
    nest_rhs = true;
  }

  bool emitted_op = false;
  // If there are comments between the operator and the RHS, emit them now.
  if (nest_rhs) {
    if (std::optional<DocRef> comments_doc = EmitCommentsBetween(
            n.op_span().limit(), rhs.span().start(), comments, arena,
            /*last_comment_span=*/nullptr)) {
      // If the RHS is already being nested, don't nest the comments. probably.
      lhs_ref = ConcatN(
          arena,
          {lhs_ref,
           arena.MakeNest(ConcatN(
               arena, {arena.MakeText(BinopKindFormat(n.binop_kind())),
                       arena.space(), *comments_doc, arena.hard_line()}))});
      emitted_op = true;
    }
  } else if (std::optional<DocRef> comments_doc = EmitCommentsNested(
                 n.op_span().limit(), rhs.span().start(), comments, arena)) {
    // The space is needed since we didn't nest the RHS yet
    lhs_ref = ConcatN(arena, {lhs_ref, arena.space(),
                              arena.MakeText(BinopKindFormat(n.binop_kind())),
                              arena.space(), *comments_doc});
    emitted_op = true;
    nest_rhs = true;
  }

  std::vector<DocRef> rhs_pieces;
  if (WeakerThan(rhs.GetPrecedence(), op_precedence)) {
    emit(rhs, /*parens=*/true, rhs_pieces);
  } else {
    emit(rhs, /*parens=*/false, rhs_pieces);
  }

  // Note: we associate the operator with the RHS in a group so that chained
  // binary operations can appropriately pack into lines -- if we associate with
  // the left hand term then when we enter break mode the nested RHS terms all
  // end up on their own lines. See `ModuleFmtTest.NestedBinopLogicalOr` for a
  // case study.
  DocRef rhs_ref = ConcatN(arena, rhs_pieces);
  std::vector<DocRef> more_rhs_pieces;
  if (!nest_rhs) {
    // If we didn't nest the RHS, we need to add a space to separate it from the
    // operator.
    more_rhs_pieces.push_back(arena.space());
  }
  if (!emitted_op) {
    // If we didn't emit the operator, we need to add it now.
    more_rhs_pieces.push_back(arena.MakeText(BinopKindFormat(n.binop_kind())));
    more_rhs_pieces.push_back(arena.break1());
  }
  more_rhs_pieces.push_back(rhs_ref);
  rhs_ref = ConcatNGroup(arena, more_rhs_pieces);
  if (nest_rhs) {
    rhs_ref = arena.MakeNest(rhs_ref);
  }

  return ConcatN(arena, {
                            lhs_ref,
                            rhs_ref,
                        });
}

// EOL-terminated comments we say, in the AST, that the entity's limit was at
// the end of the line. Technically the limit() pos gives you the start of the
// next line, but for purposes of line spacing we want it to reflect the very
// end of the current line.
//
// Since we don't record the limit column number for each line to query here, we
// just use an absurdly large number (int32_t max).
Pos AdjustCommentLimit(const Span& comment_span, DocArena& arena,
                       DocRef comment_doc) {
  CHECK_EQ(comment_span.limit().colno(), 0);
  CHECK_GT(comment_span.limit().lineno(), 0);
  return Pos(comment_span.start().fileno(), comment_span.limit().lineno() - 1,
             std::numeric_limits<int32_t>::max());
}

// Looks for inline comments after the `prev_limit` and adds relevant `DocRef`
// to `pieces`. Returns `last_entity_pos`, updated if comments were found.
Pos CollectInlineComments(const Pos& prev_limit, const Pos& last_entity_pos,
                          Comments& comments, DocArena& arena,
                          std::vector<DocRef>& pieces,
                          std::optional<Span> last_comment_span) {
  const Pos next_line(prev_limit.fileno(), prev_limit.lineno() + 1, 0);
  if (std::optional<DocRef> comments_doc = EmitCommentsBetween(
          last_entity_pos, next_line, comments, arena, &last_comment_span)) {
    VLOG(3) << "Saw inline comment: "
            << arena.ToDebugString(comments_doc.value())
            << " last_comment_span: "
            << last_comment_span.value().ToString(arena.file_table());
    pieces.push_back(arena.space());
    pieces.push_back(arena.space());
    pieces.push_back(arena.MakeAlign(comments_doc.value()));

    return AdjustCommentLimit(last_comment_span.value(), arena,
                              comments_doc.value());
  }
  return last_entity_pos;
}

DocRef Fmt(const Statement& n, Comments& comments, DocArena& arena,
           bool trailing_semi) {
  return Formatter(comments, arena).Format(n, trailing_semi);
}

DocRef FmtSingleStatementBlockInline(const StatementBlock& n,
                                     Comments& comments, bool add_curls,
                                     DocArena& arena) {
  std::vector<DocRef> pieces;
  if (add_curls) {
    pieces = {arena.ocurl(), arena.break1()};
  }

  pieces.push_back(Fmt(*n.statements()[0], comments, arena,
                       /*trailing_semi=*/n.trailing_semi()));

  if (add_curls) {
    pieces.push_back(arena.break1());
    pieces.push_back(arena.ccurl());
  }
  DocRef block_group = ConcatNGroup(arena, pieces);
  return arena.MakeFlatChoice(block_group, arena.MakeNest(block_group));
}

// Note: we only add leading/trailing spaces in the block if add_curls is true.
DocRef FmtBlock(const StatementBlock& n, Comments& comments, DocArena& arena,
                bool add_curls, bool force_multiline = false) {
  bool has_comments = comments.HasComments(n.span());

  if (n.statements().empty() && !has_comments) {
    if (add_curls) {
      return ConcatNGroup(arena,
                          {arena.ocurl(), arena.break0(), arena.ccurl()});
    }
    return arena.break0();
  }

  // We only want to flatten single-statement blocks -- multi-statement blocks
  // we always make line breaks between the statements.
  if (n.statements().size() == 1 && !force_multiline && !has_comments) {
    return FmtSingleStatementBlockInline(n, comments, add_curls, arena);
  }

  // Emit a '{' then nest to emit statements with semis, then emit a '}' outside
  // the nesting.
  std::vector<DocRef> top;

  if (add_curls) {
    top.push_back(arena.ocurl());
    top.push_back(arena.hard_line());
  }

  // For our initial condition, we say the last entity we emitted is right after
  // the start of the block (i.e. the open curl).
  const Pos start_entity_pos = n.span().start().BumpCol();
  Pos last_entity_pos = start_entity_pos;

  std::vector<DocRef> statements;
  bool last_stmt_was_verbatim = false;
  for (size_t i = 0; i < n.statements().size(); ++i) {
    std::vector<DocRef> stmt_pieces;
    const Statement* stmt = n.statements()[i];
    last_stmt_was_verbatim =
        std::holds_alternative<VerbatimNode*>(stmt->wrapped());

    // Get the start position for the statement.
    std::optional<Span> stmt_span = stmt->GetSpan();
    CHECK(stmt_span.has_value()) << stmt->ToString();
    const Pos& stmt_start = stmt_span->start();
    const Pos& stmt_limit = stmt_span->limit();

    VLOG(5) << "stmt: `" << stmt->ToString()
            << "` span: " << stmt_span.value().ToString(arena.file_table())
            << " last_entity_pos: " << last_entity_pos;

    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc = EmitCommentsBetween(
            last_entity_pos, stmt_start, comments, arena, &last_comment_span)) {
      VLOG(5) << "emitting comment ahead of: `" << stmt->ToString() << "`"
              << " last entity position: " << last_entity_pos
              << " last_comment_span: "
              << last_comment_span.value().ToString(arena.file_table());
      // If there's a line break between the last entity and this comment, we
      // retain it in the output (i.e. in paragraph style).
      if (last_entity_pos != start_entity_pos &&
          last_entity_pos.lineno() + 1 < last_comment_span->start().lineno()) {
        stmt_pieces.push_back(arena.hard_line());
      }

      stmt_pieces.push_back(comments_doc.value());
      stmt_pieces.push_back(arena.hard_line());

      last_entity_pos = AdjustCommentLimit(last_comment_span.value(), arena,
                                           comments_doc.value());

      // See if we want a line break between the comment we just emitted and the
      // statement we're about to emit.
      if (last_entity_pos.lineno() + 1 < stmt_start.lineno()) {
        stmt_pieces.push_back(arena.hard_line());
      }

    } else {  // No comments to emit ahead of the statement.
      VLOG(5) << "no comments to emit ahead of statement: " << stmt->ToString();
      // If there's a line break between the last entity and this statement, we
      // retain it in the output (i.e. in paragraph style).
      if (last_entity_pos.lineno() + 1 < stmt_start.lineno()) {
        stmt_pieces.push_back(arena.hard_line());
      }
    }

    // Here we emit the formatted statement.
    bool last_stmt = i + 1 == n.statements().size();
    std::vector<DocRef> stmt_semi = {
        Fmt(*stmt, comments, arena, n.trailing_semi() || !last_stmt)};

    // Now we reflect the emission of the statement.
    last_entity_pos = stmt_limit;

    stmt_pieces.push_back(ConcatNGroup(arena, stmt_semi));
    statements.push_back(ConcatN(arena, stmt_pieces));

    last_entity_pos =
        CollectInlineComments(stmt_limit, last_entity_pos, comments, arena,
                              statements, last_comment_span);

    if (!last_stmt) {
      statements.push_back(arena.hard_line());
    }
  }

  // If the last statement was a verbatim, it already included a hard line,
  // so we don't need one right now.
  bool needs_hardline = !last_stmt_was_verbatim;

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc =
          EmitCommentsBetween(last_entity_pos, n.span().limit(), comments,
                              arena, &last_comment_span)) {
    VLOG(5) << "last entity position: " << last_entity_pos
            << " last_comment_span.start: " << last_comment_span->start();

    // If there's a line break between the last entity and this comment, we
    // retain it in the output (i.e. in paragraph style).
    if (last_entity_pos.lineno() + 1 < last_comment_span->start().lineno()) {
      statements.push_back(arena.hard_line());
    }

    if (!last_stmt_was_verbatim) {
      // Skip the hard line before the last comment if the last one was a
      // verbatim, because it already included one.
      statements.push_back(arena.hard_line());
    }

    statements.push_back(comments_doc.value());

    // We always need a hard line after the last comment.
    needs_hardline = true;
  }

  top.push_back(arena.MakeNest(ConcatN(arena, statements)));
  if (add_curls) {
    if (needs_hardline) {
      top.push_back(arena.hard_line());
    }
    top.push_back(arena.ccurl());
  } else {
    // If we're not putting hard lines in we want to at least check that we'll
    // force this all into break mode for multi-line emission.
    //
    // Note that the "inline block" case is handled specially above.
    top.push_back(arena.force_break_mode());
  }

  return ConcatNGroup(arena, top);
}

DocRef Fmt(const StatementBlock& n, Comments& comments, DocArena& arena) {
  return FmtBlock(n, comments, arena, /*add_curls=*/true);
}

DocRef Fmt(const Cast& n, Comments& comments, DocArena& arena) {
  DocRef lhs = Fmt(*n.expr(), comments, arena);

  Precedence arg_precedence = n.expr()->GetPrecedence();
  if (WeakerThan(arg_precedence, Precedence::kAs)) {
    lhs = ConcatN(arena, {arena.oparen(), lhs, arena.cparen()});
  }

  return ConcatNGroup(
      arena, {lhs, arena.space(), arena.Make(Keyword::kAs), arena.break1(),
              Fmt(*n.type_annotation(), comments, arena)});
}

DocRef Fmt(const ChannelDecl& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces{
      arena.Make(Keyword::kChan),
      arena.oangle(),
      Fmt(*n.type(), comments, arena),
  };
  if (n.fifo_depth().has_value()) {
    pieces.push_back(arena.comma());
    pieces.push_back(arena.space());
    pieces.push_back(Fmt(*n.fifo_depth().value(), comments, arena));
  }
  pieces.push_back(arena.cangle());
  if (n.dims().has_value()) {
    for (const Expr* dim : *n.dims()) {
      pieces.push_back(arena.obracket());
      pieces.push_back(Fmt(*dim, comments, arena));
      pieces.push_back(arena.cbracket());
    }
  }
  pieces.push_back(arena.oparen());
  pieces.push_back(Fmt(n.channel_name_expr(), comments, arena));
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const ColonRef& n, Comments& comments, DocArena& arena) {
  DocRef subject = absl::visit(
      Visitor{[&](const auto* n) { return Fmt(*n, comments, arena); }},
      n.subject());

  return ConcatNGroup(arena, {subject, arena.colon_colon(),
                              arena.MakeText(StripAnyDotModifier(n.attr()))});
}

DocRef FmtForLoopBaseLeader(Keyword keyword, DocRef names_ref,
                            const ForLoopBase& n, Comments& comments,
                            DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.Make(keyword),
      arena.MakeNestIfFlatFits(
          /*on_nested_flat_ref=*/names_ref,
          /*on_other_ref=*/arena.MakeConcat(arena.space(), names_ref))};

  if (n.type_annotation() != nullptr) {
    pieces.push_back(arena.colon());
    pieces.push_back(arena.space());
    pieces.push_back(Fmt(*n.type_annotation(), comments, arena));
  }

  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(Keyword::kIn));

  DocRef iterable_ref = Fmt(*n.iterable(), comments, arena);
  pieces.push_back(arena.MakeNestIfFlatFits(
      /*on_nested_flat_ref=*/iterable_ref,
      /*on_other_ref=*/arena.MakeConcat(arena.space(), iterable_ref)));

  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());
  return ConcatN(arena, pieces);
}

DocRef FmtForLoopBase(Keyword keyword, const ForLoopBase& n, Comments& comments,
                      DocArena& arena) {
  CHECK(keyword == Keyword::kFor || keyword == Keyword::kUnrollFor)
      << static_cast<std::underlying_type_t<Keyword>>(keyword);
  DocRef names_ref = Fmt(*n.names(), comments, arena);
  DocRef leader = FmtForLoopBaseLeader(keyword, names_ref, n, comments, arena);

  std::vector<DocRef> body_pieces;
  body_pieces.push_back(arena.hard_line());
  body_pieces.push_back(FmtBlock(*n.body(), comments, arena,
                                 /*add_curls=*/false,
                                 /*force_multiline=*/true));
  body_pieces.push_back(arena.hard_line());
  body_pieces.push_back(arena.ccurl());
  body_pieces.push_back(ConcatNGroup(
      arena,
      {arena.oparen(), Fmt(*n.init(), comments, arena), arena.cparen()}));

  return arena.MakeConcat(leader, ConcatN(arena, body_pieces));
}

DocRef Fmt(const UnrollFor& n, Comments& comments, DocArena& arena) {
  return FmtForLoopBase(Keyword::kUnrollFor, n, comments, arena);
}

DocRef Fmt(const For& n, Comments& comments, DocArena& arena) {
  return FmtForLoopBase(Keyword::kFor, n, comments, arena);
}

DocRef Fmt(const FormatMacro& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces = {
      arena.MakeText(n.macro()),
      arena.oparen(),
      arena.MakeText(
          absl::StrCat("\"", StepsToXlsFormatString(n.format()), "\"")),
  };

  if (!n.args().empty()) {
    pieces.push_back(arena.comma());
    pieces.push_back(arena.break1());
  }

  pieces.push_back(FmtJoin<const Expr*>(n.args(), Joiner::kCommaSpace,
                                        FmtExprPtr, comments, arena));
  pieces.push_back(arena.cparen());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Slice& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;

  if (n.start() != nullptr) {
    pieces.push_back(Fmt(*n.start(), comments, arena));
  }
  pieces.push_back(arena.break0());
  pieces.push_back(arena.colon());
  if (n.limit() != nullptr) {
    pieces.push_back(Fmt(*n.limit(), comments, arena));
  }
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const WidthSlice& n, Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 Fmt(*n.start(), comments, arena),
                                 arena.break0(),
                                 arena.plus_colon(),
                                 arena.break0(),
                                 Fmt(*n.width(), comments, arena),
                             });
}

DocRef Fmt(const IndexRhs& n, Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return Fmt(*n, comments, arena); },
          [&](const Slice* n) { return Fmt(*n, comments, arena); },
          [&](const WidthSlice* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef Fmt(const Index& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  if (WeakerThan(n.lhs()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena.oparen());
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
    pieces.push_back(arena.cparen());
  } else {
    pieces.push_back(Fmt(*n.lhs(), comments, arena));
  }
  pieces.push_back(arena.obracket());
  pieces.push_back(arena.MakeAlign(Fmt(n.rhs(), comments, arena)));
  pieces.push_back(arena.cbracket());
  return ConcatNGroup(arena, pieces);
}

DocRef FmtExprOrType(const ExprOrType& n, Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return Fmt(*n, comments, arena); },
          [&](const TypeAnnotation* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

std::optional<DocRef> FmtExplicitParametrics(const Instantiation& n,
                                             Comments& comments,
                                             DocArena& arena) {
  if (n.explicit_parametrics().empty()) {
    return std::nullopt;
  }
  return ConcatNGroup(
      arena, {arena.oangle(), arena.break0(),
              FmtJoin<ExprOrType>(absl::MakeConstSpan(n.explicit_parametrics()),
                                  Joiner::kCommaSpace, FmtParametricArg,
                                  comments, arena),
              arena.cangle()});
}

DocRef Fmt(const FunctionRef& n, Comments& comments, DocArena& arena) {
  DocRef callee_doc = Fmt(*n.callee(), comments, arena);
  std::optional<DocRef> parametrics_doc =
      FmtExplicitParametrics(n, comments, arena);
  return parametrics_doc.has_value()
             ? ConcatN(arena, {callee_doc, *parametrics_doc})
             : callee_doc;
}

DocRef Fmt(const Invocation& n, Comments& comments, DocArena& arena) {
  DocRef callee_doc = Fmt(*n.callee(), comments, arena);
  std::optional<DocRef> parametrics_doc =
      FmtExplicitParametrics(n, comments, arena);

  DocRef args_doc_internal =
      FmtJoin<const Expr*>(n.args(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
                           FmtExprPtr, comments, arena);

  // Group for the args tokens.
  std::vector<DocRef> arg_pieces = {
      arena.MakeNestIfFlatFits(
          /*on_nested_flat_ref=*/args_doc_internal,
          /*on_other_ref=*/arena.MakeAlign(args_doc_internal)),
      arena.cparen()};
  DocRef args_doc = ConcatNGroup(arena, arg_pieces);
  DocRef args_doc_nested = arena.MakeNest(args_doc);

  // This is the flat version -- it simply concats the pieces together.
  DocRef flat = parametrics_doc.has_value()
                    ? ConcatN(arena, {callee_doc, parametrics_doc.value(),
                                      arena.oparen(), args_doc})
                    : ConcatN(arena, {callee_doc, arena.oparen(), args_doc});

  // This doc ref is for the "I can emit *the leader* flat" case; i.e. the
  // callee (or the callee with parametric args).
  //
  // The parametrics have a break at the start (after the oangle) that can be
  // triggered, and the arguments have a break at the start (after the oparen)
  // that can be triggered.
  DocRef leader_flat =
      parametrics_doc.has_value()
          ? ConcatN(arena, {callee_doc, arena.MakeNest(parametrics_doc.value()),
                            arena.oparen(), arena.break0(), args_doc_nested})
          : ConcatN(arena, {callee_doc, arena.oparen(), arena.break0(),
                            args_doc_nested});

  DocRef result = arena.MakeGroup(
      arena.MakeFlatChoice(/*on_flat=*/flat, /*on_break=*/leader_flat));

  return result;
}

DocRef FmtNameDefTreePtr(const NameDefTree* n, Comments& comments,
                         DocArena& arena) {
  return Fmt(*n, comments, arena);
}

DocRef FmtMatchArm(const MatchArm& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(
      FmtJoin<const NameDefTree*>(n.patterns(), Joiner::kSpaceBarBreak,
                                  FmtNameDefTreePtr, comments, arena));
  pieces.push_back(arena.space());
  pieces.push_back(arena.fat_arrow());

  const Pos& rhs_start = n.expr()->span().start();

  DocRef rhs_doc = arena.MakeGroup(Fmt(*n.expr(), comments, arena));

  // Check for a comment between the arrow position and the RHS expression. This
  // can be needed when the RHS is not a block but an expression decorated with
  // a comment as if it were a block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc = EmitCommentsBetween(
          n.span().start(), rhs_start, comments, arena, &last_comment_span)) {
    pieces.push_back(arena.space());
    pieces.push_back(arena.space());
    pieces.push_back(comments_doc.value());
    pieces.push_back(arena.hard_line());
    pieces.push_back(arena.MakeNest(rhs_doc));
  } else {
    // If the RHS is a blocked expression, e.g. a struct instance, we don't
    // align it to the fat arrow indicated column.
    if (n.expr()->IsBlockedExprAnyLeader()) {
      pieces.push_back(arena.space());
      pieces.push_back(arena.MakeGroup(rhs_doc));
    } else {
      DocRef flat_choice_group = arena.MakeGroup(arena.MakeFlatChoice(
          /*on_flat=*/arena.MakeConcat(arena.space(), rhs_doc),
          /*on_break=*/arena.MakeConcat(arena.hard_line(),
                                        arena.MakeNest(rhs_doc))));
      pieces.push_back(flat_choice_group);
    }
  }

  return ConcatN(arena, pieces);
}

DocRef Fmt(const Match& n, Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  pieces.push_back(ConcatNGroup(
      arena,
      {arena.Make(Keyword::kMatch), arena.space(),
       Fmt(n.matched(), comments, arena), arena.space(), arena.ocurl()}));

  pieces.push_back(arena.hard_line());

  std::vector<DocRef> nested;

  Pos last_member_pos = n.matched()->span().limit();

  for (size_t i = 0; i < n.arms().size(); ++i) {
    const MatchArm* arm = n.arms()[i];

    // Note: the match arm member starts at the first pattern match.
    const Pos& member_start = arm->span().start();

    const Pos& member_limit = arm->span().limit();

    // See if there are comments above the match arm.
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_member_pos, member_start, comments, arena,
                                &last_comment_span)) {
      nested.push_back(comments_doc.value());
      nested.push_back(arena.hard_line());

      // If the comment abuts the member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        nested.push_back(arena.hard_line());
      }
    }

    nested.push_back(FmtMatchArm(*arm, comments, arena));
    nested.push_back(arena.comma());

    last_member_pos = arm->span().limit();

    // See if there are inline comments after the arm.
    last_member_pos =
        CollectInlineComments(member_limit, last_member_pos, comments, arena,
                              nested, last_comment_span);

    if (i + 1 != n.arms().size()) {
      nested.push_back(arena.hard_line());
    }
  }

  pieces.push_back(arena.MakeNest(ConcatN(arena, nested)));
  pieces.push_back(arena.hard_line());
  pieces.push_back(arena.ccurl());
  return ConcatN(arena, pieces);
}

DocRef Fmt(const Spawn& n, Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {arena.Make(Keyword::kSpawn), arena.space(),
                              Fmt(*n.config(), comments, arena)}

  );
}

DocRef FmtTupleWithoutComments(const XlsTuple& n, Comments& comments,
                               DocArena& arena) {
  // 1-element tuples are a special case- we always want a trailing comma and
  // never want it to be broken up. Handle separately here.
  if (n.members().size() == 1) {
    return ConcatNGroup(arena, {
                                   arena.oparen(),
                                   Fmt(*n.members()[0], comments, arena),
                                   arena.comma(),
                                   arena.cparen(),
                               });
  }

  DocRef guts = FmtJoin<const Expr*>(
      n.members(), Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak, FmtExprPtr,
      comments, arena);

  return ConcatNGroup(
      arena, {
                 arena.oparen(),
                 arena.MakeFlatChoice(
                     /*on_flat=*/guts,
                     /*on_break=*/ConcatNGroup(arena,
                                               {
                                                   arena.hard_line(),
                                                   arena.MakeNest(guts),
                                                   arena.hard_line(),
                                               })),
                 arena.cparen(),
             });
}

DocRef FmtTuple(const XlsTuple& n, Comments& comments, DocArena& arena) {
  Span tuple_span = n.span();
  // Detect if there are any comments in the span of the tuple.
  bool any_comments = comments.HasComments(tuple_span);

  if (!any_comments) {
    // Do it the old way.
    return FmtTupleWithoutComments(n, comments, arena);
  }

  // The general algorithm is:
  //  1. Before each element, prepend comments between the end of the previous
  //  element and the start of this one.
  //  2. At the end of the tuple, append comments between the last element and
  //  the end of the tuple.

  std::vector<DocRef> pieces = {};
  absl::Span<Expr* const> items = n.members();
  Pos last_tuple_element_span_limit = tuple_span.start();
  for (size_t i = 0; i < items.size(); ++i) {
    const Expr* item = items[i];
    const bool first_element = i == 0;
    const Span& span = item->span();

    // If there are comments between end of the last element we processed,
    // and the start of this one, prepend them.
    if (std::optional<DocRef> previous_comments =
            EmitCommentsBetween(last_tuple_element_span_limit, span.start(),
                                comments, arena, nullptr)) {
      if (!first_element) {
        pieces.push_back(arena.comma());
        pieces.push_back(arena.space());
      }
      // TODO: davidplass - if the previous comment is not on the same line as
      // the previous element, insert a hard line before the comment.
      pieces.push_back(previous_comments.value());
      pieces.push_back(arena.hard_line());
    } else if (!first_element) {
      // No comments between there and here; append a newline to "terminate" the
      // previous element.
      pieces.push_back(arena.comma());
      pieces.push_back(arena.hard_line());
    }

    last_tuple_element_span_limit = span.limit();
    // Format the element itself.
    pieces.push_back(FmtExprPtr(item, comments, arena));
  }

  DocRef guts = ConcatN(arena, pieces);

  // Append comments between the last element and the end of the tuple
  if (std::optional<DocRef> terminal_comment =
          EmitCommentsBetween(last_tuple_element_span_limit, tuple_span.limit(),
                              comments, arena, nullptr)) {
    // Add trailing comma before the terminal comment too.
    guts = ConcatN(
        arena, {guts, arena.comma(), arena.space(), terminal_comment.value()});
  } else if (n.members().size() == 1) {
    // No trailing comment, but add a comma if it's a singleton.
    guts = ConcatN(arena, {guts, arena.comma()});
  }

  return ConcatN(arena, {
                            arena.oparen(),
                            arena.hard_line(),
                            arena.MakeNest(guts),
                            arena.hard_line(),
                            arena.cparen(),
                        });
}

DocRef Fmt(const XlsTuple& n, Comments& comments, DocArena& arena) {
  return FmtTuple(n, comments, arena);
}

// Note: this does not put any spacing characters after the '{' so we can
// appropriately handle the case of an empty struct having no spacing in its
// `S {}` style construct.
DocRef FmtStructLeader(const TypeAnnotation* struct_ref, Comments& comments,
                       DocArena& arena) {
  return ConcatNGroup(arena, {
                                 Fmt(*struct_ref, comments, arena),
                                 arena.break1(),
                                 arena.ocurl(),
                             });
}

DocRef FmtStructMembersFlat(
    absl::Span<const std::pair<std::string, Expr*>> members, Comments& comments,
    DocArena& arena) {
  return FmtJoin<std::pair<std::string, Expr*>>(
      members, Joiner::kCommaSpace,
      [](const auto& member, Comments& comments, DocArena& arena) {
        const auto& [name, expr] = member;
        // If the expression is an identifier that matches its corresponding
        // struct member name, we canonically use the shorthand notation of just
        // providing the identifier and leaving the member name implicitly as
        // the same symbol.
        if (const NameRef* name_ref = dynamic_cast<const NameRef*>(expr);
            name_ref != nullptr && name_ref->identifier() == name) {
          return arena.MakeText(name);
        }

        return ConcatN(arena, {arena.MakeText(name), arena.colon(),
                               arena.space(), Fmt(*expr, comments, arena)});
      },
      comments, arena);
}

DocRef FmtStructMembersBreak(
    Span struct_span, absl::Span<const std::pair<std::string, Expr*>> members,
    Comments& comments, DocArena& arena) {
  std::vector<DocRef> pieces;
  Pos previous_item_limit = struct_span.start();
  for (size_t i = 0; i < members.size(); ++i) {
    const std::pair<std::string, Expr*>& member = members[i];
    const auto& [field_name, expr] = member;

    // If there are comments between the last item and here, insert them.
    Span comment_span(previous_item_limit, expr->span().start());
    if (std::optional<DocRef> previous_comments =
            EmitCommentsBetween(comment_span.start(), comment_span.limit(),
                                comments, arena, nullptr)) {
      pieces.push_back(previous_comments.value());
      pieces.push_back(arena.hard_line());
    }
    previous_item_limit = expr->span().limit();

    // If the expression is an identifier that matches its corresponding
    // struct member name, we canonically use the shorthand notation of just
    // providing the identifier and leaving the member name implicitly as
    // the same symbol.
    DocRef member_doc;
    if (const NameRef* name_ref = dynamic_cast<const NameRef*>(expr);
        name_ref != nullptr && name_ref->identifier() == field_name) {
      member_doc = arena.MakeText(field_name);
    } else {
      // First we format the member into a doc and then decide the best way to
      // put it into the sequence.
      DocRef field_expr = Fmt(*expr, comments, arena);

      // This is the document we want to emit both when we:
      // - Know it fits in flat mode
      // - Know the start of the document (i.e. the leader on the RHS
      //   expression) can be emitted in flat mode
      //
      // That's why it has a `break1` in it (instead of a space) and a
      // reassessment of whether to enter break mode for the field
      // expression.
      DocRef on_flat =
          ConcatN(arena, {arena.MakeText(field_name), arena.colon(),
                          arena.break1(), arena.MakeGroup(field_expr)});
      DocRef nest_field_expr =
          ConcatN(arena, {arena.MakeText(field_name), arena.colon(),
                          arena.hard_line(), arena.MakeNest(field_expr)});

      DocRef on_other;
      if (expr->IsBlockedExprWithLeader()) {
        DocRef leader = ConcatN(
            arena, {arena.MakeText(field_name), arena.colon(), arena.space(),
                    FmtBlockedExprLeader(*expr, comments, arena)});
        on_other = arena.MakeModeSelect(leader, /*on_flat=*/on_flat,
                                        /*on_break=*/nest_field_expr);
      } else {
        on_other = arena.MakeFlatChoice(on_flat, nest_field_expr);
      }

      member_doc = arena.MakeNestIfFlatFits(on_flat, on_other);
    }

    pieces.push_back(member_doc);
    pieces.push_back(arena.comma());
    // TODO: https://github.com/google/xls/issues/1719 - if there is a comment
    // on the same line as this item, insert the comment first.
    if (i + 1 != members.size()) {
      // Not the last item, add a hardline
      pieces.push_back(arena.hard_line());
    }
  }

  // Insert comments between the last item and the end of the struct.
  Span comment_span(previous_item_limit, struct_span.limit());
  if (std::optional<DocRef> previous_comments =
          EmitCommentsBetween(comment_span.start(), comment_span.limit(),
                              comments, arena, nullptr)) {
    pieces.push_back(arena.hard_line());
    pieces.push_back(previous_comments.value());
  }
  return ConcatNGroup(arena, pieces);
}

DocRef FmtFlatRest(const StructInstance& n, Comments& comments,
                   DocArena& arena) {
  return ConcatN(
      arena, {arena.space(),
              FmtStructMembersFlat(n.GetUnorderedMembers(), comments, arena),
              arena.space(), arena.ccurl()});
}

DocRef FmtBreakRest(const StructInstance& n, Comments& comments,
                    DocArena& arena) {
  return ConcatN(arena,
                 {arena.hard_line(),
                  arena.MakeNest(FmtStructMembersBreak(
                      n.span(), n.GetUnorderedMembers(), comments, arena)),
                  arena.hard_line(), arena.ccurl()});
}

DocRef Fmt(const StructInstance& n, Comments& comments, DocArena& arena) {
  DocRef leader = FmtStructLeader(n.struct_ref(), comments, arena);

  if (n.GetUnorderedMembers().empty()) {  // empty struct instance
    return arena.MakeConcat(leader, arena.ccurl());
  }

  // Implementation note: we cannot reorder members to be canonically the same
  // order as the struct definition in the general case, since the struct
  // definition may be defined an an imported file, and we have auto-formatting
  // work purely at the single-file syntax level.

  // If there are comments within the span, we must go to break mode, because
  // newlines.
  DocRef on_break = FmtBreakRest(n, comments, arena);
  if (comments.HasComments(n.span())) {
    return arena.MakeConcat(leader, on_break);
  }

  DocRef on_flat = FmtFlatRest(n, comments, arena);
  return arena.MakeConcat(
      leader, arena.MakeGroup(arena.MakeFlatChoice(on_flat, on_break)));
}

DocRef Fmt(const SplatStructInstance& n, Comments& comments, DocArena& arena) {
  DocRef leader = FmtStructLeader(n.struct_ref(), comments, arena);
  DocRef splatted = Fmt(n.splatted(), comments, arena);
  if (n.members().empty()) {
    return ConcatNGroup(arena, {leader, arena.break1(), arena.dot_dot(),
                                splatted, arena.break1(), arena.ccurl()});
  }

  DocRef on_flat = ConcatN(
      arena, {arena.space(), FmtStructMembersFlat(n.members(), comments, arena),
              arena.comma(), arena.space(), arena.dot_dot(), splatted,
              arena.space(), arena.ccurl()});
  DocRef on_break = ConcatN(
      arena, {arena.hard_line(),
              arena.MakeNest(ConcatN(
                  arena, {FmtStructMembersBreak(n.span(), n.members(), comments,
                                                arena),
                          arena.hard_line(), arena.dot_dot(), splatted})),
              arena.hard_line(), arena.ccurl()});
  return arena.MakeConcat(
      leader, arena.MakeGroup(arena.MakeFlatChoice(on_flat, on_break)));
}

DocRef Fmt(const String& n, Comments& comments, DocArena& arena) {
  return arena.MakeText(n.ToString());
}

// Creates a doc that has the "test portion" of the conditional; i.e.
//
//  if <break1> $test_expr <break1> {
DocRef MakeConditionalTest(const Conditional& n, Comments& comments,
                           DocArena& arena) {
  return ConcatN(
      arena, {
                 arena.Make(Keyword::kIf),
                 arena.space(),
                 FmtExpr(*n.test(), comments, arena, /*suppress_parens=*/true),
                 arena.space(),
                 arena.ocurl(),
             });
}

// When there's an else-if, or multiple statements inside of the blocks, we
// force the formatting to be multi-line.
DocRef FmtConditionalMultiline(const Conditional& n, Comments& comments,
                               DocArena& arena) {
  std::vector<DocRef> pieces = {
      MakeConditionalTest(n, comments, arena), arena.hard_line(),
      FmtBlock(*n.consequent(), comments, arena, /*add_curls=*/false),
      arena.hard_line()};

  std::variant<StatementBlock*, Conditional*> alternate = n.alternate();
  while (std::holds_alternative<Conditional*>(alternate)) {
    Conditional* elseif = std::get<Conditional*>(alternate);
    alternate = elseif->alternate();
    pieces.push_back(arena.ccurl());
    pieces.push_back(arena.space());
    pieces.push_back(arena.Make(Keyword::kElse));
    pieces.push_back(arena.space());
    pieces.push_back(MakeConditionalTest(*elseif, comments, arena));
    pieces.push_back(arena.hard_line());
    pieces.push_back(
        FmtBlock(*elseif->consequent(), comments, arena, /*add_curls=*/false));
    pieces.push_back(arena.hard_line());
  }

  CHECK(std::holds_alternative<StatementBlock*>(alternate));

  StatementBlock* else_block = std::get<StatementBlock*>(alternate);
  pieces.push_back(arena.ccurl());
  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(Keyword::kElse));
  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());
  pieces.push_back(arena.hard_line());
  pieces.push_back(FmtBlock(*else_block, comments, arena, /*add_curls=*/false));
  pieces.push_back(arena.hard_line());
  pieces.push_back(arena.ccurl());

  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const Conditional& n, Comments& comments, DocArena& arena) {
  // If there's an else-if clause or multi-statement blocks we force it to be
  // multi-line.
  if (n.HasElseIf() || n.HasMultiStatementBlocks()) {
    return FmtConditionalMultiline(n, comments, arena);
  }

  DocRef test = MakeConditionalTest(n, comments, arena);
  std::vector<DocRef> pieces = {
      test,
      arena.break1(),
      FmtBlock(*n.consequent(), comments, arena, /*add_curls=*/false),
      arena.break1(),
  };

  CHECK(std::holds_alternative<StatementBlock*>(n.alternate()));
  const StatementBlock* else_block = std::get<StatementBlock*>(n.alternate());
  pieces.push_back(arena.ccurl());
  pieces.push_back(arena.space());
  pieces.push_back(arena.Make(Keyword::kElse));
  pieces.push_back(arena.space());
  pieces.push_back(arena.ocurl());
  pieces.push_back(arena.break1());
  pieces.push_back(FmtBlock(*else_block, comments, arena, /*add_curls=*/false));
  pieces.push_back(arena.break1());
  pieces.push_back(arena.ccurl());
  return ConcatNGroup(arena, pieces);
}

DocRef Fmt(const ConstAssert& n, Comments& comments, DocArena& arena) {
  CHECK(false) << "ConstAssert should be handled by Formatter::Format.";
  return arena.empty();
}

DocRef Fmt(const TupleIndex& n, Comments& comments, DocArena& arena) {
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

DocRef Fmt(const ZeroMacro& n, Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 arena.MakeText("zero!"),
                                 arena.oangle(),
                                 FmtExprOrType(n.type(), comments, arena),
                                 arena.cangle(),
                                 arena.oparen(),
                                 arena.cparen(),
                             });
}

DocRef Fmt(const AllOnesMacro& n, Comments& comments, DocArena& arena) {
  return ConcatNGroup(arena, {
                                 arena.MakeText("all_ones!"),
                                 arena.oangle(),
                                 FmtExprOrType(n.type(), comments, arena),
                                 arena.cangle(),
                                 arena.oparen(),
                                 arena.cparen(),
                             });
}

DocRef Fmt(const Unop& n, Comments& comments, DocArena& arena) {
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

// Default formatting for let-as-expression is to not emit the RHS with a
// semicolon on it.
DocRef Fmt(const Let& n, Comments& comments, DocArena& arena) {
  return Formatter(comments, arena).Format(n, /*trailing_semi=*/false);
}

DocRef Fmt(const Range& n, Comments& comments, DocArena& arena) {
  return ConcatNGroup(
      arena, {Fmt(*n.start(), comments, arena), arena.break0(), arena.dot_dot(),
              arena.break0(), Fmt(*n.end(), comments, arena)});
}

DocRef Fmt(const NameDefTree::Leaf& n, Comments& comments, DocArena& arena) {
  return absl::visit(
      Visitor{
          [&](const NameDef* n) { return Fmt(*n, comments, arena); },
          [&](const NameRef* n) { return Fmt(*n, comments, arena); },
          [&](const WildcardPattern* n) { return Fmt(*n, comments, arena); },
          [&](const RestOfTuple* n) { return Fmt(*n, comments, arena); },
          [&](const Number* n) { return Fmt(*n, comments, arena); },
          [&](const ColonRef* n) { return Fmt(*n, comments, arena); },
          [&](const Range* n) { return Fmt(*n, comments, arena); },
      },
      n);
}

DocRef Fmt(const NameDefTree& n, Comments& comments, DocArena& arena) {
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

class FmtExprVisitor : public ExprVisitor {
 public:
  FmtExprVisitor(DocArena& arena, Comments& comments)
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
  Comments& comments_;
  std::optional<DocRef> result_;
};

// Note: suppress_parens just suppresses parentheses for the outermost
// expression "n", not transitively.
DocRef FmtExpr(const Expr& n, Comments& comments, DocArena& arena,
               bool suppress_parens) {
  FmtExprVisitor v(arena, comments);
  CHECK_OK(n.AcceptExpr(&v));
  DocRef result = v.result();
  if (n.in_parens() && !suppress_parens) {
    return ConcatNGroup(arena, {arena.oparen(), result, arena.cparen()});
  }
  return result;
}

// Creates a document for the "leader" of the given expression.
//
// Precondition: `e` must be a blocked expression with a leader component; e.g.
// invocation (leader is callee), conditional (leader is test), etc.
DocRef FmtBlockedExprLeader(const Expr& e, Comments& comments,
                            DocArena& arena) {
  CHECK(e.IsBlockedExprWithLeader());
  switch (e.kind()) {
    case AstNodeKind::kInvocation: {
      return arena.MakeConcat(
          Fmt(static_cast<const Invocation&>(e).callee(), comments, arena),
          arena.oparen());
    }
    case AstNodeKind::kConditional: {
      const Expr& test = *static_cast<const Conditional&>(e).test();
      return ConcatN(
          arena, {arena.Make(Keyword::kIf), arena.space(),
                  Fmt(test, comments, arena), arena.space(), arena.ocurl()});
    }
    case AstNodeKind::kMatch: {
      const Expr& test = *static_cast<const Match&>(e).matched();
      return ConcatN(
          arena, {arena.Make(Keyword::kMatch), arena.space(),
                  Fmt(test, comments, arena), arena.space(), arena.ocurl()});
    }
    case AstNodeKind::kArray: {
      const TypeAnnotation& type =
          *static_cast<const Array&>(e).type_annotation();
      return ConcatN(
          arena, {Fmt(type, comments, arena), arena.colon(), arena.obracket()});
    }
    case AstNodeKind::kStructInstance: {
      const StructInstance& n = static_cast<const StructInstance&>(e);
      return ConcatN(arena, {FmtStructLeader(n.struct_ref(), comments, arena),
                             arena.space(), arena.ocurl()});
    }
    case AstNodeKind::kSplatStructInstance: {
      const SplatStructInstance& n = static_cast<const SplatStructInstance&>(e);
      return ConcatN(arena, {FmtStructLeader(n.struct_ref(), comments, arena),
                             arena.space(), arena.ocurl()});
    }
    case AstNodeKind::kFor:
    case AstNodeKind::kUnrollFor: {
      const ForLoopBase& n = static_cast<const ForLoopBase&>(e);
      Keyword keyword =
          e.kind() == AstNodeKind::kFor ? Keyword::kFor : Keyword::kUnrollFor;
      DocRef names_ref = Fmt(*n.names(), comments, arena);
      return FmtForLoopBaseLeader(keyword, names_ref, n, comments, arena);
    }
    default:
      LOG(FATAL) << "Unhandled node kind for FmtBlockedExprLeader: `"
                 << e.ToString() << "` @ "
                 << e.span().ToString(arena.file_table());
  }
}

DocRef Fmt(const Expr& n, Comments& comments, DocArena& arena) {
  return FmtExpr(n, comments, arena, /*suppress_parens=*/false);
}

}  // namespace

DocRef Formatter::Format(const Expr& n) {
  return FmtExpr(n, comments_, arena_, /*suppress_parens=*/false);
}

DocRef Formatter::Format(const ConstantDef& n) {
  // There are 7 places a comment can be in a constant definition:
  // [pub 1] const 2 name 3 [: 4 type 5] = 6 value 7;

  // leader_pieces contains docrefs from `pub` up to and including the *first
  // comments* in categories 3-6.
  // Note: the comments in category 1 and 2 are not included in the leader.
  std::vector<DocRef> leader_pieces;
  if (n.is_public()) {
    leader_pieces.push_back(arena_.Make(Keyword::kPub));
    leader_pieces.push_back(arena_.break1());
  }
  leader_pieces.push_back(arena_.Make(Keyword::kConst));
  leader_pieces.push_back(arena_.break1());
  leader_pieces.push_back(arena_.MakeText(n.identifier()));

  // The right hand side.
  std::vector<DocRef> rhs_pieces;
  // Destination for where we should put items from this point forward.
  // Initially it's the leader but after the first comment it switches to the
  // rhs, which will have to be nested.
  std::vector<DocRef>* dest = &leader_pieces;
  bool nest_rhs = false;

  if (n.type_annotation() != nullptr) {
    // Comments between the name and the type (category 3 and 4). We don't know
    // there the colon is, so we put it just before the type.
    std::optional<DocRef> comments_doc = EmitCommentsBetween(
        n.name_def()->GetSpan()->limit(),
        n.type_annotation()->GetSpan()->start(), comments_, arena_,
        /*last_comment_span=*/nullptr);
    if (comments_doc.has_value()) {
      dest->push_back(ConcatN(
          arena_, {arena_.break1(), *comments_doc, arena_.hard_line()}));
      // From now on we need to nest.
      nest_rhs = true;
      dest = &rhs_pieces;
    }

    dest->push_back(arena_.colon());
    dest->push_back(arena_.break1());
    dest->push_back(Fmt(*n.type_annotation(), comments_, arena_));

    // Find comments between the end of the type annotation and the start of
    // the value and put them between the type and the =
    comments_doc =
        EmitCommentsBetween(n.type_annotation()->GetSpan()->limit(),
                            n.value()->GetSpan()->start(), comments_, arena_,
                            /*last_comment_span=*/nullptr);
    if (comments_doc.has_value()) {
      dest->push_back(ConcatN(
          arena_, {arena_.break1(), *comments_doc, arena_.hard_line()}));
      // From now on we need to nest.
      nest_rhs = true;
      dest = &rhs_pieces;
    } else if (nest_rhs) {
      // No comments between the type and the value; put a space before the =
      dest->push_back(arena_.break1());
    }
  } else {
    // Comments in category 3 & 6: between the name and the value. We don't know
    // where the = is, so we put all comments in this category before it.
    std::optional<DocRef> comments_doc =
        EmitCommentsBetween(n.name_def()->GetSpan()->limit(),
                            n.value()->GetSpan()->start(), comments_, arena_,
                            /*last_comment_span=*/nullptr);
    if (comments_doc.has_value()) {
      leader_pieces.push_back(ConcatN(
          arena_, {arena_.break1(), *comments_doc, arena_.hard_line()}));
      dest = &rhs_pieces;
      nest_rhs = true;
    }
  }

  if (nest_rhs) {
    // Make the = part of the RHS, and nest the whole RHS.
    rhs_pieces.push_back(arena_.equals());
    rhs_pieces.push_back(arena_.space());
  } else {
    leader_pieces.push_back(arena_.break1());
    leader_pieces.push_back(arena_.equals());
    leader_pieces.push_back(arena_.space());
  }

  DocRef lhs;
  if (nest_rhs) {
    // If we're nesting the rhs, the lhs can't be grouped, because we know
    // there's at least one hardline in it.
    lhs = ConcatN(arena_, leader_pieces);
  } else {
    lhs = ConcatNGroup(arena_, leader_pieces);
  }

  DocRef value_doc = Fmt(*n.value(), comments_, arena_);
  std::optional<DocRef> comments_doc = EmitCommentsBetween(
      n.value()->GetSpan()->limit(), n.span().limit(), comments_, arena_,
      /*last_comment_span=*/nullptr);
  if (comments_doc.has_value()) {
    // There are comments between the end of the value and the semicolon.

    // TODO: google/xls#1697 - check if the comment is not on the same line as
    // the value. If so, emit a hard_line before the comment.
    rhs_pieces.push_back(value_doc);
    rhs_pieces.push_back(arena_.space());
    rhs_pieces.push_back(*comments_doc);
    rhs_pieces.push_back(arena_.hard_line());
    if (nest_rhs) {
      // The whole RHS will be nested.
      rhs_pieces.push_back(arena_.semi());
    } else {
      rhs_pieces.push_back(arena_.MakeNest(arena_.semi()));
    }
  } else {
    // Reduce the width by 1 so we know we can emit the semi inline.
    rhs_pieces.push_back(arena_.MakeReduceTextWidth(value_doc, 1));
    rhs_pieces.push_back(arena_.semi());
  }
  DocRef rhs = ConcatN(arena_, rhs_pieces);
  if (nest_rhs) {
    rhs = arena_.MakeNest(rhs);
  }

  // Now get the comments between the *start* of the node and the start of the
  // name (category 1 & 2)
  DocRef pre_comment = arena_.empty();
  Span lhs_comments_span(n.span().start(), n.name_def()->GetSpan()->start());
  comments_doc = EmitCommentsBetween(
      lhs_comments_span.start(), lhs_comments_span.limit(), comments_, arena_,
      /*last_comment_span=*/nullptr);
  if (comments_doc.has_value()) {
    pre_comment = ConcatN(arena_, {*comments_doc, arena_.hard_line()});
  }

  return ConcatN(arena_, {pre_comment, lhs, rhs});
}

DocRef Formatter::Format(const ConstAssert& n) {
  return ConcatNGroup(arena_, {
                                  arena_.MakeText("const_assert!"),
                                  arena_.oparen(),
                                  Fmt(*n.arg(), comments_, arena_),
                                  arena_.cparen(),
                              });
}

DocRef Formatter::Format(const VerbatimNode& n) {
  if (n.text().empty()) {
    return arena_.empty();
  }
  return arena_.MakeZeroIndent(arena_.MakeText(std::string(n.text())));
}

DocRef Formatter::Format(const Statement& n, bool trailing_semi) {
  auto maybe_concat_semi = [&](DocRef d) {
    if (trailing_semi) {
      return arena_.MakeConcat(d, arena_.semi());
    }
    return d;
  };
  return absl::visit(
      Visitor{
          [&](const VerbatimNode* n) { return Format(*n); },
          [&](const Expr* n) {
            return maybe_concat_semi(Fmt(*n, comments_, arena_));
          },
          [&](const TypeAlias* n) { return maybe_concat_semi(Format(*n)); },
          [&](const Let* n) { return Format(*n, trailing_semi); },
          [&](const ConstAssert* n) { return maybe_concat_semi(Format(*n)); },
      },
      n.wrapped());
}

// Formats parameters (i.e. function parameters) with leading '(' and trailing
// ')'.
DocRef Formatter::FormatParams(absl::Span<const Param* const> params) {
  DocRef guts = FmtJoin<const Param*>(
      params, Joiner::kCommaBreak1AsGroupNoTrailingComma,
      [](const Param* param, Comments& comments, DocArena& arena) {
        DocRef id = arena.MakeText(param->identifier());
        if (auto* st =
                dynamic_cast<SelfTypeAnnotation*>(param->type_annotation());
            st != nullptr && !st->explicit_type()) {
          return id;
        }
        DocRef type = Fmt(*param->type_annotation(), comments, arena);
        return ConcatN(arena, {id, arena.colon(), arena.space(), type});
      },
      comments_, arena_);

  return ConcatNGroup(
      arena_, {arena_.oparen(),
               arena_.MakeAlign(arena_.MakeConcat(guts, arena_.cparen()))});
}

DocRef Formatter::Format(const ParametricBinding& n) {
  std::vector<DocRef> pieces = {
      arena_.MakeText(n.identifier()),
      arena_.colon(),
      arena_.break1(),
      Fmt(*n.type_annotation(), comments_, arena_),
  };
  if (n.expr() != nullptr) {
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.equals());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.ocurl());
    pieces.push_back(arena_.break0());
    pieces.push_back(arena_.MakeNest(Fmt(*n.expr(), comments_, arena_)));
    pieces.push_back(arena_.ccurl());
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::Format(const ParametricBinding* n) {
  CHECK(n != nullptr);
  return Format(*n);
}

DocRef Formatter::Format(const Function& n) {
  std::vector<DocRef> signature_pieces;
  if (n.is_public()) {
    signature_pieces.push_back(arena_.Make(Keyword::kPub));
    signature_pieces.push_back(arena_.space());
  }
  signature_pieces.push_back(arena_.Make(Keyword::kFn));
  signature_pieces.push_back(arena_.space());
  signature_pieces.push_back(arena_.MakeText(n.identifier()));

  if (n.IsParametric()) {
    DocRef flat_parametrics = ConcatNGroup(
        arena_, {arena_.oangle(),
                 FmtJoin<const ParametricBinding*>(
                     n.parametric_bindings(), Joiner::kCommaSpace,
                     [&](const ParametricBinding* n, Comments& comments,
                         DocArena& arena) { return Format(n); },
                     comments_, arena_),
                 arena_.cangle()});

    DocRef parametric_guts =
        ConcatN(arena_, {arena_.oangle(),
                         arena_.MakeAlign(FmtJoin<const ParametricBinding*>(
                             n.parametric_bindings(),
                             Joiner::kCommaBreak1AsGroupNoTrailingComma,
                             [&](const ParametricBinding* n, Comments& comments,
                                 DocArena& arena) { return Format(n); },
                             comments_, arena_)),
                         arena_.cangle()});
    DocRef break_parametrics = ConcatNGroup(
        arena_, {
                    arena_.break0(),
                    arena_.MakeFlatChoice(parametric_guts,
                                          arena_.MakeNest(parametric_guts)),
                });
    signature_pieces.push_back(arena_.MakeNestIfFlatFits(
        /*on_nested_flat_ref=*/flat_parametrics,
        /*on_other_ref=*/break_parametrics));
  }

  {
    std::vector<DocRef> params_pieces;

    params_pieces.push_back(arena_.break0());
    params_pieces.push_back(FormatParams(n.params()));

    if (n.return_type() == nullptr) {
      params_pieces.push_back(arena_.break1());
      params_pieces.push_back(arena_.ocurl());
    } else {
      params_pieces.push_back(
          ConcatNGroup(arena_, {
                                   arena_.break1(),
                                   arena_.arrow(),
                                   arena_.space(),
                                   Fmt(*n.return_type(), comments_, arena_),
                                   arena_.space(),
                                   arena_.ocurl(),
                               }));
    }

    signature_pieces.push_back(
        arena_.MakeNest(ConcatNGroup(arena_, params_pieces)));
  }

  std::vector<DocRef> fn_pieces;
  if (n.extern_verilog_module().has_value()) {
    auto code_template = (*n.extern_verilog_module()).code_template();
    fn_pieces.push_back(
        ConcatN(arena_, {
                            arena_.MakeText("#[extern_verilog(\""),
                            arena_.MakeText(code_template),
                            arena_.MakeText("\")]"),
                            arena_.hard_line(),
                        }));
  }

  if (n.body()->empty()) {
    // For empty function we don't put spaces between the curls.
    fn_pieces.push_back(ConcatNGroup(arena_, signature_pieces));
    fn_pieces.push_back(
        FmtBlock(*n.body(), comments_, arena_, /*add_curls=*/false));
    fn_pieces.push_back(arena_.ccurl());
  } else {
    // For non-empty functions, we break after the signature and before
    // the ccurl.
    fn_pieces.push_back(ConcatNGroup(arena_, signature_pieces));
    fn_pieces.push_back(arena_.break1());
    fn_pieces.push_back(
        FmtBlock(*n.body(), comments_, arena_, /*add_curls=*/false));
    fn_pieces.push_back(arena_.break1());
    fn_pieces.push_back(arena_.ccurl());
  };

  return ConcatNGroup(arena_, fn_pieces);
}

DocRef Formatter::Format(const ProcMember& n) {
  return ConcatNGroup(
      arena_, {Fmt(*n.name_def(), comments_, arena_), arena_.colon(),
               arena_.break1(), Fmt(*n.type_annotation(), comments_, arena_)});
}

DocRef Formatter::Format(const Proc& n) {
  std::vector<DocRef> signature_pieces;
  if (n.is_public()) {
    signature_pieces.push_back(arena_.Make(Keyword::kPub));
    signature_pieces.push_back(arena_.space());
  }
  signature_pieces.push_back(arena_.Make(Keyword::kProc));
  signature_pieces.push_back(arena_.space());
  signature_pieces.push_back(arena_.MakeText(n.identifier()));

  if (n.IsParametric()) {
    signature_pieces.push_back(ConcatNGroup(
        arena_, {arena_.oangle(),
                 FmtJoin<const ParametricBinding*>(
                     n.parametric_bindings(), Joiner::kCommaSpace,
                     [&](const ParametricBinding* n, Comments& comments,
                         DocArena& arena) { return Format(n); },
                     comments_, arena_),
                 arena_.cangle()}));
  }
  signature_pieces.push_back(arena_.break1());
  signature_pieces.push_back(arena_.ocurl());

  Pos last_stmt_limit = n.body_span().start();

  // We update this with the position that's relevant for config start
  // comments_.
  std::optional<Pos> config_comment_start_pos;
  std::optional<Pos> init_comment_start_pos;
  std::optional<Pos> next_comment_start_pos;

  std::vector<DocRef> stmt_pieces;
  for (const ProcStmt& stmt : n.stmts()) {
    absl::visit(
        Visitor{
            [&](const Function* f) {
              // Note: we will emit these below.
              //
              // Though we defer emission, we still want to grab data to tell
              // us the relevant comment span. The relevant comment span is
              // from the "last statement's limit position" to this function's
              // start position.
              if (f == &n.config()) {
                config_comment_start_pos = last_stmt_limit;
              } else if (f == &n.init()) {
                init_comment_start_pos = last_stmt_limit;
              } else if (f == &n.next()) {
                next_comment_start_pos = last_stmt_limit;
              } else {
                LOG(FATAL) << "Unexpected proc member function: "
                           << f->identifier() << " @ "
                           << f->span().ToString(arena_.file_table());
              }
              last_stmt_limit = f->span().limit();
            },
            [&](const ProcMember* n) {
              if (std::optional<DocRef> maybe_doc =
                      EmitCommentsBetween(last_stmt_limit, n->span().start(),
                                          comments_, arena_, nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(Format(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const ConstantDef* n) {
              if (std::optional<DocRef> maybe_doc =
                      EmitCommentsBetween(last_stmt_limit, n->span().start(),
                                          comments_, arena_, nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(Format(*n));
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const TypeAlias* n) {
              if (std::optional<DocRef> maybe_doc =
                      EmitCommentsBetween(last_stmt_limit, n->span().start(),
                                          comments_, arena_, nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(Format(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const ConstAssert* n) {
              if (std::optional<DocRef> maybe_doc =
                      EmitCommentsBetween(last_stmt_limit, n->span().start(),
                                          comments_, arena_, nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(Format(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
        },
        stmt);
  }

  CHECK(config_comment_start_pos.has_value());
  std::optional<DocRef> config_comment =
      EmitCommentsBetween(config_comment_start_pos, n.config().span().start(),
                          comments_, arena_, nullptr);
  CHECK(init_comment_start_pos.has_value());
  std::optional<DocRef> init_comment =
      EmitCommentsBetween(init_comment_start_pos, n.init().span().start(),
                          comments_, arena_, nullptr);
  CHECK(next_comment_start_pos.has_value());
  std::optional<DocRef> next_comment =
      EmitCommentsBetween(next_comment_start_pos, n.next().span().start(),
                          comments_, arena_, nullptr);
  // Comments between the last statement and the end of the proc.
  std::optional<DocRef> end_comment = EmitCommentsBetween(
      last_stmt_limit, n.body_span().limit(), comments_, arena_, nullptr);

  std::vector<DocRef> config_pieces = {
      arena_.MakeText("config"),
      FormatParams(n.config().params()),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FmtBlock(*n.config().body(), comments_, arena_, /*add_curls=*/false),
      arena_.break1(),
      arena_.ccurl(),
  };

  std::vector<DocRef> init_pieces = {
      arena_.MakeText("init"),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FmtBlock(*n.init().body(), comments_, arena_, /*add_curls=*/false),
      arena_.break1(),
      arena_.ccurl(),
  };

  std::vector<DocRef> next_pieces = {
      arena_.MakeText("next"),
      FormatParams(n.next().params()),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FmtBlock(*n.next().body(), comments_, arena_, /*add_curls=*/false),
      arena_.break1(),
      arena_.ccurl(),
  };

  auto nest_and_hardline = [&](std::optional<DocRef> doc_ref) {
    if (!doc_ref.has_value()) {
      return arena_.empty();
    }
    return arena_.MakeNest(arena_.MakeConcat(*doc_ref, arena_.hard_line()));
  };

  DocRef config_comment_doc_ref = nest_and_hardline(config_comment);
  DocRef init_comment_doc_ref = nest_and_hardline(init_comment);
  DocRef next_comment_doc_ref = nest_and_hardline(next_comment);
  // The last comment is special, because we don't want to nest the
  // following curl.
  DocRef end_comment_doc_ref =
      end_comment.has_value()
          ? arena_.MakeConcat(arena_.MakeNest(*end_comment), arena_.hard_line())
          : arena_.empty();

  std::vector<DocRef> proc_pieces = {
      ConcatNGroup(arena_, signature_pieces),
      arena_.hard_line(),
      stmt_pieces.empty()
          ? arena_.empty()
          : ConcatNGroup(arena_,
                         {
                             arena_.MakeNest(ConcatNGroup(arena_, stmt_pieces)),
                             arena_.hard_line(),
                         }),
      config_comment_doc_ref,
      arena_.MakeNest(ConcatNGroup(arena_, config_pieces)),
      arena_.hard_line(),
      arena_.hard_line(),
      init_comment_doc_ref,
      arena_.MakeNest(ConcatNGroup(arena_, init_pieces)),
      arena_.hard_line(),
      arena_.hard_line(),
      next_comment_doc_ref,
      arena_.MakeNest(ConcatNGroup(arena_, next_pieces)),
      arena_.hard_line(),
      end_comment_doc_ref,
      arena_.ccurl(),
  };

  return ConcatNGroup(arena_, proc_pieces);
}

DocRef Formatter::Format(const TestFunction& n) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena_.MakeText("#[test]"));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(Format(n.fn()));
  return ConcatN(arena_, pieces);
}

DocRef Formatter::Format(const TestProc& n) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena_.MakeText("#[test_proc]"));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(Format(*n.proc()));
  return ConcatN(arena_, pieces);
}

DocRef Formatter::Format(const QuickCheck& n) {
  std::vector<DocRef> pieces;
  switch (n.test_cases().tag()) {
    case QuickCheckTestCasesTag::kExhaustive:
      pieces.push_back(arena_.MakeText("#[quickcheck(exhaustive)]"));
      break;
    case QuickCheckTestCasesTag::kCounted:
      if (n.test_cases().count().has_value()) {
        pieces.push_back(arena_.MakeText(absl::StrFormat(
            "#[quickcheck(test_count=%d)]", *n.test_cases().count())));
      } else {
        pieces.push_back(arena_.MakeText("#[quickcheck]"));
      }
      break;
  }
  pieces.push_back(arena_.hard_line());
  pieces.push_back(Format(*n.fn()));
  return ConcatN(arena_, pieces);
}

static void FmtStructMembers(const StructDefBase& n, Comments& comments,
                             DocArena& arena, std::vector<DocRef>& pieces) {
  if (!n.members().empty()) {
    pieces.push_back(arena.break1());
  }

  std::vector<DocRef> body_pieces;
  Pos last_member_pos = n.span().start();
  for (size_t i = 0; i < n.members().size(); ++i) {
    const auto* member = n.members()[i];

    const Span member_span = member->span();
    const Pos& member_start = member_span.start();

    // See if there are comments between the last member and the start of this
    // member.
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_member_pos, member_start, comments, arena,
                                &last_comment_span)) {
      body_pieces.push_back(comments_doc.value());
      body_pieces.push_back(arena.hard_line());

      // If the comment abuts the member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        body_pieces.push_back(arena.hard_line());
      }
    }

    last_member_pos = member_span.limit();

    body_pieces.push_back(arena.MakeText(member->name()));
    body_pieces.push_back(arena.colon());
    body_pieces.push_back(arena.space());
    body_pieces.push_back(Fmt(*member->type(), comments, arena));
    bool last_member = i + 1 == n.members().size();
    if (last_member) {
      body_pieces.push_back(arena.MakeFlatChoice(/*on_flat=*/arena.empty(),
                                                 /*on_break=*/arena.comma()));
    } else {
      body_pieces.push_back(arena.comma());
    }

    // See if there are inline comments after the member.
    Pos new_last_member_pos =
        CollectInlineComments(member_span.limit(), last_member_pos, comments,
                              arena, body_pieces, last_comment_span);

    bool had_inline = new_last_member_pos != last_member_pos;
    if (!last_member) {
      body_pieces.push_back(had_inline ? arena.hard_line() : arena.break1());
    }
    last_member_pos = new_last_member_pos;
  }

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  bool emitted_trailing_comment = false;
  if (std::optional<DocRef> comments_doc =
          EmitCommentsBetween(last_member_pos, n.span().limit(), comments,
                              arena, &last_comment_span)) {
    body_pieces.push_back(arena.hard_line());
    body_pieces.push_back(comments_doc.value());
    emitted_trailing_comment = true;
  }

  pieces.push_back(arena.MakeNest(ConcatN(arena, body_pieces)));

  if (!n.members().empty() || emitted_trailing_comment) {
    pieces.push_back(arena.break1());
  }
}

DocRef Formatter::FormatStructDefBase(
    const StructDefBase& n, Keyword keyword,
    const std::optional<std::string>& extern_type_name) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (extern_type_name.has_value()) {
    attr = FmtSvTypeAttribute(*extern_type_name, arena_);
  }
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(keyword));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.MakeText(n.identifier()));

  if (!n.parametric_bindings().empty()) {
    pieces.push_back(arena_.oangle());
    pieces.push_back(FmtJoin<const ParametricBinding*>(
        n.parametric_bindings(), Joiner::kCommaSpace,
        [&](const ParametricBinding* n, Comments& comments, DocArena& arena) {
          return Format(n);
        },
        comments_, arena_));
    pieces.push_back(arena_.cangle());
  }

  pieces.push_back(arena_.space());
  pieces.push_back(arena_.ocurl());

  FmtStructMembers(n, comments_, arena_, pieces);

  pieces.push_back(arena_.ccurl());
  return JoinWithAttr(attr, ConcatNGroup(arena_, pieces), arena_);
}

DocRef Formatter::Format(const StructDef& n) {
  return FormatStructDefBase(n, Keyword::kStruct, n.extern_type_name());
}

DocRef Formatter::Format(const ProcDef& n) {
  return FormatStructDefBase(n, Keyword::kProc,
                             /*extern_type_name=*/std::nullopt);
}

DocRef Formatter::Format(const ImplMember& n) {
  return absl::visit(Visitor{
                         [&](const Function* n) { return Format(*n); },
                         [&](const ConstantDef* n) { return Format(*n); },
                         [&](const VerbatimNode* n) { return Format(*n); },
                     },
                     n);
}

DocRef Formatter::Format(const Impl& n) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(Keyword::kImpl));
  pieces.push_back(arena_.space());
  pieces.push_back(Fmt(*n.struct_ref(), comments_, arena_));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.ocurl());
  if (!n.members().empty()) {
    pieces.push_back(arena_.break1());
  }
  std::vector<DocRef> body_pieces;
  Pos last_member_pos = n.span().start();
  bool last_was_func = false;
  for (int i = 0; i < n.members().size(); i++) {
    const ImplMember member = n.members().at(i);
    if (i > 0 && (last_was_func || std::holds_alternative<Function*>(member))) {
      body_pieces.push_back(arena_.hard_line());
    }
    last_was_func = std::holds_alternative<Function*>(member);

    // See if there are comments_ between the last member and the start of this
    // member.
    Span member_span = ToAstNode(member)->GetSpan().value();
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_member_pos, member_span.start(), comments_,
                                arena_, &last_comment_span)) {
      body_pieces.push_back(comments_doc.value());
      body_pieces.push_back(arena_.hard_line());
    }
    body_pieces.push_back(Format(member));
    last_member_pos = member_span.limit();

    last_member_pos =
        CollectInlineComments(member_span.limit(), last_member_pos, comments_,
                              arena_, body_pieces, last_comment_span);

    body_pieces.push_back(arena_.hard_line());
  }
  if (!body_pieces.empty()) {
    // Remove last line break.
    body_pieces.pop_back();
    pieces.push_back(arena_.MakeNest(ConcatN(arena_, body_pieces)));
    pieces.push_back(arena_.hard_line());
  }
  pieces.push_back(arena_.ccurl());

  return JoinWithAttr(attr, ConcatNGroup(arena_, pieces), arena_);
}

DocRef Formatter::Format(const EnumMember& n) {
  return ConcatNGroup(
      arena_,
      {Fmt(*n.name_def, comments_, arena_), arena_.space(), arena_.equals(),
       arena_.break1(), Fmt(*n.value, comments_, arena_), arena_.comma()});
}

DocRef Formatter::Format(const EnumDef& n) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (n.extern_type_name()) {
    attr = FmtSvTypeAttribute(*n.extern_type_name(), arena_);
  }
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(Keyword::kEnum));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.MakeText(n.identifier()));

  pieces.push_back(arena_.space());
  if (n.type_annotation() != nullptr) {
    pieces.push_back(arena_.colon());
    pieces.push_back(arena_.space());
    pieces.push_back(Fmt(*n.type_annotation(), comments_, arena_));
    pieces.push_back(arena_.space());
  }

  pieces.push_back(arena_.ocurl());
  pieces.push_back(arena_.hard_line());

  std::vector<DocRef> nested;
  Pos last_member_pos = n.span().start();
  for (size_t i = 0; i < n.values().size(); ++i) {
    const EnumMember& node = n.values()[i];

    // If there are comment blocks between the last member position and the
    // member we're about the process, we need to emit them.
    std::optional<Span> member_span = node.GetSpan();
    CHECK(member_span.has_value());
    Pos member_start = member_span->start();

    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_member_pos, member_start, comments_,
                                arena_, &last_comment_span)) {
      nested.push_back(comments_doc.value());
      nested.push_back(arena_.hard_line());

      // If the comment abuts the member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        nested.push_back(arena_.hard_line());
      }
    }

    last_member_pos = member_span->limit();

    // Here we actually emit the formatted member.
    nested.push_back(Format(node));
    if (i + 1 != n.values().size()) {
      nested.push_back(arena_.hard_line());
    }
  }

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc =
          EmitCommentsBetween(last_member_pos, n.span().limit(), comments_,
                              arena_, &last_comment_span)) {
    nested.push_back(arena_.hard_line());
    nested.push_back(comments_doc.value());
  }

  DocRef nested_ref = ConcatN(arena_, nested);
  pieces.push_back(arena_.MakeNest(nested_ref));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(arena_.ccurl());
  return JoinWithAttr(attr, ConcatN(arena_, pieces), arena_);
}

DocRef Formatter::Format(const Import& n) {
  std::vector<DocRef> dotted_pieces;
  for (size_t i = 0; i < n.subject().size(); ++i) {
    const std::string& subject_part = n.subject()[i];
    DocRef this_doc_ref;
    if (i + 1 == n.subject().size()) {
      this_doc_ref = ConcatNGroup(arena_, {arena_.MakeText(subject_part)});
    } else {
      this_doc_ref = ConcatNGroup(arena_, {arena_.MakeText(subject_part),
                                           arena_.dot(), arena_.break0()});
    }
    dotted_pieces.push_back(this_doc_ref);
  }

  std::vector<DocRef> pieces = {
      arena_.Make(Keyword::kImport), arena_.space(),
      arena_.MakeAlign(ConcatNGroup(arena_, dotted_pieces))};

  if (const std::optional<std::string>& alias = n.alias()) {
    DocRef alias_text = arena_.MakeText(alias.value());
    DocRef as = arena_.Make(Keyword::kAs);

    // Flat version is " as alias"
    DocRef flat_as = ConcatN(arena_, {arena_.space(), as, arena_.space(),
                                      alias_text, arena_.semi()});
    // Break version puts the "as alias" at an indent on the next line.
    DocRef break_as = ConcatN(
        arena_,
        {arena_.hard_line(),
         arena_.MakeNest(ConcatN(
             arena_, {as, arena_.space(), alias_text, arena_.semi()}))});
    // Choose the flat version if it fits.
    pieces.push_back(arena_.MakeFlatChoice(flat_as, break_as));
  } else {
    pieces.push_back(arena_.semi());
  }

  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::Format(const Use& n) {
  // TODO(cdleary): 2024-12-07 This is just a stopgap, we should add reflow
  // capability.
  return arena_.MakeText(n.ToString());
}

DocRef Formatter::Format(const Let& n, bool trailing_semi) {
  std::vector<DocRef> leader_pieces = {
      arena_.Make(n.is_const() ? Keyword::kConst : Keyword::kLet),
      arena_.space(), Fmt(*n.name_def_tree(), comments_, arena_)};
  if (const TypeAnnotation* t = n.type_annotation()) {
    leader_pieces.push_back(arena_.colon());
    leader_pieces.push_back(arena_.space());
    leader_pieces.push_back(Fmt(*t, comments_, arena_));
  }

  leader_pieces.push_back(arena_.space());
  leader_pieces.push_back(arena_.equals());

  const DocRef rhs_doc_internal = Fmt(*n.rhs(), comments_, arena_);

  DocRef rhs_doc = rhs_doc_internal;
  if (trailing_semi) {
    // Reduce the width by 1 so we know we can emit the semi inline.
    rhs_doc = arena_.MakeConcat(arena_.MakeReduceTextWidth(rhs_doc_internal, 1),
                                arena_.semi());
  }

  DocRef body;
  if (n.rhs()->IsBlockedExprAnyLeader()) {
    // For blocked expressions we don't align them to the equals in the let,
    // because it'd shove constructs like `let really_long_identifier = for ...`
    // too far to the right hand side.
    //
    // Note: if you do e.g. a binary operation on blocked constructs as the
    // RHS it /will/ align because we don't look for blocked constructs
    // transitively -- seems reasonable given that's going to look funky no
    // matter what.
    //
    // Note: if it's an expression that acts as a block of contents, but has
    // leading chars, e.g. an invocation, we don't know with reasonable
    // confidence it'll fit on the current line.

    DocRef on_other_ref = ConcatN(arena_, {arena_.space(), rhs_doc});

    // If the blocky expression has a leader, and that leader doesn't fit in
    // the line, we want to nest the whole thing so it has space to look
    // normal and it knows it's in break mode.
    if (n.rhs()->IsBlockedExprWithLeader()) {
      // If the leading component fits, then see what we can fit flat from the
      // RHS, we know we can at least fit that.
      //
      // If the leading component does not fit, emit the whole construct nested
      // on the next line.
      DocRef leader = arena_.MakeConcat(
          arena_.space(), FmtBlockedExprLeader(*n.rhs(), comments_, arena_));
      DocRef nested =
          arena_.MakeNest(arena_.MakeConcat(arena_.hard_line(), rhs_doc));
      on_other_ref = arena_.MakeModeSelect(leader, /*on_flat=*/on_other_ref,
                                           /*on_break=*/nested);
    }

    body = arena_.MakeNestIfFlatFits(
        /*on_nested_flat_ref=*/rhs_doc,
        /*on_other_ref=*/on_other_ref);
  } else {
    // Same as above but with an aligned RHS.
    DocRef aligned_rhs = arena_.MakeAlign(arena_.MakeGroup(rhs_doc));
    body = arena_.MakeNestIfFlatFits(
        /*on_nested_flat_ref=*/rhs_doc,
        /*on_other_ref=*/arena_.MakeConcat(arena_.space(), aligned_rhs));
  }

  DocRef leader = ConcatN(arena_, leader_pieces);
  return ConcatNGroup(arena_, {leader, body});
}

DocRef Formatter::Format(const TypeAlias& n) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (n.extern_type_name()) {
    attr = FmtSvTypeAttribute(*n.extern_type_name(), arena_);
  }
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(Keyword::kType));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.MakeText(n.identifier()));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.equals());
  pieces.push_back(arena_.break1());
  pieces.push_back(Fmt(n.type_annotation(), comments_, arena_));
  return JoinWithAttr(attr, ConcatNGroup(arena_, pieces), arena_);
}

DocRef Formatter::Format(const ModuleMember& n) {
  return absl::visit(Visitor{
                         [&](const Function* n) { return Format(*n); },
                         [&](const Proc* n) { return Format(*n); },
                         [&](const TestFunction* n) { return Format(*n); },
                         [&](const TestProc* n) { return Format(*n); },
                         [&](const QuickCheck* n) { return Format(*n); },
                         [&](const TypeAlias* n) {
                           return arena_.MakeConcat(Format(*n), arena_.semi());
                         },
                         [&](const StructDef* n) { return Format(*n); },
                         [&](const ProcDef* n) { return Format(*n); },
                         [&](const Impl* n) { return Format(*n); },
                         [&](const ConstantDef* n) { return Format(*n); },
                         [&](const EnumDef* n) { return Format(*n); },
                         [&](const Import* n) { return Format(*n); },
                         [&](const Use* n) { return Format(*n); },
                         [&](const ConstAssert* n) {
                           return arena_.MakeConcat(Format(*n), arena_.semi());
                         },
                         [&](const VerbatimNode* n) { return Format(*n); },
                     },
                     n);
}

// Returns whether the given members are of the given "MemberT" and "grouped" --
// that is, one is placed directly on the line above the other. We use this as
// an indicator they should also be grouped in the formatted output for certain
// constructs.
template <typename MemberT>
static bool AreGroupedMembers(const ModuleMember& above,
                              const ModuleMember& below) {
  if (!std::holds_alternative<MemberT*>(above) ||
      !std::holds_alternative<MemberT*>(below)) {
    return false;
  }

  const auto* above_member = std::get<MemberT*>(above);
  const auto* below_member = std::get<MemberT*>(below);
  const Pos& above_limit = above_member->span().limit();
  // Note: if the column number is 0 in the limit, it's an exclusive limit, so
  // really the last character /inclusive/ is on the prior line. This is what
  // this ternary is about.
  int64_t above_effective_lineno = above_limit.colno() == 0
                                       ? above_limit.lineno() - 1
                                       : above_limit.lineno();
  return above_effective_lineno + 1 == below_member->span().start().lineno();
}

// Returns whether the given members are both of the given "MemberT" types.
template <typename MemberT>
static bool AreSameTypes(const ModuleMember& above, const ModuleMember& below) {
  return std::holds_alternative<MemberT*>(above) &&
         std::holds_alternative<MemberT*>(below);
}

// Calculates how many hard lines should be emitted after the given node,
// based on its type and the type of the next one.
static int NumHardLinesAfter(const AstNode* node, const ModuleMember& member,
                             absl::Span<ModuleMember const> siblings, int i) {
  int num_hard_lines = 0;
  if (i + 1 == siblings.size()) {
    // For the last module member we just put a trailing newline at EOF.
    num_hard_lines = 1;
  } else if (AreGroupedMembers<Import>(member, siblings[i + 1]) ||
             AreGroupedMembers<TypeAlias>(member, siblings[i + 1]) ||
             AreGroupedMembers<StructDef>(member, siblings[i + 1]) ||
             AreGroupedMembers<ConstantDef>(member, siblings[i + 1])) {
    // If two (e.g. imports) are adjacent to each other (i.e. no intervening
    // newline) we keep them adjacent to each other in the formatted output.
    num_hard_lines = 1;
  } else if (i < siblings.size() - 1 &&
             AreSameTypes<VerbatimNode>(member, siblings[i + 1])) {
    // Two adjacent verbatim nodes should not have any newlines
    // between them because verbatim nodes already have a newline at the
    // end.
    num_hard_lines = 1;
  } else {
    // For other module members we separate them by an intervening newline.
    num_hard_lines = 2;
  }

  if (dynamic_cast<const VerbatimNode*>(node) != nullptr) {
    // Verbatim nodes already have a newline at the end.
    num_hard_lines--;
  }
  return num_hard_lines;
}

absl::StatusOr<DocRef> Formatter::Format(const Module& n) {
  std::vector<DocRef> pieces;

  if (!n.attributes().empty()) {
    for (ModuleAttribute annotation : n.attributes()) {
      switch (annotation) {
        case ModuleAttribute::kAllowNonstandardConstantNaming:
          pieces.push_back(
              arena_.MakeText("#![allow(nonstandard_constant_naming)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kAllowNonstandardMemberNaming:
          pieces.push_back(
              arena_.MakeText("#![allow(nonstandard_member_naming)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kTypeInferenceVersion2:
          pieces.push_back(arena_.MakeText("#![feature(type_inference_v2)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kAllowUseSyntax:
          pieces.push_back(arena_.MakeText("#![feature(use_syntax)]"));
          pieces.push_back(arena_.hard_line());
          break;
      }
    }
    pieces.push_back(arena_.hard_line());
  }

  std::optional<Pos> last_entity_pos;
  for (size_t i = 0; i < n.top().size(); ++i) {
    const ModuleMember& member = n.top()[i];

    const AstNode* node = ToAstNode(member);

    // If this is a desugared proc function, we skip it, and handle formatting
    // it when we get to the proc node.
    // TODO: https://github.com/google/xls/issues/1029 remove desugared proc
    // functions.
    if (const Function* f = dynamic_cast<const Function*>(node);
        f != nullptr && f->tag() != FunctionTag::kNormal) {
      continue;
    }

    if (const VerbatimNode* v = dynamic_cast<const VerbatimNode*>(node);
        v != nullptr && v->IsEmpty()) {
      if (i < n.top().size() - 1 &&
          !AreSameTypes<VerbatimNode>(member, n.top()[i + 1])) {
        // Add a newline after this node if the next one is not a verbatim node.
        // There's no way to know if the contents of the verbatim node should
        // be part of the previous group or not, so we separate it with a
        // newline.
        pieces.push_back(arena_.hard_line());
      }
      // Skip empty verbatim nodes.
      continue;
    }

    VLOG(3) << "Formatter.Format; " << node->GetNodeTypeName()
            << " module member: " << node->ToString() << " span: "
            << node->GetSpan().value().ToString(arena_.file_table());

    // If there are comment blocks between the last member position and the
    // member we're about the process, we need to emit them.
    std::optional<Span> member_span = node->GetSpan();
    CHECK(member_span.has_value()) << node->GetNodeTypeName();
    const Pos& member_start = member_span->start();
    const Pos& member_limit = member_span->limit();

    // Check the start of this member is >= the last member limit.
    if (last_entity_pos.has_value()) {
      CHECK_GE(member_start, last_entity_pos.value());
    }

    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_entity_pos, member_start, comments_,
                                arena_, &last_comment_span)) {
      pieces.push_back(comments_doc.value());
      pieces.push_back(arena_.hard_line());

      VLOG(3) << "last_comment_span: "
              << last_comment_span.value().ToString(arena_.file_table())
              << " this member start: " << member_start;

      // If the comment abuts the module member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        pieces.push_back(arena_.hard_line());
      }
    }

    // Check the last member position is monotonically increasing.
    if (last_entity_pos.has_value()) {
      CHECK_GT(member_span->limit(), last_entity_pos.value());
    }

    // Here we actually emit the formatted member.
    pieces.push_back(Format(member));

    // Now we reflect the emission of the member.
    last_entity_pos = member_span->limit();

    // See if there are inline comments after the statement.
    last_entity_pos =
        CollectInlineComments(member_limit, last_entity_pos.value(), comments_,
                              arena_, pieces, last_comment_span);

    int num_hard_lines = NumHardLinesAfter(node, member, n.top(), i);
    for (int i = 0; i < num_hard_lines; ++i) {
      pieces.push_back(arena_.hard_line());
    }
  }

  if (std::optional<Pos> last_data_limit = comments_.last_data_limit();
      last_data_limit.has_value() && last_entity_pos < last_data_limit) {
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc =
            EmitCommentsBetween(last_entity_pos, last_data_limit.value(),
                                comments_, arena_, &last_comment_span)) {
      pieces.push_back(comments_doc.value());
      pieces.push_back(arena_.hard_line());
    }
  }

  // Check if there are any comments that were within the span of the
  // module, but not placed.
  if (comments_.last_data_limit().has_value()) {
    Pos last_data_limit = comments_.last_data_limit().value();

    Span span = Span(Pos(last_data_limit.fileno(), 0, 0), last_data_limit);
    if (n.GetSpan().has_value()) {
      span = *n.GetSpan();
    }
    for (const CommentData* comment : comments_.GetComments(span)) {
      if (!comments_.WasPlaced(comment)) {
        const std::string& comment_text =
            std::string{absl::StripTrailingAsciiWhitespace(comment->text)};
        return absl::InternalError(absl::StrFormat(
            "Formatting was skipped because a comment at %s would be "
            "deleted by the formatter: //%s\nThis is probably due to a bug "
            "(which may not have been reported yet). To complete formatting, "
            "try moving the comment to a different line.",
            comment->span.ToString(arena_.file_table()), comment_text));
      }
    }
  }

  return ConcatN(arena_, pieces);
}

static absl::StatusOr<std::string> AutoFmt(const Module& m, Comments& comments,
                                           int64_t text_width) {
  DocArena arena(*m.file_table());
  Formatter formatter(comments, arena);
  XLS_ASSIGN_OR_RETURN(DocRef ref, formatter.Format(m));
  return PrettyPrint(arena, ref, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    int64_t text_width) {
  XLS_RET_CHECK(m.fs_path().has_value());
  FormatDisabler disabler(vfs, comments, *m.fs_path());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> clone,
      CloneModule(m, std::bind_front(&FormatDisabler::operator(), &disabler)));
  return AutoFmt(*clone, comments, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    std::string contents, int64_t text_width) {
  FormatDisabler disabler(vfs, comments, contents);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> clone,
      CloneModule(m, std::bind_front(&FormatDisabler::operator(), &disabler)));
  return AutoFmt(*clone, comments, text_width);
}

// AutoFmt output should be the same as input after whitespace is eliminated
// excepting that:
//
// * we may introduce/remove commas
// * we may reflow comments
// * we may turn struct instance attributes implicit if the names line up
// * (unhandled, non-regexp) we may dedup unnecessarily doubled parentheses
// * (unhandled, non-regexp) we may remove unnecessary parens e.g. on
//   conditionals and match tested exprs
//
// Note that some of these transforms are overly conservative/aggressive to
// avoid false positives.
static std::string FilterForOpportunisticComparison(std::string_view input) {
  return absl::StrReplaceAll(input, {
                                        {"\n", ""},
                                        {" ", ""},
                                        {",", ""},
                                        {"//", ""},
                                    });
}

std::optional<AutoFmtPostconditionViolation>
ObeysAutoFmtOpportunisticPostcondition(std::string_view original,
                                       std::string_view autofmt) {
  std::string want = FilterForOpportunisticComparison(original);
  std::string got = FilterForOpportunisticComparison(autofmt);
  return want == got
             ? std::nullopt
             : std::make_optional(AutoFmtPostconditionViolation{want, got});
}

}  // namespace xls::dslx
