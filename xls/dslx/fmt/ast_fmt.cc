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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/attribute_data.h"
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
#include "xls/ir/channel.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {

// Note: if a comment doc is emitted (i.e. return value has_value()) it does not
// have a trailing hard-line. This is for consistency with other emission
// routines which generally don't emit any whitespace afterwards, just their
// doc.
std::optional<DocRef> Formatter::FormatCommentsBetween(
    std::optional<Pos> start_pos, const Pos& limit_pos,
    std::optional<Span>* last_comment_span) {
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

  const FileTable& file_table = arena_.file_table();
  VLOG(3) << "Looking for comments in span: " << span.ToString(file_table);

  std::vector<DocRef> pieces;

  std::vector<const CommentData*> items = comments_.GetComments(span);
  VLOG(3) << "Found " << items.size() << " comment data items";
  std::optional<Span> previous_comment_span;
  for (size_t i = 0; i < items.size(); ++i) {
    const CommentData* comment_data = items[i];
    comments_.PlaceComment(comment_data);

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
      pieces.push_back(arena_.hard_line());
    }

    pieces.push_back(arena_.MakePrefixedReflow(
        "//",
        std::string{absl::StripTrailingAsciiWhitespace(comment_data->text)}));

    if (i + 1 != items.size()) {
      pieces.push_back(arena_.hard_line());
    }

    previous_comment_span = comment_data->span;
    if (last_comment_span != nullptr) {
      *last_comment_span = comment_data->span;
    }
  }

  if (pieces.empty()) {
    return std::nullopt;
  }

  return ConcatN(arena_, pieces);
}

// If there is a '.' modifier in the attribute given by s (e.g. if it's of the
// form "ProcName.config") strips off the trailing modifier and returns the
// stem.
//
// TODO(https://github.com/google/xls/issues/1029): 2023-12-05 Ideally we'd have
// proc references and avoid strange modifiers on identifiers.
static std::string StripAnyDotModifier(std::string_view s) {
  CHECK_LE(std::count(s.begin(), s.end(), '.'), 1);
  // Check for special identifier for proc config, which is ProcName.config
  // internally, but in spawns we just want to say ProcName.
  if (auto pos = s.rfind('.'); pos != std::string::npos) {
    return std::string(s.substr(0, pos));
  }

  return std::string(s);
}

// A parametric argument, as in parametric instantiation:
//
//    f<FOO as u32>()
//      ^^^^^^^^^^~~~ parametric argument, expr or types can go here generally
DocRef Formatter::FormatParametricArg(const ExprOrType& n) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) {
            DocRef guts = FormatExpr(*n);
            if (dynamic_cast<const NameRef*>(n) != nullptr ||
                dynamic_cast<const ColonRef*>(n) != nullptr ||
                dynamic_cast<const Number*>(n) != nullptr) {
              return guts;  // No need for enclosing curlies.
            }
            return ConcatN(arena_, {arena_.ocurl(), guts, arena_.ccurl()});
          },
          [&](const TypeAnnotation* n) { return FormatTypeAnnotation(*n); },
      },
      n);
}

DocRef Formatter::Format(const Expr* n) {
  CHECK(n != nullptr);
  return FormatExpr(*n);
}

template <typename T>
DocRef Formatter::FormatJoin(absl::Span<const T> items, Joiner joiner,
                             const std::function<DocRef(const T&)>& fmt,
                             bool group) {
  std::vector<DocRef> result;
  result.reserve(items.size());
  for (const T& item : items) {
    result.push_back(fmt(item));
  }
  return FormatJoin(result, joiner, group);
}

DocRef Formatter::FormatJoin(absl::Span<const DocRef> items, Joiner joiner,
                             bool group) {
  if (items.empty()) {
    return arena_.empty();
  }
  std::vector<DocRef> pieces;
  for (size_t i = 0; i < items.size(); ++i) {
    DocRef member = items[i];
    if (i + 1 != items.size()) {  // Not the last item.
      switch (joiner) {
        case Joiner::kCommaSpace:
          pieces.push_back(member);
          pieces.push_back(arena_.comma());
          pieces.push_back(arena_.space());
          break;
        case Joiner::kCommaHardlineTrailingCommaAlways:
          pieces.push_back(member);
          pieces.push_back(arena_.comma());
          pieces.push_back(arena_.hard_line());
          break;
        case Joiner::kCommaBreak1:
          pieces.push_back(member);
          pieces.push_back(arena_.comma());
          pieces.push_back(arena_.break1());
          break;
        case Joiner::kCommaBreak1AsGroupNoTrailingComma:
        case Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak:
        case Joiner::kCommaBreak1AsGroupTrailingCommaAlways: {
          std::vector<DocRef> this_pieces;
          if (i != 0) {  // If it's the first item we don't put a leading space.
            this_pieces.push_back(arena_.break1());
          }
          this_pieces.push_back(member);
          this_pieces.push_back(arena_.comma());
          pieces.push_back(ConcatNGroup(arena_, this_pieces));
          break;
        }
        case Joiner::kSpaceBarBreak:
          pieces.push_back(member);
          pieces.push_back(arena_.space());
          pieces.push_back(arena_.bar());
          pieces.push_back(arena_.break1());
          break;
        case Joiner::kHardLine:
          pieces.push_back(member);
          pieces.push_back(arena_.hard_line());
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
            pieces.push_back(ConcatNGroup(arena_, {arena_.break1(), member}));
          }

          if (joiner == Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak) {
            // With this pattern if we're in break mode (implying we spanned
            // multiple lines), we allow a trailing comma.
            pieces.push_back(
                arena_.MakeFlatChoice(arena_.empty(), arena_.comma()));
          } else if (joiner == Joiner::kCommaBreak1AsGroupTrailingCommaAlways) {
            pieces.push_back(arena_.comma());
          }
          break;
        }
        case Joiner::kCommaHardlineTrailingCommaAlways:
          pieces.push_back(member);
          pieces.push_back(arena_.comma());
          break;
        default:
          pieces.push_back(member);
          break;
      }
    }
  }
  return group ? ConcatNGroup(arena_, pieces) : ConcatN(arena_, pieces);
}

DocRef Formatter::FormatLambda(const Lambda& n) {
  DocRef body = FormatExpr(*n.body());
  DocRef params = FormatJoin<const Param*>(
      n.params(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
      [this](const Param* param) {
        DocRef id = arena_.MakeText(param->identifier());
        if (auto* tvta = dynamic_cast<TypeVariableTypeAnnotation*>(
                param->type_annotation());
            tvta != nullptr && tvta->internal()) {
          return id;
        }
        DocRef type = FormatTypeAnnotation(*param->type_annotation());
        return ConcatN(arena_, {id, arena_.colon(), arena_.space(), type});
      });

  std::vector<DocRef> pieces = {arena_.bar(), params, arena_.bar(),
                                arena_.space()};
  if (n.ExplicitReturn()) {
    pieces.push_back(arena_.arrow());
    pieces.push_back(arena_.space());
    pieces.push_back(FormatTypeAnnotation(*n.return_type()));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(body);

  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatBuiltinTypeAnnotation(const BuiltinTypeAnnotation& n) {
  return arena_.MakeText(BuiltinTypeToString(n.builtin_type()));
}

DocRef Formatter::FormatArrayTypeAnnotation(const ArrayTypeAnnotation& n) {
  return ConcatNGroup(
      arena_, {FormatTypeAnnotation(*n.element_type()), arena_.obracket(),
               arena_.MakeAlign(FormatExpr(*n.dim())), arena_.cbracket()});
}

DocRef Formatter::Format(const TypeAnnotation* n) {
  CHECK(n != nullptr);
  return FormatTypeAnnotation(*n);
}

DocRef Formatter::FormatTupleTypeAnnotation(const TupleTypeAnnotation& n) {
  DocRef guts = FormatJoin<const TypeAnnotation*>(
      n.members(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
      [this](const TypeAnnotation* t) { return Format(t); });

  return ConcatNGroup(
      arena_, {
                  arena_.oparen(),
                  arena_.MakeFlatChoice(
                      /*on_flat=*/guts,
                      /*on_break=*/ConcatNGroup(arena_,
                                                {
                                                    arena_.hard_line(),
                                                    arena_.MakeNest(guts),
                                                    arena_.hard_line(),
                                                })),
                  arena_.cparen(),
              });
}
DocRef Formatter::FormatTypeRef(const TypeRef& n) {
  return absl::visit(
      Visitor{
          [&](const ColonRef* n) { return FormatColonRef(*n); },
          [&](const UseTreeEntry* n) {
            std::string_view identifier =
                n->GetLeafNameDef().value()->identifier();
            return arena_.MakeText(std::string(identifier));
          },
          [&](const auto* n) { return arena_.MakeText(n->identifier()); },
      },
      n.type_definition());
}

DocRef Formatter::FormatTypeRefTypeAnnotation(const TypeRefTypeAnnotation& n) {
  std::vector<DocRef> pieces = {FormatTypeRef(*n.type_ref())};
  if (!n.parametrics().empty()) {
    pieces.push_back(arena_.oangle());
    pieces.push_back(FormatJoin<ExprOrType>(
        absl::MakeConstSpan(n.parametrics()), Joiner::kCommaSpace,
        [this](const ExprOrType& e) { return FormatParametricArg(e); }));
    pieces.push_back(arena_.cangle());
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatChannelTypeAnnotation(const ChannelTypeAnnotation& n) {
  std::vector<DocRef> pieces = {
      arena_.Make(Keyword::kChan),
      arena_.oangle(),
      FormatTypeAnnotation(*n.payload()),
      arena_.cangle(),
  };
  if (n.dims().has_value()) {
    pieces.reserve(pieces.size() + 3 * n.dims()->size());
    for (const Expr* dim : *n.dims()) {
      pieces.push_back(arena_.obracket());
      pieces.push_back(FormatExpr(*dim));
      pieces.push_back(arena_.cbracket());
    }
  }

  pieces.push_back(arena_.space());
  pieces.push_back(arena_.Make(
      n.direction() == ChannelDirection::kIn ? Keyword::kIn : Keyword::kOut));
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatTypeVariableTypeAnnotation(
    const TypeVariableTypeAnnotation& n) {
  std::vector<DocRef> pieces = {FormatNameRef(*n.type_variable())};
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatTypeAnnotation(const TypeAnnotation& n) {
  if (auto* t = dynamic_cast<const BuiltinTypeAnnotation*>(&n)) {
    return FormatBuiltinTypeAnnotation(*t);
  }
  if (auto* t = dynamic_cast<const TupleTypeAnnotation*>(&n)) {
    return FormatTupleTypeAnnotation(*t);
  }
  if (auto* t = dynamic_cast<const ArrayTypeAnnotation*>(&n)) {
    return FormatArrayTypeAnnotation(*t);
  }
  if (auto* t = dynamic_cast<const TypeRefTypeAnnotation*>(&n)) {
    return FormatTypeRefTypeAnnotation(*t);
  }
  if (auto* t = dynamic_cast<const ChannelTypeAnnotation*>(&n)) {
    return FormatChannelTypeAnnotation(*t);
  }
  if (auto* t = dynamic_cast<const TypeVariableTypeAnnotation*>(&n)) {
    return FormatTypeVariableTypeAnnotation(*t);
  }
  if (dynamic_cast<const GenericTypeAnnotation*>(&n)) {
    return arena_.Make(Keyword::kType);
  }
  if (dynamic_cast<const SelfTypeAnnotation*>(&n)) {
    return arena_.Make(Keyword::kSelfType);
  }

  LOG(FATAL) << "handle type annotation: " << n.ToString()
             << " type: " << n.GetNodeTypeName();
}
DocRef Formatter::FormatJoinWithAttrs(absl::Span<const DocRef> attrs,
                                      DocRef rest) {
  if (attrs.empty()) {
    return rest;
  }
  return arena_.MakeConcat(ConcatNGroup(arena_, attrs), rest);
}

DocRef Formatter::FormatJoinWithAttr(std::optional<DocRef> attr, DocRef rest) {
  std::vector<DocRef> attrs;
  if (attr.has_value()) {
    attrs.push_back(*attr);
  }
  return FormatJoinWithAttrs(attrs, rest);
}

DocRef Formatter::FormatAttribute(const Attribute& n) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena_.MakeText("#"));
  pieces.push_back(arena_.obracket());
  pieces.push_back(arena_.MakeText(AttributeKindToString(n.attribute_kind())));

  if (!n.args().empty()) {
    pieces.push_back(arena_.oparen());
    const std::vector<AttributeData::Argument>& args = n.args();
    for (size_t i = 0; i < args.size(); ++i) {
      const AttributeData::Argument& arg = args[i];
      absl::visit(
          Visitor{
              [&](std::string str) { pieces.push_back(arena_.MakeText(str)); },
              [&](AttributeData::StringLiteralArgument arg) {
                pieces.push_back(
                    arena_.MakeText(absl::Substitute("\"$0\"", arg.text)));
              },
              [&](AttributeData::IntKeyValueArgument arg) {
                pieces.push_back(arena_.MakeText(
                    absl::Substitute("$0=$1", arg.first, arg.second)));
              },
              [&](AttributeData::StringKeyValueArgument arg) {
                if (arg.is_backticked) {
                  pieces.push_back(arena_.MakeText(
                      absl::Substitute("$0=`$1`", arg.first, arg.second)));
                } else {
                  pieces.push_back(arena_.MakeText(
                      absl::Substitute("$0=$1", arg.first, arg.second)));
                }
              },
          },
          arg);
      if (i < args.size() - 1) {
        pieces.push_back(arena_.comma());
        pieces.push_back(arena_.space());
      }
    }
    pieces.push_back(arena_.cparen());
  }
  pieces.push_back(arena_.cbracket());
  pieces.push_back(arena_.hard_line());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatChannelConfig(const ChannelConfig& n) {
  std::vector<DocRef> pieces{
      arena_.MakeText("#"),
      arena_.obracket(),
      arena_.MakeText("channel"),
      arena_.oparen(),
  };
  for (const auto& [idx, key_value] : ::iter::enumerate(n.GetDslxKwargs())) {
    const auto& [key, value] = key_value;
    pieces.push_back(arena_.MakeText(key));
    pieces.push_back(arena_.equals());
    pieces.push_back(arena_.MakeText(value));
    if (idx < n.GetDslxKwargs().size() - 1) {
      pieces.push_back(arena_.comma());
      pieces.push_back(arena_.space());
    }
  }

  pieces.push_back(arena_.cparen());
  pieces.push_back(arena_.cbracket());
  pieces.push_back(arena_.hard_line());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatNameDef(const NameDef& n) {
  return arena_.MakeText(n.identifier());
}

DocRef Formatter::FormatNameRef(const NameRef& n) {
  return arena_.MakeText(StripAnyDotModifier(n.identifier()));
}

DocRef Formatter::FormatNumber(const Number& n) {
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
    num_text = arena_.MakeText(absl::StrFormat("'%s'", guts));
  } else {
    num_text = arena_.MakeText(n.text());
  }
  if (const TypeAnnotation* type = n.type_annotation()) {
    return ConcatNGroup(
        arena_, {FormatTypeAnnotation(*type), arena_.colon(), num_text});
  }
  return num_text;
}

DocRef Formatter::FormatWildcardPattern(const WildcardPattern& n) {
  return arena_.underscore();
}

DocRef Formatter::FormatRestOfTuple(const RestOfTuple& n) {
  return arena_.dot_dot();
}

DocRef Formatter::FormatMakeArrayLeader(const Array& n) {
  const TypeAnnotation* t = n.type_annotation();
  if (t == nullptr) {
    return arena_.obracket();
  }
  std::vector<DocRef> pieces;
  pieces.push_back(FormatTypeAnnotation(*t));
  pieces.push_back(arena_.colon());
  pieces.push_back(arena_.obracket());
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatFlatBody(const Array& n) {
  std::vector<DocRef> flat_pieces;
  flat_pieces.push_back(
      FormatJoin<const Expr*>(n.members(), Joiner::kCommaSpace,
                              [this](const Expr* e) { return Format(e); }));
  if (n.has_ellipsis()) {
    // Note: while zero members with ellipsis is invalid at type checking, we
    // may choose not to flag it as a parse-time error, in which case we could
    // have it in the AST.
    if (!n.members().empty()) {
      flat_pieces.push_back(arena_.comma());
    }

    flat_pieces.push_back(arena_.space());
    flat_pieces.push_back(arena_.MakeText("..."));
  }
  flat_pieces.push_back(arena_.cbracket());
  return ConcatN(arena_, flat_pieces);
}

DocRef Formatter::FormatBreakBody(const Array& n) {
  std::vector<DocRef> rest;
  rest.push_back(arena_.break0());

  std::vector<DocRef> member_pieces;
  member_pieces.push_back(FormatJoin<const Expr*>(
      n.members(), Joiner::kCommaBreak1AsGroupTrailingCommaAlways,
      [this](const Expr* e) { return Format(e); }));

  if (n.has_ellipsis()) {
    member_pieces.push_back(
        ConcatNGroup(arena_, {arena_.break1(), arena_.MakeText("...")}));
  }

  DocRef inner = ConcatNGroup(arena_, member_pieces);
  rest.push_back(arena_.MakeFlatChoice(inner, arena_.MakeNest(inner)));
  rest.push_back(arena_.break0());
  rest.push_back(arena_.cbracket());

  return ConcatNGroup(arena_, rest);
}

DocRef Formatter::FormatArray(const Array& n) {
  DocRef on_break_body = FormatBreakBody(n);
  DocRef on_flat_body = FormatFlatBody(n);

  DocRef body = arena_.MakeGroup(arena_.MakeFlatChoice(
      /*on_flat=*/on_flat_body, /*on_break=*/on_break_body));
  return arena_.MakeConcat(FormatMakeArrayLeader(n), body);
}

DocRef Formatter::FormatAttr(const Attr& n) {
  Precedence op_precedence = n.GetPrecedenceWithoutParens();
  const Expr& lhs = *n.lhs();
  Precedence lhs_precedence = lhs.GetPrecedence();
  std::vector<DocRef> pieces;
  if (WeakerThan(lhs_precedence, op_precedence) && IsInfix(lhs_precedence)) {
    pieces.push_back(arena_.oparen());
    pieces.push_back(FormatExpr(lhs));
    pieces.push_back(arena_.cparen());
  } else {
    pieces.push_back(FormatExpr(lhs));
  }
  pieces.push_back(arena_.dot());
  pieces.push_back(arena_.MakeText(std::string{n.attr()}));
  return ConcatNGroup(arena_, pieces);
}

std::optional<DocRef> Formatter::FormatCommentsNested(const Pos start,
                                                      const Pos limit) {
  std::vector<const CommentData*> items =
      comments_.GetComments(Span(start, limit));
  if (items.empty()) {
    return std::nullopt;
  }

  std::vector<DocRef> pieces;
  // Add the first comment "in line"
  auto first = FormatCommentsBetween(start, items[0]->span.limit(),
                                     /*last_comment_span=*/nullptr);
  pieces.push_back(*first);
  pieces.push_back(arena_.hard_line());
  if (items.size() > 1) {
    // Add the nth through last comment as a new nested document.
    auto nested_comments = FormatCommentsBetween(items[1]->span.start(), limit,
                                                 /*last_comment_span=*/nullptr);
    // EmitCommentsBetween doesn't indent (or nest) the 2nd through Nth
    // comments, so we have to do it manually here.
    pieces.push_back(arena_.MakeNest(*nested_comments));
    pieces.push_back(arena_.hard_line());
  }
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatBinop(const Binop& n) {
  Precedence op_precedence = n.GetPrecedenceWithoutParens();
  const Expr& lhs = *n.lhs();
  const Expr& rhs = *n.rhs();
  Precedence lhs_precedence = lhs.GetPrecedence();

  auto emit = [&](const Expr& e, bool parens, std::vector<DocRef>& pieces) {
    if (parens) {
      pieces.push_back(arena_.oparen());
      pieces.push_back(FormatExpr(e));
      pieces.push_back(arena_.cparen());
    } else {
      pieces.push_back(FormatExpr(e));
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

  DocRef lhs_ref = ConcatN(arena_, lhs_pieces);
  bool nest_rhs = false;

  // If there are comments between the LHS and the operator, we want to emit
  // them before the operator.
  if (std::optional<DocRef> comments_doc =
          FormatCommentsNested(lhs.span().limit(), n.op_span().start())) {
    lhs_ref = ConcatN(arena_, {lhs_ref, arena_.space(), *comments_doc});
    nest_rhs = true;
  }

  bool emitted_op = false;
  // If there are comments between the operator and the RHS, emit them now.
  if (nest_rhs) {
    if (std::optional<DocRef> comments_doc =
            FormatCommentsBetween(n.op_span().limit(), rhs.span().start(),
                                  /*last_comment_span=*/nullptr)) {
      // If the RHS is already being nested, don't nest the comments. probably.
      lhs_ref = ConcatN(
          arena_,
          {lhs_ref,
           arena_.MakeNest(ConcatN(
               arena_, {arena_.MakeText(BinopKindFormat(n.binop_kind())),
                        arena_.space(), *comments_doc, arena_.hard_line()}))});
      emitted_op = true;
    }
  } else if (std::optional<DocRef> comments_doc = FormatCommentsNested(
                 n.op_span().limit(), rhs.span().start())) {
    // The space is needed since we didn't nest the RHS yet
    lhs_ref = ConcatN(arena_, {lhs_ref, arena_.space(),
                               arena_.MakeText(BinopKindFormat(n.binop_kind())),
                               arena_.space(), *comments_doc});
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
  DocRef rhs_ref = ConcatN(arena_, rhs_pieces);
  std::vector<DocRef> more_rhs_pieces;
  if (!nest_rhs) {
    // If we didn't nest the RHS, we need to add a space to separate it from the
    // operator.
    more_rhs_pieces.push_back(arena_.space());
  }
  if (!emitted_op) {
    // If we didn't emit the operator, we need to add it now.
    more_rhs_pieces.push_back(arena_.MakeText(BinopKindFormat(n.binop_kind())));
    more_rhs_pieces.push_back(arena_.break1());
  }
  more_rhs_pieces.push_back(rhs_ref);
  rhs_ref = ConcatNGroup(arena_, more_rhs_pieces);
  if (nest_rhs) {
    rhs_ref = arena_.MakeNest(rhs_ref);
  }

  return ConcatN(arena_, {
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
Pos AdjustCommentLimit(const Span& comment_span, DocArena& arena_,
                       DocRef comment_doc) {
  CHECK_EQ(comment_span.limit().colno(), 0);
  CHECK_GT(comment_span.limit().lineno(), 0);
  return Pos(comment_span.start().fileno(), comment_span.limit().lineno() - 1,
             std::numeric_limits<int32_t>::max());
}

// Looks for inline comments after the `prev_limit` and adds relevant `DocRef`
// to `pieces`. Returns `last_entity_pos`, updated if comments were found.
Pos Formatter::FormatCollectInlineComments(
    const Pos& prev_limit, const Pos& last_entity_pos,
    std::vector<DocRef>& pieces, std::optional<Span> last_comment_span) {
  const Pos next_line(prev_limit.fileno(), prev_limit.lineno() + 1, 0);
  if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
          last_entity_pos, next_line, &last_comment_span)) {
    VLOG(3) << "Saw inline comment: "
            << arena_.ToDebugString(comments_doc.value())
            << " last_comment_span: "
            << last_comment_span.value().ToString(arena_.file_table());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.MakeAlign(comments_doc.value()));

    return AdjustCommentLimit(last_comment_span.value(), arena_,
                              comments_doc.value());
  }
  return last_entity_pos;
}

DocRef Formatter::FormatSingleStatementBlockInline(const StatementBlock& n,
                                                   bool add_curls) {
  std::vector<DocRef> pieces;
  if (add_curls) {
    pieces = {arena_.ocurl(), arena_.break1()};
  }

  pieces.push_back(
      FormatStatement(*n.statements()[0], /*trailing_semi=*/n.trailing_semi()));

  if (add_curls) {
    pieces.push_back(arena_.break1());
    pieces.push_back(arena_.ccurl());
  }
  DocRef block_group = ConcatNGroup(arena_, pieces);
  return arena_.MakeFlatChoice(block_group, arena_.MakeNest(block_group));
}

// Note: we only add leading/trailing spaces in the block if add_curls is true.
DocRef Formatter::FormatBlock(const StatementBlock& n,
                              const FormatBlockOptions& options) {
  int actual_start_idx = options.start_idx.has_value() ? *options.start_idx : 0;
  int actual_end_idx =
      options.end_idx.has_value() ? *options.end_idx : n.statements().size();
  absl::Span<const DocRef> prepend_statements = options.prepend_statements;
  absl::Span<const DocRef> append_statements = options.append_statements;
  bool force_trailing_semi = options.force_trailing_semi;
  bool add_curls = options.add_curls;
  bool force_multiline = options.force_multiline;

  bool has_comments = comments_.HasComments(n.span());

  if (actual_start_idx == actual_end_idx && !has_comments &&
      prepend_statements.empty() && append_statements.empty()) {
    if (add_curls) {
      return ConcatNGroup(arena_,
                          {arena_.ocurl(), arena_.break0(), arena_.ccurl()});
    }
    return arena_.break0();
  }

  // We only want to flatten single-statement blocks -- multi-statement blocks
  // we always make line breaks between the statements.
  bool is_default_block =
      !options.start_idx.has_value() && !options.end_idx.has_value() &&
      options.prepend_statements.empty() && options.append_statements.empty() &&
      !options.force_trailing_semi;
  if (actual_end_idx - actual_start_idx == 1 && is_default_block &&
      !force_multiline && !has_comments) {
    return FormatSingleStatementBlockInline(n, add_curls);
  }

  // Emit a '{' then nest to emit statements with semis, then emit a '}' outside
  // the nesting.
  std::vector<DocRef> top;

  if (add_curls) {
    top.push_back(arena_.ocurl());
    top.push_back(arena_.hard_line());
  }

  // For our initial condition, we say the last entity we emitted is right after
  // the start of the block (i.e. the open curl).
  const Pos start_entity_pos = n.span().start().BumpCol();
  Pos last_entity_pos = start_entity_pos;

  std::vector<DocRef> statements;

  for (DocRef doc : prepend_statements) {
    statements.push_back(doc);
    statements.push_back(arena_.hard_line());
  }

  bool last_stmt_was_verbatim = false;
  for (int i = actual_start_idx; i < actual_end_idx; ++i) {
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
            << "` span: " << stmt_span.value().ToString(arena_.file_table())
            << " last_entity_pos: " << last_entity_pos;

    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_entity_pos, stmt_start, &last_comment_span)) {
      VLOG(5) << "emitting comment ahead of: `" << stmt->ToString() << "`"
              << " last entity position: " << last_entity_pos
              << " last_comment_span: "
              << last_comment_span.value().ToString(arena_.file_table());
      // If there's a line break between the last entity and this comment, we
      // retain it in the output (i.e. in paragraph style).
      if (last_entity_pos != start_entity_pos &&
          last_entity_pos.lineno() + 1 < last_comment_span->start().lineno()) {
        stmt_pieces.push_back(arena_.hard_line());
      }

      stmt_pieces.push_back(comments_doc.value());
      stmt_pieces.push_back(arena_.hard_line());

      last_entity_pos = AdjustCommentLimit(last_comment_span.value(), arena_,
                                           comments_doc.value());

      // See if we want a line break between the comment we just emitted and the
      // statement we're about to emit.
      if (last_entity_pos.lineno() + 1 < stmt_start.lineno()) {
        stmt_pieces.push_back(arena_.hard_line());
      }

    } else {  // No comments to emit ahead of the statement.
      VLOG(5) << "no comments to emit ahead of statement: " << stmt->ToString();
      // If there's a line break between the last entity and this statement, we
      // retain it in the output (i.e. in paragraph style).
      if (last_entity_pos.lineno() + 1 < stmt_start.lineno()) {
        stmt_pieces.push_back(arena_.hard_line());
      }
    }

    // Here we emit the formatted statement.
    bool last_stmt = i + 1 == actual_end_idx;
    bool need_semi = n.trailing_semi() || !last_stmt ||
                     !append_statements.empty() || force_trailing_semi;
    std::vector<DocRef> stmt_semi = {FormatStatement(*stmt, need_semi)};

    // Now we reflect the emission of the statement.
    last_entity_pos = stmt_limit;

    stmt_pieces.push_back(ConcatNGroup(arena_, stmt_semi));
    statements.push_back(ConcatN(arena_, stmt_pieces));

    last_entity_pos = FormatCollectInlineComments(
        stmt_limit, last_entity_pos, statements, last_comment_span);

    if (!last_stmt || !append_statements.empty()) {
      statements.push_back(arena_.hard_line());
    }
  }

  // If the last statement was a verbatim, it already included a hard line,
  // so we don't need one right now.
  bool needs_hardline = !last_stmt_was_verbatim;

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
          last_entity_pos, n.span().limit(), &last_comment_span)) {
    VLOG(5) << "last entity position: " << last_entity_pos
            << " last_comment_span.start: " << last_comment_span->start();

    // If there's a line break between the last entity and this comment, we
    // retain it in the output (i.e. in paragraph style).
    if (last_entity_pos.lineno() + 1 < last_comment_span->start().lineno()) {
      statements.push_back(arena_.hard_line());
    }

    if (!last_stmt_was_verbatim) {
      // Skip the hard line before the last comment if the last one was a
      // verbatim, because it already included one.
      statements.push_back(arena_.hard_line());
    }

    statements.push_back(comments_doc.value());
    if (!append_statements.empty()) {
      statements.push_back(arena_.hard_line());
    }

    // We always need a hard line after the last comment.
    needs_hardline = true;
  }

  for (size_t i = 0; i < append_statements.size(); ++i) {
    statements.push_back(append_statements[i]);
    if (i + 1 < append_statements.size()) {
      statements.push_back(arena_.hard_line());
    }
  }

  if (options.add_nest) {
    top.push_back(arena_.MakeNest(ConcatN(arena_, statements)));
  } else {
    top.push_back(ConcatN(arena_, statements));
  }
  if (add_curls) {
    if (needs_hardline || !append_statements.empty()) {
      top.push_back(arena_.hard_line());
    }
    top.push_back(arena_.ccurl());
  } else {
    // If we're not putting hard lines in we want to at least check that we'll
    // force this all into break mode for multi-line emission.
    //
    // Note that the "inline block" case is handled specially above.
    top.push_back(arena_.force_break_mode());
  }

  return ConcatNGroup(arena_, top);
}

DocRef Formatter::FormatCast(const Cast& n) {
  DocRef lhs = FormatExpr(*n.expr());

  Precedence arg_precedence = n.expr()->GetPrecedence();
  if (WeakerThan(arg_precedence, Precedence::kAs)) {
    lhs = ConcatN(arena_, {arena_.oparen(), lhs, arena_.cparen()});
  }

  return ConcatNGroup(
      arena_, {lhs, arena_.space(), arena_.Make(Keyword::kAs), arena_.break1(),
               FormatTypeAnnotation(*n.type_annotation())});
}

DocRef Formatter::FormatChannelDecl(const ChannelDecl& n) {
  std::optional<DocRef> channel_attribute;
  if (n.channel_config().has_value()) {
    channel_attribute = FormatChannelConfig(*n.channel_config());
  }
  std::vector<DocRef> pieces{
      channel_attribute.value_or(arena_.empty()),
      arena_.Make(Keyword::kChan),
      arena_.oangle(),
      FormatTypeAnnotation(*n.type()),
  };
  // channel_config().has_value() -> fifo_config().has_value(), but we've
  // already handled it above.
  if (!n.channel_config().has_value() && n.fifo_depth().has_value()) {
    pieces.push_back(arena_.comma());
    pieces.push_back(arena_.space());
    pieces.push_back(FormatExpr(*n.fifo_depth().value()));
  }
  pieces.push_back(arena_.cangle());
  if (n.dims().has_value()) {
    for (const Expr* dim : *n.dims()) {
      pieces.push_back(arena_.obracket());
      pieces.push_back(FormatExpr(*dim));
      pieces.push_back(arena_.cbracket());
    }
  }
  pieces.push_back(arena_.oparen());
  pieces.push_back(FormatExpr(n.channel_name_expr()));
  pieces.push_back(arena_.cparen());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatColonRef(const ColonRef& n) {
  DocRef subject = absl::visit(
      Visitor{
          [&](const Expr* n) { return FormatExpr(*n); },
          [&](const TypeAnnotation* n) { return FormatTypeAnnotation(*n); },
      },
      n.subject());

  return ConcatNGroup(arena_, {subject, arena_.colon_colon(),
                               arena_.MakeText(StripAnyDotModifier(n.attr()))});
}

DocRef Formatter::FormatForLoopBaseLeader(Keyword keyword, DocRef names_ref,
                                          const ForLoopBase& n,
                                          bool is_const_for) {
  std::vector<DocRef> pieces;
  if (is_const_for) {
    pieces.push_back(arena_.Make(Keyword::kConst));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(keyword));
  pieces.push_back(arena_.MakeNestIfFlatFits(
      /*on_nested_flat_ref=*/names_ref,
      /*on_other_ref=*/arena_.MakeConcat(arena_.space(), names_ref)));

  if (n.type_annotation() != nullptr) {
    pieces.push_back(arena_.colon());
    pieces.push_back(arena_.space());
    pieces.push_back(FormatTypeAnnotation(*n.type_annotation()));
  }

  pieces.push_back(arena_.space());
  pieces.push_back(arena_.Make(Keyword::kIn));

  DocRef iterable_ref = FormatExpr(*n.iterable());
  pieces.push_back(arena_.MakeNestIfFlatFits(
      /*on_nested_flat_ref=*/iterable_ref,
      /*on_other_ref=*/arena_.MakeConcat(arena_.space(), iterable_ref)));

  pieces.push_back(arena_.space());
  pieces.push_back(arena_.ocurl());
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatForLoopBase(Keyword keyword, const ForLoopBase& n,
                                    bool is_const_for) {
  CHECK(keyword == Keyword::kFor || keyword == Keyword::kUnrollFor)
      << static_cast<std::underlying_type_t<Keyword>>(keyword);
  DocRef names_ref = FormatNameDefTree(*n.names());
  DocRef leader = FormatForLoopBaseLeader(keyword, names_ref, n, is_const_for);

  std::vector<DocRef> body_pieces;
  body_pieces.push_back(arena_.hard_line());
  body_pieces.push_back(FormatBlock(
      *n.body(),
      FormatBlockOptions{.add_curls = false, .force_multiline = true}));
  body_pieces.push_back(arena_.hard_line());
  body_pieces.push_back(arena_.ccurl());
  body_pieces.push_back(ConcatNGroup(
      arena_, {arena_.oparen(), FormatExpr(*n.init()), arena_.cparen()}));

  return arena_.MakeConcat(leader, ConcatN(arena_, body_pieces));
}

DocRef Formatter::FormatConstFor(const ConstFor& n) {
  Keyword keyword = n.IsUnrollFor() ? Keyword::kUnrollFor : Keyword::kFor;
  return FormatForLoopBase(keyword, n, !n.IsUnrollFor());
}

DocRef Formatter::FormatFor(const For& n) {
  return FormatForLoopBase(Keyword::kFor, n, /*is_const_for=*/false);
}

DocRef Formatter::FormatFormatMacro(const FormatMacro& n) {
  std::vector<DocRef> pieces = {arena_.MakeText(n.macro()), arena_.oparen()};
  if (n.condition().has_value()) {
    pieces.push_back(FormatExpr(**n.condition()));
    pieces.push_back(arena_.comma());
    pieces.push_back(arena_.break1());
  }
  if (n.verbosity().has_value()) {
    pieces.push_back(FormatExpr(**n.verbosity()));
    pieces.push_back(arena_.comma());
    pieces.push_back(arena_.break1());
  }

  pieces.push_back(arena_.MakeText(
      absl::StrCat("\"", StepsToXlsFormatString(n.format()), "\"")));

  if (!n.args().empty()) {
    pieces.push_back(arena_.comma());
    pieces.push_back(arena_.break1());
  }

  pieces.push_back(
      FormatJoin<const Expr*>(n.args(), Joiner::kCommaSpace,
                              [this](const Expr* e) { return Format(e); }));
  pieces.push_back(arena_.cparen());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatSlice(const Slice& n) {
  std::vector<DocRef> pieces;

  if (n.start() != nullptr) {
    pieces.push_back(FormatExpr(*n.start()));
  }
  pieces.push_back(arena_.break0());
  pieces.push_back(arena_.colon());
  if (n.limit() != nullptr) {
    pieces.push_back(FormatExpr(*n.limit()));
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatWidthSlice(const WidthSlice& n) {
  return ConcatNGroup(arena_, {
                                  FormatExpr(*n.start()),
                                  arena_.break0(),
                                  arena_.plus_colon(),
                                  arena_.break0(),
                                  FormatTypeAnnotation(*n.width()),
                              });
}

DocRef Formatter::FormatIndexRhs(const IndexRhs& n) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return FormatExpr(*n); },
          [&](const Slice* n) { return FormatSlice(*n); },
          [&](const WidthSlice* n) { return FormatWidthSlice(*n); },
      },
      n);
}

DocRef Formatter::FormatIndex(const Index& n) {
  std::vector<DocRef> pieces;
  if (WeakerThan(n.lhs()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena_.oparen());
    pieces.push_back(FormatExpr(*n.lhs()));
    pieces.push_back(arena_.cparen());
  } else {
    pieces.push_back(FormatExpr(*n.lhs()));
  }
  pieces.push_back(arena_.obracket());
  pieces.push_back(arena_.MakeAlign(FormatIndexRhs(n.rhs())));
  pieces.push_back(arena_.cbracket());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatExprOrType(const ExprOrType& n) {
  return absl::visit(
      Visitor{
          [&](const Expr* n) { return FormatExpr(*n); },
          [&](const TypeAnnotation* n) { return FormatTypeAnnotation(*n); },
      },
      n);
}

std::optional<DocRef> Formatter::FormatExplicitParametrics(
    absl::Span<const ExprOrType> parametrics) {
  if (parametrics.empty()) {
    return std::nullopt;
  }
  return ConcatNGroup(arena_,
                      {arena_.oangle(), arena_.break0(),
                       FormatJoin<ExprOrType>(parametrics, Joiner::kCommaSpace,
                                              [this](const ExprOrType& e) {
                                                return FormatParametricArg(e);
                                              }),
                       arena_.cangle()});
}

DocRef Formatter::FormatFunctionRef(const FunctionRef& n) {
  DocRef callee_doc = FormatExpr(*n.callee());
  std::optional<DocRef> parametrics_doc =
      FormatExplicitParametrics(absl::MakeConstSpan(n.explicit_parametrics()));
  return parametrics_doc.has_value()
             ? ConcatN(arena_, {callee_doc, *parametrics_doc})
             : callee_doc;
}

DocRef Formatter::FormatInvocation(const Invocation& n) {
  DocRef callee_doc = FormatExpr(*n.callee());
  std::optional<DocRef> parametrics_doc =
      FormatExplicitParametrics(n.explicit_parametrics());

  DocRef args_doc_internal = FormatJoin<const Expr*>(
      n.args(), Joiner::kCommaBreak1AsGroupNoTrailingComma,
      [this](const Expr* e) { return Format(e); });

  // Group for the args tokens.
  std::vector<DocRef> arg_pieces = {
      arena_.MakeNestIfFlatFits(
          /*on_nested_flat_ref=*/args_doc_internal,
          /*on_other_ref=*/arena_.MakeAlign(args_doc_internal)),
      arena_.cparen()};
  DocRef args_doc = ConcatNGroup(arena_, arg_pieces);
  DocRef args_doc_nested = arena_.MakeNest(args_doc);

  // This is the flat version -- it simply concats the pieces together.
  DocRef flat = parametrics_doc.has_value()
                    ? ConcatN(arena_, {callee_doc, parametrics_doc.value(),
                                       arena_.oparen(), args_doc})
                    : ConcatN(arena_, {callee_doc, arena_.oparen(), args_doc});

  // This doc ref is for the "I can emit *the leader* flat" case; i.e. the
  // callee (or the callee with parametric args).
  //
  // The parametrics have a break at the start (after the oangle) that can be
  // triggered, and the arguments have a break at the start (after the oparen)
  // that can be triggered.
  DocRef leader_flat =
      parametrics_doc.has_value()
          ? ConcatN(arena_,
                    {callee_doc, arena_.MakeNest(parametrics_doc.value()),
                     arena_.oparen(), arena_.break0(), args_doc_nested})
          : ConcatN(arena_, {callee_doc, arena_.oparen(), arena_.break0(),
                             args_doc_nested});

  DocRef result = arena_.MakeGroup(
      arena_.MakeFlatChoice(/*on_flat=*/flat, /*on_break=*/leader_flat));

  return result;
}

DocRef Formatter::Format(const NameDefTree* n) { return FormatNameDefTree(*n); }

DocRef Formatter::FormatMatchArm(const MatchArm& n) {
  std::vector<DocRef> pieces;
  pieces.push_back(FormatJoin<const NameDefTree*>(
      n.patterns(), Joiner::kSpaceBarBreak,
      [this](const NameDefTree* n) { return Format(n); }));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.fat_arrow());

  const Pos& rhs_start = n.expr()->span().start();

  DocRef rhs_doc = arena_.MakeGroup(FormatExpr(*n.expr()));

  // Check for a comment between the arrow position and the RHS expression. This
  // can be needed when the RHS is not a block but an expression decorated with
  // a comment as if it were a block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
          n.span().start(), rhs_start, &last_comment_span)) {
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.space());
    pieces.push_back(comments_doc.value());
    pieces.push_back(arena_.hard_line());
    pieces.push_back(arena_.MakeNest(rhs_doc));
  } else {
    // If the RHS is a blocked expression, e.g. a struct instance, we don't
    // align it to the fat arrow indicated column.
    if (IsBlockedExprNoLeader(*n.expr()) ||
        IsBlockedExprWithLeader(*n.expr())) {
      pieces.push_back(arena_.space());
      pieces.push_back(arena_.MakeGroup(rhs_doc));
    } else {
      DocRef flat_choice_group = arena_.MakeGroup(arena_.MakeFlatChoice(
          /*on_flat=*/arena_.MakeConcat(arena_.space(), rhs_doc),
          /*on_break=*/arena_.MakeConcat(arena_.hard_line(),
                                         arena_.MakeNest(rhs_doc))));
      pieces.push_back(flat_choice_group);
    }
  }

  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatMatch(const Match& n) {
  std::vector<DocRef> pieces;
  if (n.IsConst()) {
    pieces.push_back(arena_.Make(Keyword::kConst));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(ConcatNGroup(
      arena_, {arena_.Make(Keyword::kMatch), arena_.space(),
               Format(n.matched()), arena_.space(), arena_.ocurl()}));

  pieces.push_back(arena_.hard_line());

  std::vector<DocRef> nested;

  Pos last_member_pos = n.matched()->span().limit();

  for (size_t i = 0; i < n.arms().size(); ++i) {
    const MatchArm* arm = n.arms()[i];

    // Note: the match arm member starts at the first pattern match.
    const Pos& member_start = arm->span().start();

    const Pos& member_limit = arm->span().limit();

    // See if there are comments above the match arm.
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_member_pos, member_start, &last_comment_span)) {
      nested.push_back(comments_doc.value());
      nested.push_back(arena_.hard_line());

      // If the comment abuts the member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        nested.push_back(arena_.hard_line());
      }
    }

    nested.push_back(FormatMatchArm(*arm));
    nested.push_back(arena_.comma());

    last_member_pos = arm->span().limit();

    // See if there are inline comments after the arm.
    last_member_pos = FormatCollectInlineComments(member_limit, last_member_pos,
                                                  nested, last_comment_span);

    if (i + 1 != n.arms().size()) {
      nested.push_back(arena_.hard_line());
    }
  }

  pieces.push_back(arena_.MakeNest(ConcatN(arena_, nested)));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(arena_.ccurl());
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatSpawn(const Spawn& n) {
  return ConcatNGroup(arena_, {arena_.MakeText("spawn"), arena_.space(),
                               FormatInvocation(*n.config())}

  );
}

DocRef Formatter::FormatTupleWithoutComments(const XlsTuple& n) {
  // 1-element tuples are a special case- we always want a trailing comma and
  // never want it to be broken up. Handle separately here.
  if (n.members().size() == 1) {
    return ConcatNGroup(arena_, {
                                    arena_.oparen(),
                                    FormatExpr(*n.members()[0]),
                                    arena_.comma(),
                                    arena_.cparen(),
                                });
  }

  DocRef guts = FormatJoin<const Expr*>(
      n.members(), Joiner::kCommaBreak1AsGroupTrailingCommaOnBreak,
      [this](const Expr* e) { return Format(e); });

  return ConcatNGroup(
      arena_, {
                  arena_.oparen(),
                  arena_.MakeFlatChoice(
                      /*on_flat=*/guts,
                      /*on_break=*/ConcatNGroup(arena_,
                                                {
                                                    arena_.hard_line(),
                                                    arena_.MakeNest(guts),
                                                    arena_.hard_line(),
                                                })),
                  arena_.cparen(),
              });
}

DocRef Formatter::FormatTuple(const XlsTuple& n) {
  Span tuple_span = n.span();
  // Detect if there are any comments in the span of the tuple.
  bool any_comments = comments_.HasComments(tuple_span);

  if (!any_comments) {
    // Do it the old way.
    return FormatTupleWithoutComments(n);
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

    // If there are comments between end of the last element we processed, and
    // the start of this one, prepend them.
    if (std::optional<DocRef> previous_comments = FormatCommentsBetween(
            last_tuple_element_span_limit, span.start(), nullptr)) {
      if (!first_element) {
        pieces.push_back(arena_.comma());
        pieces.push_back(arena_.space());
      }
      // TODO: davidplass - if the previous comment is not on the same line as
      // the previous element, insert a hard line before the comment.
      pieces.push_back(previous_comments.value());
      pieces.push_back(arena_.hard_line());
    } else if (!first_element) {
      // No comments between there and here; append a newline to "terminate" the
      // previous element.
      pieces.push_back(arena_.comma());
      pieces.push_back(arena_.hard_line());
    }

    last_tuple_element_span_limit = span.limit();
    // Format the element itself.
    pieces.push_back(Format(item));
  }

  DocRef guts = ConcatN(arena_, pieces);

  // Append comments between the last element and the end of the tuple
  if (std::optional<DocRef> terminal_comment = FormatCommentsBetween(
          last_tuple_element_span_limit, tuple_span.limit(), nullptr)) {
    // Add trailing comma before the terminal comment too.
    guts = ConcatN(arena_, {guts, arena_.comma(), arena_.space(),
                            terminal_comment.value()});
  } else if (n.members().size() == 1) {
    // No trailing comment, but add a comma if it's a singleton.
    guts = ConcatN(arena_, {guts, arena_.comma()});
  }

  return ConcatN(arena_, {
                             arena_.oparen(),
                             arena_.hard_line(),
                             arena_.MakeNest(guts),
                             arena_.hard_line(),
                             arena_.cparen(),
                         });
}

// Note: this does not put any spacing characters after the '{' so we can
// appropriately handle the case of an empty struct having no spacing in its
// `S {}` style construct.
DocRef Formatter::FormatStructLeader(const TypeAnnotation* struct_ref) {
  return ConcatNGroup(arena_, {
                                  FormatTypeAnnotation(*struct_ref),
                                  arena_.break1(),
                                  arena_.ocurl(),
                              });
}

DocRef Formatter::FormatStructMembersFlat(
    absl::Span<const std::pair<std::string, Expr*>> members) {
  return FormatJoin<std::pair<std::string, Expr*>>(
      members, Joiner::kCommaSpace, [this](const auto& member) {
        const auto& [name, expr] = member;
        // If the expression is an identifier that matches its corresponding
        // struct member name, we canonically use the shorthand notation of just
        // providing the identifier and leaving the member name implicitly as
        // the same symbol.
        if (const NameRef* name_ref = dynamic_cast<const NameRef*>(expr);
            name_ref != nullptr && name_ref->identifier() == name) {
          return arena_.MakeText(name);
        }

        return ConcatN(arena_, {arena_.MakeText(name), arena_.colon(),
                                arena_.space(), FormatExpr(*expr)});
      });
}

DocRef Formatter::FormatStructMembersBreak(
    Span struct_span, absl::Span<const std::pair<std::string, Expr*>> members) {
  std::vector<DocRef> pieces;
  Pos previous_item_limit = struct_span.start();
  for (size_t i = 0; i < members.size(); ++i) {
    const std::pair<std::string, Expr*>& member = members[i];
    const auto& [field_name, expr] = member;

    // If there are comments between the last item and here, insert them.
    Span comment_span(previous_item_limit, expr->span().start());
    if (std::optional<DocRef> previous_comments = FormatCommentsBetween(
            comment_span.start(), comment_span.limit(), nullptr)) {
      pieces.push_back(previous_comments.value());
      pieces.push_back(arena_.hard_line());
    }
    previous_item_limit = expr->span().limit();

    // If the expression is an identifier that matches its corresponding
    // struct member name, we canonically use the shorthand notation of just
    // providing the identifier and leaving the member name implicitly as
    // the same symbol.
    DocRef member_doc;
    if (const NameRef* name_ref = dynamic_cast<const NameRef*>(expr);
        name_ref != nullptr && name_ref->identifier() == field_name) {
      member_doc = arena_.MakeText(field_name);
    } else {
      // First we format the member into a doc and then decide the best way to
      // put it into the sequence.
      DocRef field_expr = FormatExpr(*expr);

      // This is the document we want to emit both when we:
      // - Know it fits in flat mode
      // - Know the start of the document (i.e. the leader on the RHS
      //   expression) can be emitted in flat mode
      //
      // That's why it has a `break1` in it (instead of a space) and a
      // reassessment of whether to enter break mode for the field
      // expression.
      DocRef on_flat =
          ConcatN(arena_, {arena_.MakeText(field_name), arena_.colon(),
                           arena_.break1(), arena_.MakeGroup(field_expr)});
      DocRef nest_field_expr =
          ConcatN(arena_, {arena_.MakeText(field_name), arena_.colon(),
                           arena_.hard_line(), arena_.MakeNest(field_expr)});

      DocRef on_other;
      if (expr->IsBlockedExprWithLeader()) {
        DocRef leader =
            ConcatN(arena_, {arena_.MakeText(field_name), arena_.colon(),
                             arena_.space(), FormatBlockedExprLeader(*expr)});
        on_other = arena_.MakeModeSelect(leader, /*on_flat=*/on_flat,
                                         /*on_break=*/nest_field_expr);
      } else {
        on_other = arena_.MakeFlatChoice(on_flat, nest_field_expr);
      }

      member_doc = arena_.MakeNestIfFlatFits(on_flat, on_other);
    }

    pieces.push_back(member_doc);
    pieces.push_back(arena_.comma());
    // TODO: https://github.com/google/xls/issues/1719 - if there is a comment
    // on the same line as this item, insert the comment first.
    if (i + 1 != members.size()) {
      // Not the last item, add a hardline
      pieces.push_back(arena_.hard_line());
    }
  }

  // Insert comments between the last item and the end of the struct.
  Span comment_span(previous_item_limit, struct_span.limit());
  if (std::optional<DocRef> previous_comments = FormatCommentsBetween(
          comment_span.start(), comment_span.limit(), nullptr)) {
    pieces.push_back(arena_.hard_line());
    pieces.push_back(previous_comments.value());
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatFlatRest(const StructInstance& n) {
  return ConcatN(
      arena_, {arena_.space(), FormatStructMembersFlat(n.GetUnorderedMembers()),
               arena_.space(), arena_.ccurl()});
}

DocRef Formatter::FormatBreakRest(const StructInstance& n) {
  return ConcatN(arena_, {arena_.hard_line(),
                          arena_.MakeNest(FormatStructMembersBreak(
                              n.span(), n.GetUnorderedMembers())),
                          arena_.hard_line(), arena_.ccurl()});
}

DocRef Formatter::FormatStructInstance(const StructInstance& n) {
  DocRef leader = FormatStructLeader(n.struct_ref());

  if (n.GetUnorderedMembers().empty()) {  // empty struct instance
    return arena_.MakeConcat(leader, arena_.ccurl());
  }

  // Implementation note: we cannot reorder members to be canonically the same
  // order as the struct definition in the general case, since the struct
  // definition may be defined an an imported file, and we have auto-formatting
  // work purely at the single-file syntax level.

  // If there are comments within the span, we must go to break mode, because
  // newlines.
  DocRef on_break = FormatBreakRest(n);
  if (comments_.HasComments(n.span())) {
    return arena_.MakeConcat(leader, on_break);
  }

  DocRef on_flat = FormatFlatRest(n);
  return arena_.MakeConcat(
      leader, arena_.MakeGroup(arena_.MakeFlatChoice(on_flat, on_break)));
}

DocRef Formatter::FormatSplatStructInstance(const SplatStructInstance& n) {
  DocRef leader = FormatStructLeader(n.struct_ref());
  DocRef splatted = Format(n.splatted());
  if (n.members().empty()) {
    return ConcatNGroup(arena_, {leader, arena_.break1(), arena_.dot_dot(),
                                 splatted, arena_.break1(), arena_.ccurl()});
  }

  DocRef on_flat =
      ConcatN(arena_, {arena_.space(), FormatStructMembersFlat(n.members()),
                       arena_.comma(), arena_.space(), arena_.dot_dot(),
                       splatted, arena_.space(), arena_.ccurl()});
  DocRef on_break = ConcatN(
      arena_, {arena_.hard_line(),
               arena_.MakeNest(ConcatN(
                   arena_, {FormatStructMembersBreak(n.span(), n.members()),
                            arena_.hard_line(), arena_.dot_dot(), splatted})),
               arena_.hard_line(), arena_.ccurl()});
  return arena_.MakeConcat(
      leader, arena_.MakeGroup(arena_.MakeFlatChoice(on_flat, on_break)));
}

DocRef Formatter::FormatString(const String& n) {
  return arena_.MakeText(n.ToString());
}

// Creates a doc that has the "test portion" of the conditional; i.e.
//
//  if <break1> $test_expr <break1> {
DocRef Formatter::FormatMakeConditionalTest(const Conditional& n) {
  std::vector<DocRef> pieces;
  if (n.IsConst() && !n.IsElseIf()) {
    pieces.push_back(arena_.Make(Keyword::kConst));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(Keyword::kIf));
  pieces.push_back(arena_.space());
  pieces.push_back(FormatExpr(*n.test(), /*suppress_parens=*/true));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.ocurl());

  return ConcatNGroup(arena_, pieces);
}

// When there's an else-if, or multiple statements inside of the blocks, we
// force the formatting to be multi-line.
DocRef Formatter::FormatConditionalMultiline(const Conditional& n) {
  std::vector<DocRef> pieces = {
      FormatMakeConditionalTest(n), arena_.hard_line(),
      FormatBlock(*n.consequent(), FormatBlockOptions{.add_curls = false}),
      arena_.hard_line()};

  bool has_else = n.HasElse();
  std::variant<StatementBlock*, Conditional*> alternate = n.alternate();
  while (std::holds_alternative<Conditional*>(alternate)) {
    Conditional* elseif = std::get<Conditional*>(alternate);
    alternate = elseif->alternate();
    has_else = elseif->HasElse();
    pieces.push_back(arena_.ccurl());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.Make(Keyword::kElse));
    pieces.push_back(arena_.space());
    pieces.push_back(FormatMakeConditionalTest(*elseif));
    pieces.push_back(arena_.hard_line());
    pieces.push_back(FormatBlock(*elseif->consequent(),
                                 FormatBlockOptions{.add_curls = false}));
    pieces.push_back(arena_.hard_line());
  }

  if (has_else) {
    CHECK(std::holds_alternative<StatementBlock*>(alternate));
    StatementBlock* else_block = std::get<StatementBlock*>(alternate);
    pieces.push_back(arena_.ccurl());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.Make(Keyword::kElse));
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.ocurl());
    pieces.push_back(arena_.hard_line());
    pieces.push_back(
        FormatBlock(*else_block, FormatBlockOptions{.add_curls = false}));
    pieces.push_back(arena_.hard_line());
  }
  pieces.push_back(arena_.ccurl());

  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatConditional(const Conditional& n) {
  // If there's an else-if clause or multi-statement blocks we force it to be
  // multi-line.
  if (n.HasElseIf() || n.HasMultiStatementBlocks()) {
    return FormatConditionalMultiline(n);
  }

  DocRef test = FormatMakeConditionalTest(n);
  std::vector<DocRef> pieces = {
      test,
      arena_.break1(),
      FormatBlock(*n.consequent(), FormatBlockOptions{.add_curls = false}),
      arena_.break1(),
  };

  if (n.HasElse()) {
    CHECK(std::holds_alternative<StatementBlock*>(n.alternate()));
    const StatementBlock* else_block = std::get<StatementBlock*>(n.alternate());
    pieces.push_back(arena_.ccurl());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.Make(Keyword::kElse));
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.ocurl());
    pieces.push_back(arena_.break1());
    pieces.push_back(
        FormatBlock(*else_block, FormatBlockOptions{.add_curls = false}));
    pieces.push_back(arena_.break1());
  }

  pieces.push_back(arena_.ccurl());
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatTupleIndex(const TupleIndex& n) {
  std::vector<DocRef> pieces;
  if (WeakerThan(n.lhs()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena_.oparen());
    pieces.push_back(FormatExpr(*n.lhs()));
    pieces.push_back(arena_.cparen());
  } else {
    pieces.push_back(FormatExpr(*n.lhs()));
  }

  pieces.push_back(arena_.dot());
  pieces.push_back(FormatNumber(*n.index()));
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatZeroMacro(const ZeroMacro& n) {
  return ConcatNGroup(arena_, {
                                  arena_.MakeText("zero!"),
                                  arena_.oangle(),
                                  FormatExprOrType(n.type()),
                                  arena_.cangle(),
                                  arena_.oparen(),
                                  arena_.cparen(),
                              });
}

DocRef Formatter::FormatAllOnesMacro(const AllOnesMacro& n) {
  return ConcatNGroup(arena_, {
                                  arena_.MakeText("all_ones!"),
                                  arena_.oangle(),
                                  FormatExprOrType(n.type()),
                                  arena_.cangle(),
                                  arena_.oparen(),
                                  arena_.cparen(),
                              });
}

DocRef Formatter::FormatUnop(const Unop& n) {
  std::vector<DocRef> pieces = {arena_.MakeText(UnopKindFormat(n.unop_kind()))};
  if (WeakerThan(n.operand()->GetPrecedence(), n.GetPrecedence())) {
    pieces.push_back(arena_.oparen());
    pieces.push_back(FormatExpr(*n.operand()));
    pieces.push_back(arena_.cparen());
  } else {
    pieces.push_back(FormatExpr(*n.operand()));
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatRange(const Range& n) {
  if (n.inclusive_end()) {
    return ConcatNGroup(
        arena_, {FormatExpr(*n.start()), arena_.break0(), arena_.dot_dot(),
                 arena_.equals(), FormatExpr(*n.end())});
  }
  return ConcatNGroup(
      arena_, {FormatExpr(*n.start()), arena_.break0(), arena_.dot_dot(),
               arena_.break0(), FormatExpr(*n.end())});
}

DocRef Formatter::FormatNameDefTreeLeaf(const NameDefTree::Leaf& n) {
  return absl::visit(
      Visitor{
          [&](const NameDef* n) { return FormatNameDef(*n); },
          [&](const NameRef* n) { return FormatNameRef(*n); },
          [&](const WildcardPattern* n) { return FormatWildcardPattern(*n); },
          [&](const RestOfTuple* n) { return FormatRestOfTuple(*n); },
          [&](const Number* n) { return FormatNumber(*n); },
          [&](const ColonRef* n) { return FormatColonRef(*n); },
          [&](const Range* n) { return FormatRange(*n); },
      },
      n);
}

DocRef Formatter::FormatNameDefTree(const NameDefTree& n) {
  if (n.is_leaf()) {
    return FormatNameDefTreeLeaf(n.leaf());
  }
  std::vector<DocRef> pieces = {arena_.oparen()};
  std::vector<std::variant<NameDefTree::Leaf, NameDefTree*>> flattened =
      n.Flatten1();
  for (size_t i = 0; i < flattened.size(); ++i) {
    const auto& item = flattened[i];
    absl::visit(Visitor{
                    [&](const NameDefTree::Leaf& leaf) {
                      pieces.push_back(FormatNameDefTreeLeaf(leaf));
                    },
                    [&](const NameDefTree* subtree) {
                      pieces.push_back(FormatNameDefTree(*subtree));
                    },
                },
                item);
    if (i + 1 != flattened.size()) {
      pieces.push_back(arena_.comma());
      pieces.push_back(arena_.break1());
    }
  }
  pieces.push_back(arena_.cparen());
  return ConcatNGroup(arena_, pieces);
}

class FmtExprVisitor : public ExprVisitor {
 public:
  explicit FmtExprVisitor(Formatter& formatter) : formatter_(formatter) {}

  ~FmtExprVisitor() override = default;

#define DEFINE_HANDLER(__type)                               \
  absl::Status Handle##__type(const __type* expr) override { \
    result_ = formatter_.Format##__type(*expr);              \
    return absl::OkStatus();                                 \
  }

  XLS_DSLX_EXPR_NODE_EACH(DEFINE_HANDLER)

#undef DEFINE_HANDLER

  DocRef result() const { return result_.value(); }

 private:
  Formatter& formatter_;
  std::optional<DocRef> result_;
};

// Note: suppress_parens just suppresses parentheses for the outermost
// expression "n", not transitively.
DocRef Formatter::FormatExpr(const Expr& n, bool suppress_parens) {
  FmtExprVisitor v(*this);
  CHECK_OK(n.AcceptExpr(&v));
  DocRef result = v.result();
  if (n.in_parens() && !suppress_parens) {
    return ConcatNGroup(arena_, {arena_.oparen(), result, arena_.cparen()});
  }
  return result;
}

// Creates a document for the "leader" of the given expression.
//
// Precondition: `e` must be a blocked expression with a leader component; e.g.
// invocation (leader is callee), conditional (leader is test), etc.
DocRef Formatter::FormatBlockedExprLeader(const Expr& e) {
  CHECK(e.IsBlockedExprWithLeader());
  switch (e.kind()) {
    case AstNodeKind::kInvocation: {
      return arena_.MakeConcat(
          Format(static_cast<const Invocation&>(e).callee()), arena_.oparen());
    }
    case AstNodeKind::kConditional: {
      const Expr& test = *static_cast<const Conditional&>(e).test();
      return ConcatN(
          arena_, {arena_.Make(Keyword::kIf), arena_.space(), FormatExpr(test),
                   arena_.space(), arena_.ocurl()});
    }
    case AstNodeKind::kMatch: {
      const Expr& test = *static_cast<const Match&>(e).matched();
      return ConcatN(arena_,
                     {arena_.Make(Keyword::kMatch), arena_.space(),
                      FormatExpr(test), arena_.space(), arena_.ocurl()});
    }
    case AstNodeKind::kArray: {
      const TypeAnnotation& type =
          *static_cast<const Array&>(e).type_annotation();
      return ConcatN(arena_, {FormatTypeAnnotation(type), arena_.colon(),
                              arena_.obracket()});
    }
    case AstNodeKind::kStructInstance: {
      const StructInstance& n = static_cast<const StructInstance&>(e);
      return ConcatN(arena_, {FormatStructLeader(n.struct_ref()),
                              arena_.space(), arena_.ocurl()});
    }
    case AstNodeKind::kSplatStructInstance: {
      const SplatStructInstance& n = static_cast<const SplatStructInstance&>(e);
      return ConcatN(arena_, {FormatStructLeader(n.struct_ref()),
                              arena_.space(), arena_.ocurl()});
    }
    case AstNodeKind::kFor: {
      const ForLoopBase& n = static_cast<const ForLoopBase&>(e);
      DocRef names_ref = FormatNameDefTree(*n.names());
      return FormatForLoopBaseLeader(Keyword::kFor, names_ref, n,
                                     /*is_const_for=*/false);
    }
    case AstNodeKind::kConstFor: {
      const ConstFor& n = static_cast<const ConstFor&>(e);
      Keyword keyword = n.IsUnrollFor() ? Keyword::kUnrollFor : Keyword::kFor;
      DocRef names_ref = FormatNameDefTree(*n.names());
      return FormatForLoopBaseLeader(keyword, names_ref, n, !n.IsUnrollFor());
    }
    default:
      LOG(FATAL) << "Unhandled node kind for FmtBlockedExprLeader: `"
                 << e.ToString() << "` @ "
                 << e.span().ToString(arena_.file_table());
  }
}

DocRef Formatter::FormatConstantDef(const ConstantDef& n) {
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
    std::optional<DocRef> comments_doc = FormatCommentsBetween(
        n.name_def()->GetSpan()->limit(),
        n.type_annotation()->GetSpan()->start(), /*last_comment_span=*/nullptr);
    if (comments_doc.has_value()) {
      dest->push_back(ConcatN(
          arena_, {arena_.break1(), *comments_doc, arena_.hard_line()}));
      // From now on we need to nest.
      nest_rhs = true;
      dest = &rhs_pieces;
    }

    dest->push_back(arena_.colon());
    dest->push_back(arena_.break1());
    dest->push_back(FormatTypeAnnotation(*n.type_annotation()));

    // Find comments between the end of the type annotation and the start of the
    // value and put them between the type and the =
    comments_doc = FormatCommentsBetween(
        n.type_annotation()->GetSpan()->limit(), n.value()->GetSpan()->start(),
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
    std::optional<DocRef> comments_doc = FormatCommentsBetween(
        n.name_def()->GetSpan()->limit(), n.value()->GetSpan()->start(),
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

  DocRef value_doc = FormatExpr(*n.value());
  std::optional<DocRef> comments_doc =
      FormatCommentsBetween(n.value()->GetSpan()->limit(), n.span().limit(),
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
  comments_doc = FormatCommentsBetween(lhs_comments_span.start(),
                                       lhs_comments_span.limit(),
                                       /*last_comment_span=*/nullptr);
  if (comments_doc.has_value()) {
    pre_comment = ConcatN(arena_, {*comments_doc, arena_.hard_line()});
  }

  return ConcatN(arena_, {pre_comment, lhs, rhs});
}

DocRef Formatter::FormatConstAssert(const ConstAssert& n) {
  DocRef arg_doc = FormatExpr(*n.arg());

  // Helper lambda for the case where we have a blocked expression with a
  // "leader" doc.
  auto make_blocked = [&]() {
    DocRef leader = FormatBlockedExprLeader(*n.arg());
    DocRef nested =
        arena_.MakeNest(arena_.MakeConcat(arena_.hard_line(), arg_doc));
    // If the leader doc fits, we emit the arg doc directly with the leader
    // starting emission in flat mode; otherwise we emit the nested version and
    // we start in break mode.
    return arena_.MakeModeSelect(leader, /*on_flat=*/arg_doc,
                                 /*on_break=*/nested);
  };

  DocRef arg_with_nest = arena_.MakeNestIfFlatFits(
      /*on_nested_flat_ref=*/arg_doc,
      /*on_other_ref=*/n.arg()->IsBlockedExprWithLeader()
          ? make_blocked()
          : arena_.MakeAlign(arg_doc));
  return ConcatNGroup(arena_, {
                                  arena_.MakeText("const_assert!"),
                                  arena_.oparen(),
                                  arg_with_nest,
                                  arena_.cparen(),
                              });
}

DocRef Formatter::FormatVerbatimNode(const VerbatimNode& n) {
  if (n.text().empty()) {
    return arena_.empty();
  }
  return arena_.MakeZeroIndent(arena_.MakeText(std::string(n.text())));
}

DocRef Formatter::FormatStatement(const Statement& n, bool trailing_semi) {
  auto maybe_concat_semi = [&](DocRef d) {
    if (trailing_semi) {
      return arena_.MakeConcat(d, arena_.semi());
    }
    return d;
  };
  return absl::visit(
      Visitor{
          [&](const VerbatimNode* n) { return FormatVerbatimNode(*n); },
          [&](const Expr* n) { return maybe_concat_semi(FormatExpr(*n)); },
          [&](const TypeAlias* n) {
            return maybe_concat_semi(FormatTypeAlias(*n));
          },
          [&](const Let* n) { return FormatLet(*n, trailing_semi); },
          [&](const ConstAssert* n) {
            return maybe_concat_semi(FormatConstAssert(*n));
          },
      },
      n.wrapped());
}

// Formats parameters (i.e. function parameters) with leading '(' and trailing
// ')'.
DocRef Formatter::FormatParams(absl::Span<const Param* const> params) {
  DocRef guts = FormatJoin<const Param*>(
      params, Joiner::kCommaBreak1AsGroupNoTrailingComma,
      [this](const Param* param) {
        DocRef id = arena_.MakeText(param->identifier());
        if (auto* st =
                dynamic_cast<SelfTypeAnnotation*>(param->type_annotation());
            st != nullptr && !st->explicit_type()) {
          return id;
        }
        DocRef type = FormatTypeAnnotation(*param->type_annotation());
        return ConcatN(arena_, {id, arena_.colon(), arena_.space(), type});
      });

  return ConcatNGroup(
      arena_, {arena_.oparen(),
               arena_.MakeAlign(arena_.MakeConcat(guts, arena_.cparen()))});
}

DocRef Formatter::FormatParametricBinding(const ParametricBinding& n) {
  std::vector<DocRef> pieces = {
      arena_.MakeText(n.identifier()),
      arena_.colon(),
      arena_.break1(),
      FormatTypeAnnotation(*n.type_annotation()),
  };
  if (n.expr() != nullptr) {
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.equals());
    pieces.push_back(arena_.space());
    pieces.push_back(arena_.ocurl());
    pieces.push_back(arena_.break0());
    pieces.push_back(arena_.MakeNest(FormatExpr(*n.expr())));
    pieces.push_back(arena_.ccurl());
  }
  return ConcatNGroup(arena_, pieces);
}

DocRef Formatter::FormatParametricBindingPtr(const ParametricBinding* n) {
  CHECK(n != nullptr);
  return FormatParametricBinding(*n);
}

DocRef Formatter::FormatFunction(const Function& n, bool is_test) {
  std::vector<DocRef> signature_pieces;

  // Note that the functions in a trait are implicitly public.
  if (n.is_public() && (!n.IsStub() || n.parent() == nullptr ||
                        n.parent()->kind() != AstNodeKind::kTrait)) {
    signature_pieces.push_back(arena_.Make(Keyword::kPub));
    signature_pieces.push_back(arena_.space());
  }
  signature_pieces.push_back(arena_.Make(Keyword::kFn));
  signature_pieces.push_back(arena_.space());
  signature_pieces.push_back(arena_.MakeText(n.identifier()));

  if (n.IsParametric()) {
    DocRef flat_parametrics =
        ConcatNGroup(arena_, {arena_.oangle(),
                              FormatJoin<const ParametricBinding*>(
                                  n.parametric_bindings(), Joiner::kCommaSpace,
                                  [&](const ParametricBinding* n) {
                                    return FormatParametricBindingPtr(n);
                                  }),
                              arena_.cangle()});

    DocRef parametric_guts =
        ConcatN(arena_, {arena_.oangle(),
                         arena_.MakeAlign(FormatJoin<const ParametricBinding*>(
                             n.parametric_bindings(),
                             Joiner::kCommaBreak1AsGroupNoTrailingComma,
                             [&](const ParametricBinding* n) {
                               return FormatParametricBindingPtr(n);
                             })),
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
      if (n.IsStub()) {
        params_pieces.push_back(arena_.semi());
      } else {
        params_pieces.push_back(arena_.break1());
        params_pieces.push_back(arena_.ocurl());
      }
    } else {
      std::vector<DocRef> return_pieces;
      return_pieces.push_back(arena_.break1());
      return_pieces.push_back(arena_.arrow());
      return_pieces.push_back(arena_.space());
      return_pieces.push_back(FormatTypeAnnotation(*n.return_type()));
      if (n.IsStub()) {
        return_pieces.push_back(arena_.semi());
      } else {
        return_pieces.push_back(arena_.space());
        return_pieces.push_back(arena_.ocurl());
      }
      params_pieces.push_back(ConcatNGroup(arena_, return_pieces));
    }

    signature_pieces.push_back(
        arena_.MakeNest(ConcatNGroup(arena_, params_pieces)));
  }

  std::vector<DocRef> fn_pieces;
  std::optional<const Attribute*> fuzz_test_attr =
      GetAttribute(&n, AttributeKind::kFuzzTest);
  if (fuzz_test_attr.has_value()) {
    fn_pieces.push_back(FormatAttribute(**fuzz_test_attr));
  }

  if (n.extern_verilog_module().has_value()) {
    auto code_template = (*n.extern_verilog_module()).code_template();
    fn_pieces.push_back(
        ConcatN(arena_, {
                            arena_.MakeText("#[extern_verilog(\""),
                            arena_.MakeText(code_template),
                            arena_.MakeText("\")]"),
                            arena_.hard_line(),
                        }));
  } else if (n.is_test_utility() && !is_test) {
    fn_pieces.push_back(
        ConcatN(arena_, {
                            arena_.MakeText("#"),
                            arena_.obracket(),
                            arena_.MakeText(std::string(kCfgTestAttr)),
                            arena_.cbracket(),
                            arena_.hard_line(),
                        }));
  }

  bool force_multiline = false;
  if (n.impl().has_value() &&
      (n.identifier() == "new" || n.identifier() == "next")) {
    std::optional<const StructDefBase*> target = n.GetTargetStruct();
    if (target.has_value() && (*target)->kind() == AstNodeKind::kProcDef) {
      force_multiline = true;
    }
  }

  fn_pieces.push_back(ConcatNGroup(arena_, signature_pieces));
  if (!n.IsStub()) {
    if (n.body()->empty()) {
      // For empty function we don't put spaces between the curls.
      fn_pieces.push_back(FormatBlock(
          *n.body(), FormatBlockOptions{.add_curls = false,
                                        .force_multiline = force_multiline}));
      fn_pieces.push_back(arena_.ccurl());
    } else {
      // For non-empty functions, we break after the signature and before
      // the ccurl.
      fn_pieces.push_back(force_multiline ? arena_.hard_line()
                                          : arena_.break1());
      fn_pieces.push_back(FormatBlock(
          *n.body(), FormatBlockOptions{.add_curls = false,
                                        .force_multiline = force_multiline}));
      fn_pieces.push_back(force_multiline ? arena_.hard_line()
                                          : arena_.break1());
      fn_pieces.push_back(arena_.ccurl());
    }
  }

  return ConcatNGroup(arena_, fn_pieces);
}

DocRef Formatter::FormatProcMember(const ProcMember& n) {
  return ConcatNGroup(
      arena_, {FormatNameDef(*n.name_def()), arena_.colon(), arena_.break1(),
               FormatTypeAnnotation(*n.type_annotation())});
}

DocRef Formatter::FormatProc(const Proc& n, bool is_test) {
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
    signature_pieces.push_back(
        ConcatNGroup(arena_, {arena_.oangle(),
                              FormatJoin<const ParametricBinding*>(
                                  n.parametric_bindings(), Joiner::kCommaSpace,
                                  [&](const ParametricBinding* n) {
                                    return FormatParametricBindingPtr(n);
                                  }),
                              arena_.cangle()}));
  }
  signature_pieces.push_back(arena_.break1());
  signature_pieces.push_back(arena_.ocurl());

  Pos last_stmt_limit = n.body_span().start();

  // We update this with the position that's relevant for config start comments.
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
              if (std::optional<DocRef> maybe_doc = FormatCommentsBetween(
                      last_stmt_limit, n->span().start(), nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(FormatProcMember(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const ConstantDef* n) {
              if (std::optional<DocRef> maybe_doc = FormatCommentsBetween(
                      last_stmt_limit, n->span().start(), nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(FormatConstantDef(*n));
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const TypeAlias* n) {
              if (std::optional<DocRef> maybe_doc = FormatCommentsBetween(
                      last_stmt_limit, n->span().start(), nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(FormatTypeAlias(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
            [&](const ConstAssert* n) {
              if (std::optional<DocRef> maybe_doc = FormatCommentsBetween(
                      last_stmt_limit, n->span().start(), nullptr)) {
                stmt_pieces.push_back(
                    arena_.MakeConcat(maybe_doc.value(), arena_.hard_line()));
              }
              stmt_pieces.push_back(FormatConstAssert(*n));
              stmt_pieces.push_back(arena_.semi());
              stmt_pieces.push_back(arena_.hard_line());
              last_stmt_limit = n->span().limit();
            },
        },
        stmt);
  }

  CHECK(config_comment_start_pos.has_value());
  std::optional<DocRef> config_comment = FormatCommentsBetween(
      config_comment_start_pos, n.config().span().start(), nullptr);
  CHECK(init_comment_start_pos.has_value());
  std::optional<DocRef> init_comment = FormatCommentsBetween(
      init_comment_start_pos, n.init().span().start(), nullptr);
  CHECK(next_comment_start_pos.has_value());
  std::optional<DocRef> next_comment = FormatCommentsBetween(
      next_comment_start_pos, n.next().span().start(), nullptr);
  // Comments between the last statement and the end of the proc.
  std::optional<DocRef> end_comment =
      FormatCommentsBetween(last_stmt_limit, n.body_span().limit(), nullptr);

  std::vector<DocRef> config_pieces = {
      arena_.MakeText("config"),
      FormatParams(n.config().params()),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FormatBlock(*n.config().body(), FormatBlockOptions{.add_curls = false}),
      arena_.break1(),
      arena_.ccurl(),
  };

  std::vector<DocRef> init_pieces = {
      arena_.MakeText("init"),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FormatBlock(*n.init().body(), FormatBlockOptions{.add_curls = false}),
      arena_.break1(),
      arena_.ccurl(),
  };

  std::vector<DocRef> next_pieces = {
      arena_.MakeText("next"),
      FormatParams(n.next().params()),
      arena_.space(),
      arena_.ocurl(),
      arena_.break1(),
      FormatBlock(*n.next().body(), FormatBlockOptions{.add_curls = false}),
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
      ConcatNGroup(arena_, attribute_pieces),
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

DocRef Formatter::FormatTestFunction(const TestFunction& n) {
  std::vector<DocRef> pieces;
  pieces.push_back(arena_.MakeText("#[test]"));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(FormatFunction(n.fn(), /*is_test=*/true));
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatTestProc(const TestProc& n) {
  std::vector<DocRef> pieces;
  if (n.expected_fail_label().has_value()) {
    pieces.push_back(arena_.MakeText(
        absl::StrFormat("#[test_proc(expected_fail_label=\"%s\")]",
                        n.expected_fail_label().value())));
  } else {
    pieces.push_back(arena_.MakeText("#[test_proc]"));
  }
  pieces.push_back(arena_.hard_line());
  pieces.push_back(FormatProc(*n.proc(), /*is_test=*/true));
  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatQuickCheck(const QuickCheck& n) {
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
  pieces.push_back(FormatFunction(*n.fn()));
  return ConcatN(arena_, pieces);
}

void Formatter::FormatStructMembers(const StructDefBase& n,
                                    bool force_multiline,
                                    std::vector<DocRef>& pieces) {
  if (!n.members().empty()) {
    pieces.push_back(force_multiline ? arena_.hard_line() : arena_.break1());
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
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_member_pos, member_start, &last_comment_span)) {
      body_pieces.push_back(comments_doc.value());
      body_pieces.push_back(arena_.hard_line());

      // If the comment abuts the member we don't put a newline in
      // between, we assume the comment is associated with the member.
      if (last_comment_span->limit().lineno() != member_start.lineno()) {
        body_pieces.push_back(arena_.hard_line());
      }
    }

    last_member_pos = member_span.limit();

    body_pieces.push_back(arena_.MakeText(member->name()));
    body_pieces.push_back(arena_.colon());
    body_pieces.push_back(arena_.space());
    body_pieces.push_back(FormatTypeAnnotation(*member->type()));
    bool last_member = i + 1 == n.members().size();
    if (last_member) {
      body_pieces.push_back(arena_.MakeFlatChoice(/*on_flat=*/arena_.empty(),
                                                  /*on_break=*/arena_.comma()));
    } else {
      body_pieces.push_back(arena_.comma());
    }

    // See if there are inline comments after the member.
    Pos new_last_member_pos = FormatCollectInlineComments(
        member_span.limit(), last_member_pos, body_pieces, last_comment_span);

    bool had_inline = new_last_member_pos != last_member_pos;
    if (!last_member) {
      body_pieces.push_back(had_inline || force_multiline ? arena_.hard_line()
                                                          : arena_.break1());
    }
    last_member_pos = new_last_member_pos;
  }

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  bool emitted_trailing_comment = false;
  if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
          last_member_pos, n.span().limit(), &last_comment_span)) {
    body_pieces.push_back(arena_.hard_line());
    body_pieces.push_back(comments_doc.value());
    emitted_trailing_comment = true;
  }

  pieces.push_back(arena_.MakeNest(ConcatN(arena_, body_pieces)));

  if (!n.members().empty() || emitted_trailing_comment) {
    pieces.push_back(force_multiline ? arena_.hard_line() : arena_.break1());
  }
}

DocRef Formatter::FormatStructDefBase(
    const StructDefBase& n, Keyword keyword,
    const std::optional<std::string>& extern_type_name) {
  std::vector<DocRef> pieces;
  std::vector<DocRef> attrs;
  for (const Attribute* attribute : n.attributes()) {
    attrs.push_back(FormatAttribute(*attribute));
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
    pieces.push_back(FormatJoin<const ParametricBinding*>(
        n.parametric_bindings(), Joiner::kCommaSpace,
        [&](const ParametricBinding* n) {
          return FormatParametricBindingPtr(n);
        }));
    pieces.push_back(arena_.cangle());
  }

  pieces.push_back(arena_.space());
  pieces.push_back(arena_.ocurl());

  FormatStructMembers(n, /*force_multiline=*/keyword == Keyword::kProc, pieces);

  pieces.push_back(arena_.ccurl());
  return FormatJoinWithAttrs(attrs, ConcatNGroup(arena_, pieces));
}

DocRef Formatter::FormatStructDef(const StructDef& n) {
  return FormatStructDefBase(n, Keyword::kStruct, n.extern_type_name());
}

DocRef Formatter::FormatProcDef(const ProcDef& n) {
  return FormatStructDefBase(n, Keyword::kProc,
                             /*extern_type_name=*/std::nullopt);
}

DocRef Formatter::FormatImplMember(const ImplMember& n) {
  return absl::visit(
      Visitor{
          [&](const Function* n) { return FormatFunction(*n); },
          [&](const ConstantDef* n) { return FormatConstantDef(*n); },
          [&](const TypeAlias* n) {
            return arena_.MakeConcat(FormatTypeAlias(*n), arena_.semi());
          },
          [&](const VerbatimNode* n) { return FormatVerbatimNode(*n); },
      },
      n);
}

template <typename T>
DocRef Formatter::FormatImplOrTrait(const T& n, Keyword keyword,
                                    DocRef name_or_struct_ref) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(keyword));
  pieces.push_back(arena_.space());
  pieces.push_back(name_or_struct_ref);
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

    // See if there are comments between the last member and the start of this
    // member.
    Span member_span = ToAstNode(member)->GetSpan().value();
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_member_pos, member_span.start(), &last_comment_span)) {
      body_pieces.push_back(comments_doc.value());
      body_pieces.push_back(arena_.hard_line());
    }
    body_pieces.push_back(FormatImplMember(member));
    last_member_pos = member_span.limit();

    last_member_pos = FormatCollectInlineComments(
        member_span.limit(), last_member_pos, body_pieces, last_comment_span);

    body_pieces.push_back(arena_.hard_line());
  }
  if (!body_pieces.empty()) {
    // Remove last line break.
    body_pieces.pop_back();
    pieces.push_back(arena_.MakeNest(ConcatN(arena_, body_pieces)));
    pieces.push_back(arena_.hard_line());
  }
  pieces.push_back(arena_.ccurl());

  return FormatJoinWithAttr(attr, ConcatNGroup(arena_, pieces));
}

DocRef Formatter::FormatImpl(const Impl& n) {
  return FormatImplOrTrait(n, Keyword::kImpl,
                           FormatTypeAnnotation(*n.struct_ref()));
}

DocRef Formatter::FormatTrait(const Trait& n) {
  return FormatImplOrTrait(n, Keyword::kTrait, FormatNameDef(*n.name_def()));
}

DocRef Formatter::FormatEnumMember(const EnumMember& n) {
  return ConcatNGroup(
      arena_, {FormatNameDef(*n.name_def), arena_.space(), arena_.equals(),
               arena_.break1(), FormatExpr(*n.value), arena_.comma()});
}

DocRef Formatter::FormatEnumDef(const EnumDef& n) {
  std::vector<DocRef> pieces;
  std::vector<DocRef> attrs;
  for (const Attribute* attribute : n.attributes()) {
    attrs.push_back(FormatAttribute(*attribute));
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
    pieces.push_back(FormatTypeAnnotation(*n.type_annotation()));
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
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_member_pos, member_start, &last_comment_span)) {
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
    nested.push_back(FormatEnumMember(node));
    if (i + 1 != n.values().size()) {
      nested.push_back(arena_.hard_line());
    }
  }

  // See if there are any comments to emit after the last statement to the end
  // of the block.
  std::optional<Span> last_comment_span;
  if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
          last_member_pos, n.span().limit(), &last_comment_span)) {
    nested.push_back(arena_.hard_line());
    nested.push_back(comments_doc.value());
  }

  DocRef nested_ref = ConcatN(arena_, nested);
  pieces.push_back(arena_.MakeNest(nested_ref));
  pieces.push_back(arena_.hard_line());
  pieces.push_back(arena_.ccurl());
  return FormatJoinWithAttrs(attrs, ConcatN(arena_, pieces));
}

DocRef Formatter::FormatImport(const Import& n) {
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

bool Formatter::IsBlockedExprNoLeader(const Expr& e) {
  return e.IsBlockedExprNoLeader();
}

bool Formatter::IsBlockedExprWithLeader(const Expr& e) {
  return e.IsBlockedExprWithLeader();
}

DocRef Formatter::FormatUse(const Use& n) {
  // TODO(cdleary): 2024-12-07 This is just a stopgap, we should add reflow
  // capability.
  return arena_.MakeText(n.ToString());
}

DocRef Formatter::FormatLet(const Let& n, bool trailing_semi) {
  std::vector<DocRef> leader_pieces = {
      arena_.Make(n.is_const() ? Keyword::kConst : Keyword::kLet),
      arena_.space(), FormatNameDefTree(*n.name_def_tree())};
  if (const TypeAnnotation* t = n.type_annotation()) {
    leader_pieces.push_back(arena_.colon());
    leader_pieces.push_back(arena_.space());
    leader_pieces.push_back(FormatTypeAnnotation(*t));
  }

  leader_pieces.push_back(arena_.space());
  leader_pieces.push_back(arena_.equals());

  const DocRef rhs_doc_internal = FormatExpr(*n.rhs());

  DocRef rhs_doc = rhs_doc_internal;
  if (trailing_semi) {
    // Reduce the width by 1 so we know we can emit the semi inline.
    rhs_doc = arena_.MakeConcat(arena_.MakeReduceTextWidth(rhs_doc_internal, 1),
                                arena_.semi());
  }

  DocRef body;
  if (IsBlockedExprNoLeader(*n.rhs()) || IsBlockedExprWithLeader(*n.rhs())) {
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
    if (IsBlockedExprWithLeader(*n.rhs())) {
      // If the leading component fits, then see what we can fit flat from the
      // RHS, we know we can at least fit that.
      //
      // If the leading component does not fit, emit the whole construct nested
      // on the next line.
      DocRef leader =
          arena_.MakeConcat(arena_.space(), FormatBlockedExprLeader(*n.rhs()));
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

DocRef Formatter::FormatTypeAlias(const TypeAlias& n) {
  std::vector<DocRef> pieces;
  std::vector<DocRef> attrs;
  for (const Attribute* attribute : n.attributes()) {
    attrs.push_back(FormatAttribute(*attribute));
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
  pieces.push_back(FormatTypeAnnotation(n.type_annotation()));
  return FormatJoinWithAttrs(attrs, ConcatNGroup(arena_, pieces));
}

DocRef Formatter::FormatProcAlias(const ProcAlias& n) {
  std::vector<DocRef> pieces;
  std::optional<DocRef> attr;
  if (n.is_public()) {
    pieces.push_back(arena_.Make(Keyword::kPub));
    pieces.push_back(arena_.space());
  }
  pieces.push_back(arena_.Make(Keyword::kProc));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.MakeText(n.identifier()));
  pieces.push_back(arena_.space());
  pieces.push_back(arena_.equals());
  pieces.push_back(arena_.break1());

  ProcAlias::Target target = n.target();
  if (std::holds_alternative<NameRef*>(target)) {
    pieces.push_back(FormatNameRef(*std::get<NameRef*>(target)));
  } else {
    pieces.push_back(FormatColonRef(*std::get<ColonRef*>(target)));
  }

  std::optional<DocRef> parametrics_doc =
      FormatExplicitParametrics(absl::MakeConstSpan(n.parametrics()));
  if (parametrics_doc.has_value()) {
    pieces.push_back(*parametrics_doc);
  }

  return FormatJoinWithAttr(attr, ConcatNGroup(arena_, pieces));
}

DocRef Formatter::FormatModuleMember(const ModuleMember& n) {
  return absl::visit(
      Visitor{
          [&](const Function* n) { return FormatFunction(*n); },
          [&](const Proc* n) { return FormatProc(*n); },
          [&](const TestFunction* n) { return FormatTestFunction(*n); },
          // Formatting the function takes care of the attributes so we don't
          // need a special formatting function for FuzzTestFunction.
          [&](const FuzzTestFunction* n) { return FormatFunction(n->fn()); },
          [&](const TestProc* n) { return FormatTestProc(*n); },
          [&](const QuickCheck* n) { return FormatQuickCheck(*n); },
          [&](const TypeAlias* n) {
            return arena_.MakeConcat(FormatTypeAlias(*n), arena_.semi());
          },
          [&](const ProcAlias* n) {
            return arena_.MakeConcat(FormatProcAlias(*n), arena_.semi());
          },
          [&](const StructDef* n) { return FormatStructDef(*n); },
          [&](const ProcDef* n) { return FormatProcDef(*n); },
          [&](const Impl* n) { return FormatImpl(*n); },
          [&](const Trait* n) { return FormatTrait(*n); },
          [&](const ConstantDef* n) { return FormatConstantDef(*n); },
          [&](const EnumDef* n) { return FormatEnumDef(*n); },
          [&](const Import* n) { return FormatImport(*n); },
          [&](const Use* n) { return FormatUse(*n); },
          [&](const ConstAssert* n) {
            return arena_.MakeConcat(FormatConstAssert(*n), arena_.semi());
          },
          [&](const VerbatimNode* n) { return FormatVerbatimNode(*n); },
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

  if (!ToAstNode(below)->attributes().empty()) {
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

absl::StatusOr<DocRef> Formatter::FormatModule(const Module& n) {
  std::vector<DocRef> pieces;

  std::optional<Span> last_comment_span;
  std::optional<Pos> last_entity_pos;
  if (!n.attributes().empty()) {
    if (std::optional<Span> span = n.GetAttributeSpan(); span.has_value()) {
      if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
              std::nullopt, span->limit(), &last_comment_span);
          comments_doc.has_value()) {
        pieces.push_back(comments_doc.value());
        pieces.push_back(arena_.hard_line());
        pieces.push_back(arena_.hard_line());
      }
      last_entity_pos = span->limit();
    }

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
        case ModuleAttribute::kTypeInferenceVersion1:
          pieces.push_back(arena_.MakeText("#![feature(type_inference_v1)]"));
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
        case ModuleAttribute::kChannelAttributes:
          pieces.push_back(arena_.MakeText("#![feature(channel_attributes)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kExplicitStateAccess:
          pieces.push_back(
              arena_.MakeText("#![feature(explicit_state_access)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kGenerics:
          pieces.push_back(arena_.MakeText("#![feature(generics)]"));
          pieces.push_back(arena_.hard_line());
          break;
        case ModuleAttribute::kTraits:
          pieces.push_back(arena_.MakeText("#[feature(traits)]"));
          pieces.push_back(arena_.hard_line());
          break;
      }
    }
    pieces.push_back(arena_.hard_line());
  }

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

    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_entity_pos, member_start, &last_comment_span)) {
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
    pieces.push_back(FormatModuleMember(member));

    // Now we reflect the emission of the member.
    last_entity_pos = member_span->limit();

    // See if there are inline comments after the statement.
    last_entity_pos = FormatCollectInlineComments(
        member_limit, last_entity_pos.value(), pieces, last_comment_span);

    int num_hard_lines = NumHardLinesAfter(node, member, n.top(), i);
    for (int i = 0; i < num_hard_lines; ++i) {
      pieces.push_back(arena_.hard_line());
    }
  }

  if (std::optional<Pos> last_data_limit = comments_.last_data_limit();
      last_data_limit.has_value() && last_entity_pos < last_data_limit) {
    std::optional<Span> last_comment_span;
    if (std::optional<DocRef> comments_doc = FormatCommentsBetween(
            last_entity_pos, last_data_limit.value(), &last_comment_span)) {
      pieces.push_back(comments_doc.value());
      pieces.push_back(arena_.hard_line());
    }
  }

  // Check if there are any comments that were within the span of the module,
  // but not placed.
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
            "deleted by the *this: //%s\nThis is probably due to a bug "
            "(which may not have been reported yet). To complete formatting, "
            "try moving the comment to a different line.",
            comment->span.ToString(arena_.file_table()), comment_text));
      }
    }
  }

  return ConcatN(arena_, pieces);
}

DocRef Formatter::FormatStatementBlock(const StatementBlock& n) {
  return FormatBlock(n, FormatBlockOptions{.add_curls = n.has_braces()});
}
DocRef Formatter::FormatXlsTuple(const XlsTuple& n) { return FormatTuple(n); }

static absl::StatusOr<std::string> AutoFmt(const Module& m,
                                           Formatter& formatter,
                                           int64_t text_width) {
  XLS_ASSIGN_OR_RETURN(DocRef ref, formatter.FormatModule(m));
  return PrettyPrint(formatter.arena(), ref, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Formatter& formatter,
                                    int64_t text_width) {
  XLS_RET_CHECK(m.fs_path().has_value());
  FormatDisabler disabler(vfs, formatter.comments(), *m.fs_path());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> clone,
      CloneModule(m, [&](const AstNode* node, Module*,
                         const absl::flat_hash_map<const AstNode*, AstNode*>&) {
        return disabler(node);
      }));
  return AutoFmt(*clone, formatter, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Formatter& formatter,
                                    std::string contents, int64_t text_width) {
  FormatDisabler disabler(vfs, formatter.comments(), contents);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> clone,
      CloneModule(m, [&](const AstNode* node, Module*,
                         const absl::flat_hash_map<const AstNode*, AstNode*>&) {
        return disabler(node);
      }));
  return AutoFmt(*clone, formatter, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    int64_t text_width) {
  DocArena arena(*m.file_table());
  Formatter formatter(comments, arena);
  return AutoFmt(vfs, m, formatter, text_width);
}

absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    std::string contents, int64_t text_width) {
  DocArena arena(*m.file_table());
  Formatter formatter(comments, arena);
  return AutoFmt(vfs, m, formatter, std::move(contents), text_width);
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
