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

#include "xls/dslx/fmt/pretty_print.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {
namespace {

using pprint_internal::Align;
using pprint_internal::Concat;
using pprint_internal::Doc;
using pprint_internal::FlatChoice;
using pprint_internal::Group;
using pprint_internal::HardLine;
using pprint_internal::InfinityRequirement;
using pprint_internal::Nest;
using pprint_internal::PrefixedReflow;
using pprint_internal::Requirement;

Requirement operator+(Requirement lhs, Requirement rhs) {
  if (std::holds_alternative<int64_t>(lhs) &&
      std::holds_alternative<int64_t>(rhs)) {
    return Requirement{std::get<int64_t>(lhs) + std::get<int64_t>(rhs)};
  }
  return InfinityRequirement();
}

std::string RequirementToString(const Requirement& r) {
  if (std::holds_alternative<int64_t>(r)) {
    return absl::StrFormat("Requirement{%d}", std::get<int64_t>(r));
  }
  return "InfinityRequirement()";
}

enum class Mode : uint8_t {
  kFlat,
  kBreak,
};

struct StackEntry {
  const Doc* doc;
  Mode mode;
  int64_t indent;
};

bool operator>(Requirement lhs, Requirement rhs) {
  return absl::visit(Visitor{
                         [&](int64_t lhs_value) {
                           return std::holds_alternative<int64_t>(rhs) &&
                                  lhs_value > std::get<int64_t>(rhs);
                         },
                         [&](const std::monostate&) { return true; },
                     },
                     lhs);
}

void PrettyPrintInternal(const DocArena& arena, const Doc& doc,
                         int64_t text_width, std::vector<std::string>& pieces) {
  // We maintain a stack to keep track of doc emission we still need to perform.
  // Every entry notes the document to emit, what mode it was in (flat or
  // line-breaking mode) and what indent level it was supposed to be emitted at.
  std::vector<StackEntry> stack = {StackEntry{&doc, Mode::kFlat, 0}};

  // Number of columns we've output in the current line. (This is reset by hard
  // line breaks.)
  int64_t outcol = 0;

  while (!stack.empty()) {
    StackEntry entry = stack.back();
    stack.pop_back();
    absl::visit(
        Visitor{
            [&](const std::string& s) {
              // Text is simply emitted to the output and we bump the output
              // column tracker accordingly.
              pieces.push_back(s);
              outcol += s.size();
            },
            [&](const HardLine&) {
              // A hardline command emits a newline and takes it to its
              // corresponding indent level that it was emitted at, and sets the
              // column tracker accordingly.
              pieces.push_back(
                  absl::StrCat("\n", std::string(entry.indent, ' ')));
              outcol = entry.indent;
            },
            [&](const Nest& nest) {
              // Nest bumps the indent in by its delta and then emits the nested
              // doc.
              stack.push_back(StackEntry{&arena.Deref(nest.arg), entry.mode,
                                         entry.indent + nest.delta});
              int64_t new_indent = entry.indent + nest.delta;
              if (outcol < new_indent) {
                pieces.push_back(std::string(new_indent - outcol, ' '));
                outcol = new_indent;
              }
            },
            [&](const Align& align) {
              XLS_VLOG(3) << "Align; outcol: " << outcol;
              // Align sets the alignment for the nested doc to the current
              // line's output column and then emits the nested doc.
              stack.push_back(StackEntry{&arena.Deref(align.arg), entry.mode,
                                         /*indent=*/outcol});
            },
            [&](const PrefixedReflow& prefixed) {
              XLS_VLOG(3) << "PrefixedReflow; prefix: " << prefixed.prefix
                          << " text: " << prefixed.text;
              std::vector<std::string_view> lines =
                  absl::StrSplit(prefixed.text, '\n');
              const std::string& prefix = prefixed.prefix;

              // Remaining columns at this indentation level.
              const int64_t remaining_cols = text_width - outcol;
              const std::string carriage_return =
                  absl::StrCat("\n", std::string(entry.indent, ' '));

              for (std::string_view line : lines) {
                if (prefix.size() + line.size() < remaining_cols) {
                  // If it all fits in available cols, place it there in its
                  // entirety.
                  pieces.push_back(absl::StrCat(prefix, line, carriage_return));
                } else {
                  // Otherwise, place tokens until we encounter EOL and then
                  // wrap. We make sure we put at least one token on each line
                  // to ensure forward progress.
                  std::vector<std::string_view> toks =
                      absl::StrSplit(line, ' ');
                  auto remaining = absl::MakeConstSpan(toks);

                  while (!remaining.empty()) {
                    outcol = entry.indent;
                    pieces.push_back(prefix);
                    outcol += prefix.size();
                    while (!remaining.empty()) {
                      std::string_view tok = remaining.front();
                      remaining.remove_prefix(1);

                      pieces.push_back(std::string{tok});
                      outcol += pieces.back().size();

                      if (!remaining.empty()) {
                        // If the next token isn't going to fit we make a
                        // carriage return and go to the next prefix insertion.
                        if (outcol + remaining.front().size() > text_width) {
                          pieces.push_back(carriage_return);
                          break;
                        }

                        // If the next token is going to fit we just put a
                        // space char.
                        pieces.push_back(" ");
                      }
                    }
                  }
                  // Note: we do not make a trailing newline, because "docs" are
                  // supposed to be self contained (i.e. emitting only their own
                  // contents), so user should put a hardline afterwards.
                }
              }
              outcol = entry.indent;
            },
            [&](const struct Concat& concat) {
              stack.push_back(StackEntry{&arena.Deref(concat.rhs), entry.mode,
                                         entry.indent});
              stack.push_back(StackEntry{&arena.Deref(concat.lhs), entry.mode,
                                         entry.indent});
            },
            [&](const struct FlatChoice& flat_choice) {
              // Flat choice emits the flat doc if we're in flat mode and the
              // break doc if we're in break mode -- this allows us to have
              // different strategies for "when we can fit in the remainder of
              // the line" vs when we can't.
              if (entry.mode == Mode::kFlat) {
                stack.push_back(StackEntry{&arena.Deref(flat_choice.on_flat),
                                           entry.mode, entry.indent});
              } else {
                stack.push_back(StackEntry{&arena.Deref(flat_choice.on_break),
                                           entry.mode, entry.indent});
              }
            },
            [&](const struct Group& group) {
              // Group evaluates whether the nested doc takes limited enough
              // flat space that it can be emitted in flat mode in the columns
              // remaining -- if so, we emit the nested doc in flat mode; if
              // not, we emit it in break mode.
              int64_t remaining_cols = text_width - outcol;
              Requirement grouped_requirement =
                  arena.Deref(group.arg).flat_requirement;
              XLS_VLOG(1) << "grouped_requirement: "
                          << RequirementToString(grouped_requirement)
                          << " remaining_cols: " << remaining_cols;
              if (grouped_requirement > remaining_cols) {
                stack.push_back(StackEntry{&arena.Deref(group.arg),
                                           Mode::kBreak, entry.indent});
              } else {
                stack.push_back(StackEntry{&arena.Deref(group.arg), Mode::kFlat,
                                           entry.indent});
              }
            }},
        entry.doc->value);
  }
}

}  // namespace

DocArena::DocArena() {
  empty_ = MakeText("");
  space_ = MakeText(" ");
  // a hardline
  hard_line_ = DocRef{items_.size()};
  items_.emplace_back(Doc{InfinityRequirement(), HardLine{}});
  // am empty-break
  break0_ = DocRef{items_.size()};
  items_.emplace_back(Doc{0, FlatChoice{empty_, hard_line_}});
  // a space-break
  break1_ = DocRef{items_.size()};
  items_.emplace_back(Doc{1, FlatChoice{space_, hard_line_}});

  oparen_ = MakeText("(");
  cparen_ = MakeText(")");
  comma_ = MakeText(",");
  colon_ = MakeText(":");
  equals_ = MakeText("=");
  dotdot_ = MakeText("..");
  underscore_ = MakeText("_");
  slash_slash_ = MakeText("//");
  ocurl_ = MakeText("{");
  ccurl_ = MakeText("}");
  semi_ = MakeText(";");
  arrow_ = MakeText("->");
  dot_ = MakeText(".");
  obracket_ = MakeText("[");
  cbracket_ = MakeText("]");
  oangle_ = MakeText("<");
  cangle_ = MakeText(">");
}

DocRef DocArena::MakeText(std::string s) {
  int64_t size = items_.size();
  items_.push_back(Doc{static_cast<int64_t>(s.size()), s});
  return DocRef{size};
}

DocRef DocArena::MakeGroup(DocRef arg_ref) {
  const Doc& arg = Deref(arg_ref);
  int64_t size = items_.size();
  items_.push_back(Doc{arg.flat_requirement, Group{arg_ref}});
  return DocRef{size};
}

DocRef DocArena::MakeAlign(DocRef arg_ref) {
  const Doc& arg = Deref(arg_ref);
  int64_t size = items_.size();
  items_.push_back(Doc{arg.flat_requirement, Align{arg_ref}});
  return DocRef{size};
}

DocRef DocArena::MakePrefixedReflow(std::string prefix, std::string text) {
  int64_t size = items_.size();
  const Requirement requirement =
      static_cast<int64_t>(prefix.size() + text.size());
  items_.push_back(
      Doc{requirement, PrefixedReflow{std::move(prefix), std::move(text)}});
  return DocRef{size};
}

DocRef DocArena::MakeNest(DocRef arg_ref, int64_t delta) {
  const Doc& arg = Deref(arg_ref);
  int64_t size = items_.size();
  items_.push_back(Doc{arg.flat_requirement, Nest{delta, arg_ref}});
  return DocRef{size};
}

DocRef DocArena::MakeConcat(DocRef lhs, DocRef rhs) {
  Requirement lhs_req = Deref(lhs).flat_requirement;
  Requirement rhs_req = Deref(rhs).flat_requirement;
  int64_t size = items_.size();
  items_.push_back(Doc{lhs_req + rhs_req, Concat{lhs, rhs}});
  return DocRef{size};
}

DocRef DocArena::MakeFlatChoice(DocRef on_flat, DocRef on_break) {
  Requirement flat_requirement = Deref(on_flat).flat_requirement;
  int64_t size = items_.size();
  items_.push_back(Doc{flat_requirement, FlatChoice{on_flat, on_break}});
  return DocRef{size};
}

DocRef DocArena::Make(Keyword kw) {
  auto it = keyword_to_ref_.find(kw);
  if (it == keyword_to_ref_.end()) {
    it = keyword_to_ref_.emplace_hint(it, kw, MakeText(KeywordToString(kw)));
  }
  return it->second;
}

std::string PrettyPrint(const DocArena& arena, DocRef ref, int64_t text_width) {
  std::vector<std::string> pieces;
  PrettyPrintInternal(arena, arena.Deref(ref), text_width, pieces);
  return absl::StrJoin(pieces, "");
}

DocRef ConcatN(DocArena& arena, absl::Span<DocRef const> docs) {
  if (docs.empty()) {
    return arena.empty();
  }

  DocRef accum = docs[0];
  for (const DocRef& rhs : docs.subspan(1)) {
    accum = arena.MakeConcat(accum, rhs);
  }
  return accum;
}

DocRef ConcatNGroup(DocArena& arena, absl::Span<DocRef const> docs) {
  return arena.MakeGroup(ConcatN(arena, docs));
}

}  // namespace xls::dslx
