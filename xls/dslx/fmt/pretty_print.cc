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
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"

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
            },
            [&](const Align& align) {
              // Align sets the alignment for the nested doc to the current
              // line's output column and then emits the nested doc.
              stack.push_back(
                  StackEntry{&arena.Deref(align.arg), entry.mode, outcol});
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
  // empty string
  empty_ = DocRef{items_.size()};
  items_.emplace_back(Doc{0, ""});
  // space string
  space_ = DocRef{items_.size()};
  items_.emplace_back(Doc{1, " "});
  // a hardline
  hard_line_ = DocRef{items_.size()};
  items_.emplace_back(Doc{InfinityRequirement(), HardLine{}});
  // am empty-break
  break0_ = DocRef{items_.size()};
  items_.emplace_back(Doc{0, FlatChoice{empty_, hard_line_}});
  // a space-break
  break1_ = DocRef{items_.size()};
  items_.emplace_back(Doc{1, FlatChoice{space_, hard_line_}});
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

std::string PrettyPrint(const DocArena& arena, DocRef ref, int64_t text_width) {
  std::vector<std::string> pieces;
  PrettyPrintInternal(arena, arena.Deref(ref), text_width, pieces);
  return absl::StrJoin(pieces, "");
}

DocRef ConcatN(DocArena& arena, DocRef lhs, absl::Span<const DocRef> rest) {
  DocRef accum = lhs;
  for (const DocRef& rhs : rest) {
    accum = arena.MakeConcat(accum, rhs);
  }
  return accum;
}

}  // namespace xls::dslx
