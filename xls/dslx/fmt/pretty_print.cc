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

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
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
using pprint_internal::NestIfFlatFits;
using pprint_internal::PrefixedReflow;
using pprint_internal::ReduceTextWidth;
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

inline std::ostream& operator<<(std::ostream& os, Mode mode) {
  switch (mode) {
    case Mode::kFlat:
      os << "flat";
      break;
    case Mode::kBreak:
      os << "break";
      break;
  }
  return os;
}

class StackEntry {
 public:
  StackEntry(const Doc* doc, Mode mode, int64_t indent, int64_t text_width)
      : doc_(doc), mode_(mode), indent_(indent), text_width_(text_width) {}

  const Doc* doc() const { return doc_; }
  Mode mode() const { return mode_; }
  int64_t indent() const { return indent_; }
  int64_t text_width() const { return text_width_; }

  StackEntry CloneWithDoc(const Doc* other) const {
    return StackEntry(other, mode_, indent_, text_width_);
  }

 private:
  const Doc* doc_;
  Mode mode_;
  int64_t indent_;
  int64_t text_width_;
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
bool operator<=(Requirement lhs, Requirement rhs) {
  bool gt = lhs > rhs;
  return !gt;
}

void PrettyPrintInternal(const DocArena& arena, const Doc& doc,
                         const int64_t default_text_width,
                         std::vector<std::string>& pieces) {
  VLOG(1) << "PrettyPrintInternal; default text width: " << default_text_width;

  // We maintain a stack to keep track of doc emission we still need to perform.
  // Every entry notes the document to emit, what mode it was in (flat or
  // line-breaking mode) and what indent level it was supposed to be emitted at.
  std::vector<StackEntry> stack = {
      StackEntry(&doc, Mode::kFlat, 0, default_text_width)};

  // Number of columns we've output in the current line. (This is reset by hard
  // line breaks.)
  int64_t real_outcol = 0;

  // This number can be >= real_outcol when we anticipate inserting leading
  // spaces, but we don't want to /realy/ insert them yet, because we want to
  // avoid whitespace before newlines on newline-only lines.
  int64_t virtual_outcol = 0;

  auto emit = [&](std::string_view s) {
    if (real_outcol < virtual_outcol) {
      pieces.push_back(std::string(virtual_outcol - real_outcol, ' '));
      real_outcol = virtual_outcol;
    }
    pieces.push_back(std::string{s});
    real_outcol += s.size();
    virtual_outcol += s.size();
  };
  auto emit_cr = [&](int64_t indent) {
    pieces.push_back("\n");
    real_outcol = 0;
    virtual_outcol = indent;
  };

  while (!stack.empty()) {
    StackEntry entry = stack.back();
    stack.pop_back();

    absl::visit(
        Visitor{
            [&](const std::string& s) {
              VLOG(3) << "emitting text: `" << s
                      << "` at virtual outcol: " << virtual_outcol << " (size "
                      << s.size() << ")";
              // Text is simply emitted to the output and we bump the output
              // column tracker accordingly.
              emit(s);
            },
            [&](const HardLine&) {
              // A hardline command emits a newline and takes it to its
              // corresponding indent level that it was emitted at, and sets the
              // column tracker accordingly.
              emit_cr(entry.indent());
            },
            [&](const Nest& nest) {
              // Nest bumps the indent in by its delta and then emits the nested
              // doc.
              int64_t new_indent = entry.indent() + nest.delta;
              stack.push_back(StackEntry{&arena.Deref(nest.arg), entry.mode(),
                                         new_indent, entry.text_width()});
              if (virtual_outcol < new_indent) {
                virtual_outcol = new_indent;
              }
            },
            [&](const Align& align) {
              VLOG(3) << "Align; outcol: " << virtual_outcol;
              // Align sets the alignment for the nested doc to the current
              // line's output column and then emits the nested doc.
              stack.push_back(StackEntry{&arena.Deref(align.arg), entry.mode(),
                                         /*indent=*/virtual_outcol,
                                         entry.text_width()});
            },
            [&](const PrefixedReflow& prefixed) {
              VLOG(3) << "PrefixedReflow; prefix: `" << prefixed.prefix
                      << "` text: `" << prefixed.text << "`";
              std::vector<std::string_view> lines =
                  absl::StrSplit(prefixed.text, '\n');
              const std::string& prefix = prefixed.prefix;

              for (size_t i = 0; i < lines.size(); ++i) {
                std::string_view line = lines[i];

                // Remaining columns at this indentation level.
                const int64_t remaining_cols =
                    entry.text_width() - virtual_outcol;

                VLOG(5) << "PrefixedReflow; handling line: `" << line
                        << "` remaining cols: " << remaining_cols;
                if (prefix.size() + line.size() < remaining_cols) {
                  // If it all fits in available cols, place it there in its
                  // entirety.
                  emit(absl::StrCat(prefix, line));
                  if (i + 1 != lines.size()) {
                    emit_cr(entry.indent());
                  }
                } else {
                  // Otherwise, place tokens until we encounter EOL and then
                  // wrap. We make sure we put at least one token on each line
                  // to ensure forward progress.

                  // We keep the leading whitespace on the side because we want
                  // to preserve that, so that folks can put custom spacing at
                  // the start of their line (e.g. think of a markdown quote
                  // block).
                  std::string_view line_no_leading_whitespace =
                      absl::StripLeadingAsciiWhitespace(line);
                  size_t leading_whitespace_size =
                      line.size() - line_no_leading_whitespace.size();

                  std::vector<std::string> toks =
                      absl::StrSplit(line_no_leading_whitespace, ' ');

                  absl::Span<const std::string> remaining_toks =
                      absl::MakeConstSpan(toks);

                  while (!remaining_toks.empty()) {
                    emit(prefix);
                    emit(std::string(leading_whitespace_size, ' '));

                    // After we emit the prefix we make sure we emit at least
                    // one token.
                    while (!remaining_toks.empty()) {
                      std::string_view tok = remaining_toks.front();
                      remaining_toks.remove_prefix(1);

                      emit(tok);

                      if (!remaining_toks.empty()) {
                        // If the next token isn't going to fit we make a
                        // carriage return and go to the next prefix insertion.
                        const std::string& next_tok = remaining_toks.front();
                        if (virtual_outcol + next_tok.size() >
                            entry.text_width()) {
                          VLOG(5) << "PrefixedReflow; adding carriage "
                                     "return in advance of: `"
                                  << next_tok
                                  << "` as it will not fit; virtual outcol: "
                                  << virtual_outcol
                                  << " tok width: " << next_tok.size()
                                  << " text width: " << entry.text_width();
                          emit_cr(entry.indent());
                          break;
                        }

                        // If the next token is going to fit we just put a
                        // space char.
                        emit(" ");
                      }
                    }
                  }
                  // Note: we do not make a trailing newline, because "docs" are
                  // supposed to be self contained (i.e. emitting only their own
                  // contents), so user should put a hardline afterwards.
                }
              }
            },
            [&](const struct Concat& concat) {
              stack.push_back(entry.CloneWithDoc(&arena.Deref(concat.rhs)));
              stack.push_back(entry.CloneWithDoc(&arena.Deref(concat.lhs)));
            },
            [&](const struct FlatChoice& flat_choice) {
              // Flat choice emits the flat doc if we're in flat mode and the
              // break doc if we're in break mode -- this allows us to have
              // different strategies for "when we can fit in the remainder of
              // the line" vs when we can't.
              if (entry.mode() == Mode::kFlat) {
                stack.push_back(
                    entry.CloneWithDoc(&arena.Deref(flat_choice.on_flat)));
              } else {
                CHECK_EQ(entry.mode(), Mode::kBreak);
                VLOG(3) << "emitting FlatChoice as break mode: "
                        << entry.doc()->ToDebugString(arena);
                stack.push_back(
                    entry.CloneWithDoc(&arena.Deref(flat_choice.on_break)));
              }
            },
            [&](const struct ReduceTextWidth& reduce_text_width) {
              stack.push_back(StackEntry(
                  &arena.Deref(reduce_text_width.arg), entry.mode(),
                  entry.indent(), entry.text_width() - reduce_text_width.cols));
            },
            [&](const struct NestIfFlatFits& nest_if_flat_fits) {
              const Doc& on_nested_flat =
                  arena.Deref(nest_if_flat_fits.on_nested_flat);
              const Doc& on_other = arena.Deref(nest_if_flat_fits.on_other);

              int64_t remaining_cols = entry.text_width() - virtual_outcol;
              int64_t remaining_cols_with_newline =
                  entry.text_width() - entry.indent() - 4;

              VLOG(3) << "NestIfFlatFits; on_other.flat_requirement: "
                      << RequirementToString(on_other.flat_requirement)
                      << " remaining_cols: " << remaining_cols
                      << " on_nested_flat.flat_requirement: "
                      << RequirementToString(on_nested_flat.flat_requirement)
                      << " remaining_cols_with_newline: "
                      << remaining_cols_with_newline;
              if (on_other.flat_requirement <= remaining_cols) {
                stack.push_back(StackEntry(&on_other, Mode::kFlat,
                                           virtual_outcol, entry.text_width()));
              } else if (on_nested_flat.flat_requirement <=
                         remaining_cols_with_newline) {
                int64_t nested_indent = entry.indent() + 4;
                emit_cr(nested_indent);
                // Note that because we've determined the "on_nested_flat" doc
                // should fit flat into this new line, the "nested_indent" value
                // will never be observed.
                stack.push_back(StackEntry{&on_nested_flat, Mode::kFlat,
                                           nested_indent, entry.text_width()});
              } else {
                stack.push_back(StackEntry{&on_other, Mode::kBreak,
                                           entry.indent(), entry.text_width()});
              }
            },
            [&](const struct Group& group) {
              // Group evaluates whether the nested doc takes limited enough
              // flat space that it can be emitted in flat mode in the columns
              // remaining -- if so, we emit the nested doc in flat mode; if
              // not, we emit it in break mode.
              int64_t remaining_cols = entry.text_width() - virtual_outcol;
              Requirement grouped_requirement =
                  arena.Deref(group.arg).flat_requirement;
              Mode mode = grouped_requirement > remaining_cols ? Mode::kBreak
                                                               : Mode::kFlat;
              VLOG(3) << "grouped_requirement: "
                      << RequirementToString(grouped_requirement)
                      << " remaining_cols: " << remaining_cols
                      << "; now using mode: " << mode << " arg: "
                      << arena.Deref(group.arg).ToDebugString(arena);
              stack.push_back(StackEntry{&arena.Deref(group.arg), mode,
                                         entry.indent(), entry.text_width()});
            }},
        entry.doc()->value);
  }
}

}  // namespace

std::string Doc::ToDebugString(const DocArena& arena) const {
  std::string payload = absl::visit(
      Visitor{
          [&](const std::string& p) -> std::string {
            return absl::StrFormat("\"%s\"", absl::CEscape(p));
          },
          [&](const HardLine& p) -> std::string { return "HardLine"; },
          [&](const FlatChoice& p) -> std::string {
            return absl::StrFormat(
                "FlatChoice{on_flat=%s, on_break=%s}",
                arena.Deref(p.on_flat).ToDebugString(arena),
                arena.Deref(p.on_break).ToDebugString(arena));
          },
          [&](const Group& p) -> std::string {
            return absl::StrFormat("Group{%s}",
                                   arena.Deref(p.arg).ToDebugString(arena));
          },
          [&](const NestIfFlatFits& p) -> std::string {
            return absl::StrFormat(
                "NestIfFlatFits{on_flat=%s, on_break=%s}",
                arena.Deref(p.on_nested_flat).ToDebugString(arena),
                arena.Deref(p.on_other).ToDebugString(arena));
          },
          [&](const Concat& p) -> std::string {
            return absl::StrFormat("Concat{%s, %s}",
                                   arena.Deref(p.lhs).ToDebugString(arena),
                                   arena.Deref(p.rhs).ToDebugString(arena));
          },
          [&](const ReduceTextWidth& p) -> std::string {
            return absl::StrFormat("ReduceTextWidth{%s, %d}",
                                   arena.Deref(p.arg).ToDebugString(arena),
                                   p.cols);
          },
          [&](const Nest& p) -> std::string { return "Nest"; },
          [&](const Align& p) -> std::string {
            return absl::StrFormat("Align{%s}",
                                   arena.Deref(p.arg).ToDebugString(arena));
          },
          [&](const PrefixedReflow& p) -> std::string {
            return absl::StrFormat("PrefixedReflow{\"%s\", \"%s\"}",
                                   absl::CEscape(p.prefix),
                                   absl::CEscape(p.text));
          },
      },
      value);
  return absl::StrFormat("Doc{%s, %s}",
                         std::holds_alternative<int64_t>(flat_requirement)
                             ? absl::StrCat(std::get<int64_t>(flat_requirement))
                             : "inf",
                         payload);
}

DocArena::DocArena(const FileTable& file_table) : file_table_(file_table) {
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

  force_break_mode_ = DocRef{items_.size()};
  items_.emplace_back(Doc{InfinityRequirement(), ""});

  oparen_ = MakeText("(");
  cparen_ = MakeText(")");
  comma_ = MakeText(",");
  colon_ = MakeText(":");
  equals_ = MakeText("=");
  dot_dot_ = MakeText("..");
  underscore_ = MakeText("_");
  slash_slash_ = MakeText("//");
  ocurl_ = MakeText("{");
  ccurl_ = MakeText("}");
  semi_ = MakeText(";");
  arrow_ = MakeText("->");
  fat_arrow_ = MakeText("=>");
  dot_ = MakeText(".");
  obracket_ = MakeText("[");
  cbracket_ = MakeText("]");
  oangle_ = MakeText("<");
  cangle_ = MakeText(">");
  plus_colon_ = MakeText("+:");
  colon_colon_ = MakeText("::");
  bar_ = MakeText("|");
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

DocRef DocArena::MakeReduceTextWidth(DocRef arg_ref, int64_t cols) {
  const Doc& arg = Deref(arg_ref);
  int64_t size = items_.size();
  items_.push_back(Doc{arg.flat_requirement, ReduceTextWidth{arg_ref, cols}});
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

DocRef DocArena::MakeNestIfFlatFits(DocRef on_nested_flat_ref,
                                    DocRef on_other_ref) {
  Requirement flat_requirement = Deref(on_other_ref).flat_requirement;
  int64_t size = items_.size();
  items_.push_back(
      Doc{flat_requirement, NestIfFlatFits{on_nested_flat_ref, on_other_ref}});
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
