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

#ifndef XLS_DSLX_FMT_AST_FMT_H_
#define XLS_DSLX_FMT_AST_FMT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// API convenience wrapper around comment data that the scanner produces -- this
// allows us to look up "what comments an AST node is responsible for" via
// `GetComments()` providing the AST node span.
class Comments {
 public:
  static Comments Create(absl::Span<const CommentData> comments);

  Comments() = default;

  // Returns all the comments contained within the given `node_span`.
  //
  // This is a convenient way for nodes to query for all their related comments.
  std::vector<const CommentData*> GetComments(const Span& node_span) const;

  // Returns whether there are any comments contained in the given span.
  bool HasComments(const Span& in_span) const;

  const std::optional<Pos>& last_data_limit() const { return last_data_limit_; }

 private:
  Comments(absl::flat_hash_map<int64_t, CommentData> line_to_comment,
           std::optional<Pos> last_data_limit)
      : line_to_comment_(std::move(line_to_comment)),
        last_data_limit_(last_data_limit) {}

  absl::flat_hash_map<int64_t, CommentData> line_to_comment_;
  std::optional<Pos> last_data_limit_;
};

// Functions with this signature create a pretty printable document from the AST
// node "n".

DocRef Fmt(const Expr& n, const Comments& comments, DocArena& arena);

// Create a pretty-printable document from the given AST Statement node `n`.
// If `trailing_semi` is true, then a trailing semicolon will also be emitted.
DocRef Fmt(const Statement& n, const Comments& comments, DocArena& arena,
           bool trailing_semi);

// Create a pretty-printable document from the given AST Let node `n`.
// If `trailing_semi` is true, then a trailing semicolon will also be emitted.
DocRef Fmt(const Let& n, const Comments& comments, DocArena& arena,
           bool trailing_semi);

DocRef Fmt(const Function& n, const Comments& comments, DocArena& arena);

DocRef Fmt(const Module& n, const Comments& comments, DocArena& arena);

inline constexpr int64_t kDslxDefaultTextWidth = 100;

// Auto-formatting entry point.
//
// Performs a reflow-capable formatting of module "m" with standard line width.
std::string AutoFmt(const Module& m, const Comments& comments,
                    int64_t text_width = kDslxDefaultTextWidth);

// If we fail the postcondition we return back the data we used to detect that
// the postcondition was violated.
struct AutoFmtPostconditionViolation {
  std::string original_transformed;
  std::string autofmt_transformed;
};

// Checks whether the auto-formatting process looks "opportunistically sound" --
// that is, this will not hold true for all examples, but it'll hold true for a
// bunch of them, and so can be a useful debugging tool.
//
// It's difficult to come up with a /simple/ postcondition for the
// auto-formatter because it does some cleanup transformations based on the
// grammar, and we want this to be a simple linear / regexp style check on the
// flattened text, so we can't account for all the transforms that the
// autoformatter may perform. Still, it's useful in testing or debugging
// scenarios where we know none of those constructs / situations are present.
std::optional<AutoFmtPostconditionViolation>
ObeysAutoFmtOpportunisticPostcondition(std::string_view original,
                                       std::string_view autofmt);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FMT_AST_FMT_H_
