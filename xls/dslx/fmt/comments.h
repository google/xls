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

#ifndef XLS_DSLX_FMT_COMMENTS_H_
#define XLS_DSLX_FMT_COMMENTS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
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

  // Removes all comments within the given `node_span`.
  void RemoveComments(const Span& node_span);

  // Returns whether there are any comments contained in the given span.
  bool HasComments(const Span& in_span) const;

  // Indicate that this comment was "placed" in the formatted output.
  void PlaceComment(const CommentData* comment) {
    placed_comments_.insert(comment);
  }
  bool WasPlaced(const CommentData* comment) const {
    return placed_comments_.contains(comment);
  }

  const std::optional<Pos>& last_data_limit() const { return last_data_limit_; }

 private:
  Comments(absl::flat_hash_map<int64_t, CommentData> line_to_comment,
           std::optional<Pos> last_data_limit)
      : line_to_comment_(std::move(line_to_comment)),
        last_data_limit_(last_data_limit) {}

  absl::flat_hash_map<int64_t, CommentData> line_to_comment_;
  std::optional<Pos> last_data_limit_;
  absl::flat_hash_set<const CommentData*> placed_comments_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FMT_COMMENTS_H_
