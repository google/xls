// Copyright 2024 The XLS Authors
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

#include "xls/dslx/fmt/comments.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// TODO: davidplass - Write tests for this class/file.
/* static */ Comments Comments::Create(absl::Span<const CommentData> comments) {
  std::optional<Pos> last_data_limit;
  absl::flat_hash_map<int64_t, CommentData> line_to_comment;
  for (const CommentData& cd : comments) {
    VLOG(3) << "comment on line: " << cd.span.start().lineno();
    // Note: we don't have multi-line comments for now, so we just note the
    // start line number for the comment.
    line_to_comment[cd.span.start().lineno()] = cd;
    if (last_data_limit.has_value()) {
      last_data_limit = std::max(cd.span.limit(), last_data_limit.value());
    } else {
      last_data_limit = cd.span.limit();
    }
  }
  return Comments{std::move(line_to_comment), last_data_limit};
}

static bool InRange(const Span& node_span, const CommentData& comment) {
  // For multiline comments, consider in range if the comment start is within
  // the node span. Since all comments end on the line below the comment,
  // multiline comments have > 1 line between the start and end.
  bool overlapping_multiline =
      (comment.span.limit().lineno() - 1 > comment.span.start().lineno()) &&
      node_span.Contains(comment.span.start());
  return overlapping_multiline || node_span.Contains(comment.span);
}

bool Comments::HasComments(const Span& in_span) const {
  for (int64_t i = in_span.start().lineno(); i <= in_span.limit().lineno();
       ++i) {
    if (auto it = line_to_comment_.find(i); it != line_to_comment_.end()) {
      const CommentData& cd = it->second;
      if (InRange(in_span, cd)) {
        return true;
      }
    }
  }
  return false;
}

std::vector<const CommentData*> Comments::GetComments(
    const Span& node_span) const {
  // Implementation note: this will typically be a single access (as most things
  // will be on a single line), so we prefer a flat hash map to a btree map.
  std::vector<const CommentData*> results;
  for (int64_t i = node_span.start().lineno(); i <= node_span.limit().lineno();
       ++i) {
    if (auto it = line_to_comment_.find(i); it != line_to_comment_.end()) {
      // Check that the comment is properly contained within the given
      // "node_span" we were targeting. E.g. the user might be requesting a
      // subspan of a line, we don't want to give a comment that came
      // afterwards.
      const CommentData& cd = it->second;
      if (InRange(node_span, cd)) {
        results.push_back(&cd);
      }
    }
  }
  return results;
}

void Comments::RemoveComments(const Span& node_span) {
  for (int64_t i = node_span.start().lineno(); i <= node_span.limit().lineno();
       ++i) {
    if (auto it = line_to_comment_.find(i); it != line_to_comment_.end()) {
      // Check that the comment is properly contained within the given
      // "node_span" we were targeting. E.g. the user might be requesting a
      // subspan of a line, we don't want to give a comment that came
      // afterwards.
      const CommentData& cd = it->second;
      if (InRange(node_span, cd)) {
        line_to_comment_.erase(it);
      }
    }
  }
}

}  // namespace xls::dslx
