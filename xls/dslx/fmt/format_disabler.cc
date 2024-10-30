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

#include "xls/dslx/fmt/format_disabler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

absl::StatusOr<std::optional<AstNode *>> FormatDisabler::operator()(
    const AstNode *node) {
  if (node == nullptr || !node->GetSpan().has_value()) {
    // If there's no node, or no span, we can't know if it's in the unformatted
    // range, so just return nullopt to indicate it should be unchanged.
    return std::nullopt;
  }

  if (unformatted_end_.has_value()) {
    // We are in "format disabled" mode.

    if (node->GetSpan()->start() < *unformatted_end_) {
      // This node is within the unformatted range; delete it by returning
      // an empty VerbatimNode. This is safe because its text has already been
      // extracted in a VerbatimNode.
      previous_node_ = node;

      return node->owner()->Make<VerbatimNode>(*node->GetSpan());

      // TODO: https://github.com/google/xls/issues/1320 - also remove comments
      // that are in this range; their text will be included in the previous
      // "VerbatimNode", so they should be removed from the Comments object
      // (which, sadly, does not support removal of comments).
    }

    // We're past the end of the unformatted range; clear the setting.
    unformatted_end_ = std::nullopt;

    // Note: continue and process this node normally now.
  }

  // Look for a disable comment between the previous node and this node.
  std::optional<const CommentData *> disable_comment = FindDisableBefore(node);
  if (!disable_comment.has_value()) {
    previous_node_ = node;
    // Node should be unchanged.
    return std::nullopt;
  }

  // TODO: https://github.com/google/xls/issues/1320 - detect two disable
  // formatting comments in a row and report an error.

  // If there's a disable comment between the previous node and this node:

  // a. Look for the next enable formatting comment in the comment stream and
  // use its end position as the end position.
  std::optional<const CommentData *> enable_comment = FindEnableAfter(node);

  // The limit defaults to the end of the module.
  Pos limit = node->owner()->span().limit();
  if (enable_comment.has_value()) {
    // Use the end of the enable comment as the end position, so it is also left
    // unformatted.
    limit = (*enable_comment)->span.limit();
  }

  // b. Extract the content between the disable comment and the end position
  // from the original text.
  Span unformatted_span((*disable_comment)->span.limit(), limit);
  XLS_ASSIGN_OR_RETURN(std::string text, GetTextInSpan(unformatted_span));

  // c. Create a new VerbatimNode with the content from step b.
  VerbatimNode *verbatim_node =
      node->owner()->Make<VerbatimNode>(unformatted_span, text);

  // d. Set a field that indicates the ending position of the last node that
  // should be unformatted.
  unformatted_end_ = limit;

  previous_node_ = node;

  // e. Replace the current node with the verbatim node from step c
  return verbatim_node;
}

std::optional<const CommentData *> FormatDisabler::FindCommentWithText(
    std::string_view text, Span span) {
  const std::vector<const CommentData *> comments = comments_.GetComments(span);
  for (const CommentData *comment : comments) {
    if (comment->text == text) {
      return comment;
    }
  }
  return std::nullopt;
}

std::optional<const CommentData *> FormatDisabler::FindDisableBefore(
    const AstNode *node) {
  if (!node->GetSpan().has_value()) {
    return std::nullopt;
  }

  // Default start to start of module
  Pos start = node->owner()->span().start();
  if (previous_node_ != nullptr && previous_node_->GetSpan().has_value()) {
    // Span from end of previous node to start of this node.
    start = previous_node_->GetSpan()->start();
  }

  Span span(start, node->GetSpan()->start());
  return FindCommentWithText(" dslx-fmt::off\n", span);
}

std::optional<const CommentData *> FormatDisabler::FindEnableAfter(
    const AstNode *node) {
  if (!node->GetSpan().has_value()) {
    return std::nullopt;
  }

  // The limit of the span is the end of the module.
  Span span(node->GetSpan()->limit(), node->owner()->span().limit());
  return FindCommentWithText(" dslx-fmt::on\n", span);
}

// Checks that the column is non-negative and within the line length.
absl::Status CheckBounds(std::string_view line, int64_t column) {
  if (column < 0 || column > line.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Column out of bounds: ", column));
  }
  return absl::OkStatus();
}

absl::Status FormatDisabler::SetContents() {
  if (!contents_.has_value()) {
    if (!path_.has_value()) {
      return absl::NotFoundError("Path not found");
    }
    XLS_ASSIGN_OR_RETURN(contents_, GetFileContents(*path_));
  }

  if (lines_.empty()) {
    lines_ = absl::StrSplit(*contents_, '\n');
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> FormatDisabler::GetTextInSpan(const Span &span) {
  XLS_RETURN_IF_ERROR(SetContents());

  int64_t start_line = span.start().lineno();
  int64_t end_line = span.limit().lineno();
  if (start_line < 0 || start_line > lines_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Start line out of bounds: ", start_line));
  }
  if (end_line < 0 || end_line > lines_.size() || end_line < start_line) {
    return absl::InvalidArgumentError(
        absl::StrCat("End line out of bounds: ", end_line));
  }

  int64_t end_column = span.limit().colno();
  int64_t start_column = span.start().colno();

  if (start_line == end_line) {
    // All on the same line; just use substring.
    std::string line = lines_[start_line];
    XLS_RETURN_IF_ERROR(CheckBounds(line, start_column));
    XLS_RETURN_IF_ERROR(CheckBounds(line, end_column));
    std::string fragment = line.substr(start_column, end_column - start_column);
    return fragment;
  }

  std::vector<std::string> results;
  for (int64_t i = start_line; i <= end_line; ++i) {
    std::string line = lines_[i];
    // Note: it's not possible to be on the start line and NOT start at the
    // 0th column since comments go to end of line, so we don't need to use
    // substring unless we're on the end line.
    if (i > start_line) {
      // Terminate the previous line.
      results.push_back("\n");
    }
    if (i == end_line) {
      XLS_RETURN_IF_ERROR(CheckBounds(line, end_column));
      line = line.substr(0, end_column);
    }
    results.push_back(line);
  }

  return absl::StrJoin(results, "");
}

}  // namespace xls::dslx
