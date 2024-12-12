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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {
namespace {

// Returns true if the node or any of its parents is a desugared function.
// TODO: https://github.com/google/xls/issues/1029 remove desuraged proc
// functions.
bool IsInDesugaredFn(const AstNode *node) {
  if (node == nullptr) {
    return false;
  }
  const Function *f = dynamic_cast<const Function *>(node);
  // If it's a function and not a "normal" function, then it's a desugared
  // function. If not, check its parent.
  return (f != nullptr && f->tag() != FunctionTag::kNormal) ||
         IsInDesugaredFn(node->parent());
}
}  // namespace

absl::StatusOr<std::optional<AstNode *>> FormatDisabler::operator()(
    const AstNode *node) {
  if (seen_nodes_.contains(node)) {
    return std::nullopt;
  }
  seen_nodes_.insert(node);

  if (node == nullptr || !node->GetSpan().has_value()) {
    // If there's no node, or no span, we can't know if it's in the unformatted
    // range, so just return nullopt to indicate it should be unchanged.
    return std::nullopt;
  }
  VLOG(5) << "FormatDisabler looking at " << node->ToString() << " @ "
          << (*node->GetSpan()).ToString(node->owner()->file_name());

  // If this node is part of a desugared proc function, we skip it, because it
  // won't be formatted anyway.
  // TODO: https://github.com/google/xls/issues/1029 remove desugared proc
  // functions.
  if (IsInDesugaredFn(node)) {
    VLOG(5) << "In desugared function, stopping";
    return std::nullopt;
  }

  if (unformatted_end_.has_value()) {
    VLOG(6) << "In format disabled mode";
    // We are in "format disabled" mode.

    if (node->GetSpan()->start() < *unformatted_end_) {
      // This node is within the unformatted range; delete it by returning
      // an empty VerbatimNode. This is safe because its text has already been
      // extracted in a VerbatimNode.
      VLOG(5) << "Setting previous node to " << node->ToString();
      previous_node_ = node;

      return node->owner()->Make<VerbatimNode>(*node->GetSpan());
    }

    // We're past the end of the unformatted range; clear the setting.
    VLOG(5) << "Past end of unformatted range";
    unformatted_end_ = std::nullopt;

    // Note: continue and process this node normally now.
  }

  // Look for a disable comment between the previous node and this node.
  std::vector<const CommentData *> disable_comments =
      FindDisablesBetween(previous_node_, node);
  if (disable_comments.empty()) {
    VLOG(5) << "No comments between previous node and this node";
    VLOG(5) << "Setting previous node to " << node->ToString();
    previous_node_ = node;
    // Node should be unchanged.
    return std::nullopt;
  }
  if (disable_comments.size() > 1) {
    if (previous_node_ == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Multiple dslx-fmt::off commands between module start and ",
          node->ToString()));
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Multiple dslx-fmt::off commands between ",
                     previous_node_->ToString(), " and ", node->ToString()));
  }
  VLOG(5) << "Found disable comment between previous node and this node.";
  const CommentData *disable_comment = disable_comments[0];

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
  Span unformatted_span(disable_comment->span.limit(), limit);
  XLS_ASSIGN_OR_RETURN(std::string text, GetTextInSpan(unformatted_span));

  // c. Create a new VerbatimNode with the content from step b.
  VerbatimNode *verbatim_node =
      node->owner()->Make<VerbatimNode>(unformatted_span, text);

  comments_.RemoveComments(unformatted_span);

  // d. Set a field that indicates the ending position of the last node that
  // should be unformatted.
  VLOG(6) << "Setting unformatted end to " << limit;
  unformatted_end_ = limit;

  previous_node_ = node;

  // e. Replace the current node with the verbatim node from step c
  VLOG(5) << "Setting previous node to, and replacing " << node->ToString()
          << " with VerbatimNode";
  return verbatim_node;
}

std::vector<const CommentData *> FormatDisabler::FindCommentsWithText(
    std::string_view text, Span span) {
  std::vector<const CommentData *> results;
  const std::vector<const CommentData *> comments = comments_.GetComments(span);
  for (const CommentData *comment : comments) {
    if (comment->text == text) {
      results.push_back(comment);
    }
  }
  return results;
}

std::vector<const CommentData *> FormatDisabler::FindDisablesBetween(
    const AstNode *before, const AstNode *current) {
  if (!current->GetSpan().has_value()) {
    return {};
  }

  // Default start to start of module
  Pos start = current->owner()->span().start();
  if (before != nullptr && before->GetSpan().has_value()) {
    // Span from end of previous node to start of this node.
    start = before->GetSpan()->start();
  }

  if (current->GetSpan()->start() < start) {
    // This node is before the previous node, so it's not disableable.
    VLOG(5) << "Node " << current->ToString() << " @ "
            << current->GetSpan()->start().ToStringNoFile()
            << " is before previous span: " << start.ToStringNoFile()
            << ", skipping";
    return {};
  }

  Span span(start, current->GetSpan()->start());
  return FindCommentsWithText(" dslx-fmt::off\n", span);
}

std::optional<const CommentData *> FormatDisabler::FindEnableAfter(
    const AstNode *node) {
  if (!node->GetSpan().has_value()) {
    return std::nullopt;
  }

  // The limit of the span is the end of the module.
  Span span(node->GetSpan()->limit(), node->owner()->span().limit());
  std::vector<const CommentData *> results =
      FindCommentsWithText(" dslx-fmt::on\n", span);
  if (results.empty()) {
    return std::nullopt;
  }
  return results[0];
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
    XLS_ASSIGN_OR_RETURN(contents_, vfs_.GetFileContents(*path_));
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
    return absl::InvalidArgumentError(absl::StrFormat(
        "Start line %d out of bounds: max %d", start_line, lines_.size()));
  }
  if (end_line < 0 || end_line > lines_.size() || end_line < start_line) {
    return absl::InvalidArgumentError(
        absl::StrFormat("End line %d out of bounds: start %d, max %d", end_line,
                        start_line, lines_.size()));
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
