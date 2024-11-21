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

#ifndef XLS_DSLX_FMT_FORMAT_DISABLER_H_
#define XLS_DSLX_FMT_FORMAT_DISABLER_H_

#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

// Implements formatter disabling via "dsl-fmt::off" and "dsl-fmt::on" comments.
//
// Used by the auto-formatter to implement the directives.
//
// NOTE: this class is stateful and should only be used once per module.
class FormatDisabler {
 public:
  FormatDisabler(VirtualizableFilesystem& vfs, Comments& comments,
                 const std::string& contents)
      : vfs_(vfs), comments_(comments), contents_(contents){};
  FormatDisabler(VirtualizableFilesystem& vfs, Comments& comments,
                 const std::filesystem::path& path)
      : vfs_(vfs), comments_(comments), path_(path){};

  // Functor that implements the 'CloneReplacer' interface.
  absl::StatusOr<std::optional<AstNode *>> operator()(const AstNode *node);

 private:
  // Returns the text in the given span.
  absl::StatusOr<std::string> GetTextInSpan(const Span &span);

  // Internal set-up method that assigns contents_ and lines_ (if they were not
  // already set), based on the contents_ or path_ fields as the source of the
  // contents.
  absl::Status SetContents();

  // Find CommentData objects with the matching text in the given span.
  std::vector<const CommentData *> FindCommentsWithText(std::string_view text,
                                                        Span span);

  // Find a CommentData object that is a disable comment between the two nodes.
  std::vector<const CommentData *> FindDisablesBetween(const AstNode *before,
                                                       const AstNode *current);

  // Find a CommentData object that is a enable comment after the given node
  std::optional<const CommentData *> FindEnableAfter(const AstNode *node);

  VirtualizableFilesystem& vfs_;

  Comments &comments_;

  // The previous node that was processed. We use this to find any "disable"
  // comments between there and the current node.
  const AstNode *previous_node_ = nullptr;

  // The position where we should resume formatting.
  std::optional<Pos> unformatted_end_ = std::nullopt;

  // Full contents of the module.
  std::optional<std::string> contents_ = std::nullopt;

  // Path of the module.
  std::optional<std::filesystem::path> path_ = std::nullopt;

  // Parsed lines of the module.
  std::vector<std::string> lines_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FMT_FORMAT_DISABLER_H_
