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

#include "xls/dslx/frontend/ast_node.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

AstNode::~AstNode() = default;

void AstNode::SetParentage() {
  for (AstNode* kiddo : GetChildren(/*want_types=*/true)) {
    kiddo->set_parent(this);
  }
}

absl::Status WalkPostOrder(AstNode* root, AstNodeVisitor* visitor,
                           bool want_types) {
  for (AstNode* child : root->GetChildren(want_types)) {
    XLS_RETURN_IF_ERROR(WalkPostOrder(child, visitor, want_types));
  }
  return root->Accept(visitor);
}

}  // namespace xls::dslx
