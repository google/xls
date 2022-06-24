// Copyright 2022 The XLS Authors
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
#ifndef XLS_DSLX_AST_CLONER_H_
#define XLS_DSLX_AST_CLONER_H_

#include "absl/status/statusor.h"
#include "xls/dslx/ast.h"

namespace xls::dslx {

// Creates a deep copy of the given AST node (inside the same module). All nodes
// in the tree are duplicated.
absl::StatusOr<AstNode*> CloneAst(AstNode* root);
absl::StatusOr<std::unique_ptr<Module>> CloneModule(Module* module);

// Verifies that the AST node tree rooted at `new_root` does not contain any of
// the AST nodes in the tree rooted at `old_root`. In practice, this will verify
// that a clone doesn't contain any 'old' AST nodes.
absl::Status VerifyClone(const AstNode* old_root, const AstNode* new_root);

}  // namespace xls::dslx

#endif  // XLS_DSLX_AST_CLONER_H_
