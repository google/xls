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

#ifndef XLS_DSLX_FRONTEND_AST_NODE_VISITOR_WITH_DEFAULT_H_
#define XLS_DSLX_FRONTEND_AST_NODE_VISITOR_WITH_DEFAULT_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"  // IWYU pragma: export
#include "xls/dslx/frontend/module.h"    // IWYU pragma: export
#include "xls/dslx/frontend/proc.h"      // IWYU pragma: export

namespace xls::dslx {

// Subtype of abstract AstNodeVisitor that returns ok status (does nothing) for
// every node type.
//
// Users can override the default behavior by overriding the DefaultHandler()
// method.
class AstNodeVisitorWithDefault : public AstNodeVisitor {
 public:
  ~AstNodeVisitorWithDefault() override = default;

  virtual absl::Status DefaultHandler(const AstNode*) {
    return absl::OkStatus();
  }

#define DECLARE_HANDLER(__type)                           \
  absl::Status Handle##__type(const __type* n) override { \
    return DefaultHandler(n);                             \
  }
  XLS_DSLX_AST_NODE_EACH(DECLARE_HANDLER)
#undef DECLARE_HANDLER
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AST_NODE_VISITOR_WITH_DEFAULT_H_
