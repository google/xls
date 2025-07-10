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
#ifndef XLS_DSLX_TRAIT_VISITOR_H_
#define XLS_DSLX_TRAIT_VISITOR_H_

#include <vector>

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"

namespace xls::dslx {

// Collects various traits/etc. about a DSLX Expr tree.
// Lazily populated as information is needed.
class TraitVisitor : public ExprVisitorWithDefault {
 public:
  template <typename T>
  absl::Status Handle(const T* expr) {
    for (AstNode* child : expr->GetChildren(true)) {
      if (Expr* child_expr = dynamic_cast<Expr*>(child); child_expr) {
        XLS_RETURN_IF_ERROR(child_expr->AcceptExpr(this));
      }
    }
    return absl::OkStatus();
  }

  template <>
  absl::Status Handle<NameRef>(const NameRef* expr) {
    name_refs_.push_back(expr);
    return absl::OkStatus();
  }

#define DEFINE_HANDLER(__type)                               \
  absl::Status Handle##__type(const __type* expr) override { \
    return Handle(expr);                                     \
  }

  XLS_DSLX_EXPR_NODE_EACH(DEFINE_HANDLER)

  const std::vector<const NameRef*>& name_refs() { return name_refs_; }

 private:
  std::vector<const NameRef*> name_refs_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TRAIT_VISITOR_H_
