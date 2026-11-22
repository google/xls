// Copyright 2026 The XLS Authors
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

#include "xls/dslx/type_system_v2/invocation_collector.h"

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"

namespace xls::dslx {

namespace {

class Visitor : public AstNodeRecursiveVisitor {
 public:
  explicit Visitor(absl::flat_hash_set<const Param*> params)
      : params_(params) {}

  absl::Status HandleInvocation(const Invocation* node) override {
    const Invocation* previous_invocation = current_invocation_;
    current_invocation_ = node;
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    current_invocation_ = previous_invocation;
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* node) override {
    if (current_invocation_ != nullptr) {
      if (node->GetDefiner() != nullptr &&
          node->GetDefiner()->kind() == AstNodeKind::kParam &&
          params_.contains(absl::down_cast<const Param*>(node->GetDefiner()))) {
        invocations_.insert(current_invocation_);
      }
    }
    return DefaultHandler(node);
  }

  const absl::flat_hash_set<const Invocation*>& invocations() const {
    return invocations_;
  }

 private:
  absl::flat_hash_set<const Param*> params_;
  const Invocation* current_invocation_ = nullptr;

  absl::flat_hash_set<const Invocation*> invocations_;
};

}  // namespace

absl::flat_hash_set<const Invocation*> CollectInvocationsReferencingParams(
    const AstNode* node, absl::flat_hash_set<const Param*> params) {
  Visitor visitor(params);
  absl::Status _ = node->Accept(&visitor);
  return visitor.invocations();
}

}  // namespace xls::dslx
