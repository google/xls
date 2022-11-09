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

#include "xls/passes/token_provenance_analysis.h"

#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/dataflow_visitor.h"

namespace xls {
namespace {

class TokenProvenanceVisitor : public DataFlowVisitor<Node*> {
 public:
  absl::Status DefaultHandler(Node* node) override {
    LeafTypeTree<Node*> ltt(node->GetType(), nullptr);
    if (node->Is<Literal>() || node->Is<Param>() || node->Is<Assert>() ||
        node->Is<Cover>() || node->Is<Trace>() || node->Is<Receive>() ||
        node->Is<Send>() || node->Is<AfterAll>()) {
      for (int64_t i = 0; i < ltt.size(); ++i) {
        if (ltt.leaf_types().at(i)->IsToken()) {
          ltt.elements().at(i) = node;
        }
      }
    } else if (TypeHasToken(node->GetType())) {
      return absl::InternalError(absl::StrFormat(
          "Node type contains token type even though it shouldn't: %s",
          node->ToString()));
    }
    return SetValue(node, std::move(ltt));
  }
};

}  // namespace

absl::StatusOr<TokenProvenance> TokenProvenanceAnalysis(FunctionBase* f) {
  TokenProvenanceVisitor visitor;
  XLS_RETURN_IF_ERROR(f->Accept(&visitor));

  absl::flat_hash_map<Node*, LeafTypeTree<Node*>> result;
  for (Node* node : f->nodes()) {
    result.insert({node, visitor.ConsumeValue(node)});
  }
  XLS_VLOG_LINES(3, ToString(result));
  return result;
}

std::string ToString(const TokenProvenance& provenance) {
  std::vector<std::string> lines;
  FunctionBase* f = provenance.begin()->first->function_base();
  lines.push_back(absl::StrFormat("TokenProvenance for `%s`", f->name()));
  for (Node* node : f->nodes()) {
    if (!provenance.contains(node)) {
      continue;
    }
    lines.push_back(absl::StrFormat(
        "  %s : %s", node->GetName(), provenance.at(node).ToString([](Node* n) {
          return n == nullptr ? "(nullptr)" : n->GetName();
        })));
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace xls
