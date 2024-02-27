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

#include <unistd.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/passes/dataflow_visitor.h"

namespace xls {
namespace {

inline bool OpHasTokenProvenance(Op op) {
  switch (op) {
    case Op::kLiteral:
    case Op::kParam:
    case Op::kNext:
    case Op::kAssert:
    case Op::kCover:
    case Op::kTrace:
    case Op::kReceive:
    case Op::kSend:
    case Op::kAfterAll:
    case Op::kMinDelay:
      return true;
    default:
      return false;
  }
}

class TokenProvenanceVisitor : public DataflowVisitor<Node*> {
 public:
  absl::Status DefaultHandler(Node* node) override {
    LeafTypeTree<Node*> ltt(node->GetType(), nullptr);
    if (OpHasTokenProvenance(node->op())) {
      for (int64_t i = 0; i < ltt.size(); ++i) {
        if (ltt.leaf_types().at(i)->IsToken()) {
          ltt.elements().at(i) = node;
        }
      }
    } else if (TypeHasToken(node->GetType())) {
      return TokenError(node);
    }
    return SetValue(node, ltt);
  }

 protected:
  static absl::Status TokenError(Node* node) {
    return absl::InternalError(absl::StrFormat(
        "Node type should not contain a token: %s", node->ToString()));
  }

  absl::Status AccumulateDataElement(Node* const& data_element, Node* node,
                                     absl::Span<const int64_t> index,
                                     Node*& element) const override {
    // Tokens should never be joined.
    if (data_element != nullptr || element != nullptr) {
      return TokenError(node);
    }
    return absl::OkStatus();
  }

  absl::Status AccumulateControlElement(Node* const& control_element,
                                        Node* node,
                                        absl::Span<const int64_t> index,
                                        Node*& element) const override {
    // Tokens should never be joined.
    if (control_element != nullptr || element != nullptr) {
      return TokenError(node);
    }
    return absl::OkStatus();
  }
};

// This variant of the TokenProvenanceVisitor keeps a list of visited nodes in
// order. This allows the topo-sorted token DAG computation to reuse the
// visitor's traversal instead of having to topo-sort.
class TokenProvenanceWithTopoSortVisitor : public TokenProvenanceVisitor {
 public:
  absl::Status DefaultHandler(Node* node) override {
    if (OpIsSideEffecting(node->op()) || node->op() == Op::kAfterAll ||
        node->op() == Op::kMinDelay) {
      if (!(node->op() == Op::kParam && !TypeHasToken(node->GetType()))) {
        // Don't include normal state, just the proc token param.
        topo_sorted_token_nodes_.push_back(node);
      }
    }
    return TokenProvenanceVisitor::DefaultHandler(node);
  }

  absl::Span<Node* const> topo_sorted_token_nodes() const {
    return topo_sorted_token_nodes_;
  }

 private:
  std::vector<Node*> topo_sorted_token_nodes_;
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

absl::StatusOr<TokenDAG> ComputeTokenDAG(FunctionBase* f) {
  XLS_ASSIGN_OR_RETURN(TokenProvenance provenance, TokenProvenanceAnalysis(f));

  TokenDAG dag;
  for (Node* node : f->nodes()) {
    if (OpIsSideEffecting(node->op()) || node->op() == Op::kAfterAll ||
        node->op() == Op::kMinDelay) {
      for (Node* operand : node->operands()) {
        if (operand->GetType()->IsToken()) {
          Node* child = provenance.at(operand).Get({});
          if (child != nullptr) {
            dag[node].insert(child);
          }
        }
      }
    }
  }

  return dag;
}

absl::StatusOr<std::vector<NodeAndPredecessors>> ComputeTopoSortedTokenDAG(
    FunctionBase* f) {
  TokenProvenanceWithTopoSortVisitor visitor;
  XLS_RETURN_IF_ERROR(f->Accept(&visitor));

  std::vector<NodeAndPredecessors> result;

  for (Node* node : visitor.topo_sorted_token_nodes()) {
    NodeAndPredecessors entry{.node = node};
    for (Node* operand : node->operands()) {
      if (operand->GetType()->IsToken()) {
        Node* child = visitor.GetValue(operand).Get({});
        if (child != nullptr) {
          entry.predecessors.insert(child);
        }
      }
    }
    result.push_back(std::move(entry));
  }
  return result;
}

}  // namespace xls
