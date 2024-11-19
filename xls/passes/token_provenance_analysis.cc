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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
    case Op::kStateRead:
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

class TokenProvenanceVisitor
    : public DataflowVisitor<absl::flat_hash_set<Node*>> {
 public:
  absl::Status DefaultHandler(Node* node) override {
    LeafTypeTree<absl::flat_hash_set<Node*>> ltt(node->GetType(),
                                                 absl::flat_hash_set<Node*>());
    if (OpHasTokenProvenance(node->op())) {
      for (int64_t i = 0; i < ltt.size(); ++i) {
        if (ltt.leaf_types().at(i)->IsToken()) {
          ltt.elements().at(i) = {node};
        }
      }
    } else if (TypeHasToken(node->GetType())) {
      return TokenError(node);
    }
    return SetValue(node, std::move(ltt));
  }

 protected:
  static absl::Status TokenError(Node* node) {
    return absl::InternalError(absl::StrFormat(
        "Node type should not contain a token: %s", node->ToString()));
  }

  // Returns true if all leaf elements of each LeafTypeTree in `trees` are
  // empty.
  static bool AreAllElementsEmpty(
      absl::Span<const LeafTypeTreeView<absl::flat_hash_set<Node*>>> trees) {
    for (const LeafTypeTreeView<absl::flat_hash_set<Node*>> tree : trees) {
      for (const absl::flat_hash_set<Node*>& sources : tree.elements()) {
        if (!sources.empty()) {
          return false;
        }
      }
    }
    return true;
  }

  absl::StatusOr<absl::flat_hash_set<Node*>> JoinElements(
      Type* element_type,
      absl::Span<absl::flat_hash_set<Node*> const* const> data_sources,
      absl::Span<const LeafTypeTreeView<absl::flat_hash_set<Node*>>>
          control_sources,
      Node* node, absl::Span<const int64_t> index) const override {
    if (!AreAllElementsEmpty(control_sources)) {
      return TokenError(node);
    }
    absl::flat_hash_set<Node*> result;
    for (absl::flat_hash_set<Node*> const* const data_source : data_sources) {
      result.insert(data_source->begin(), data_source->end());
    }
    return result;
  }
};

// This variant of the TokenProvenanceVisitor keeps a list of visited nodes in
// order. This allows the topo-sorted token DAG computation to reuse the
// visitor's traversal instead of having to topo-sort.
class TokenProvenanceWithTopoSortVisitor : public TokenProvenanceVisitor {
 public:
  absl::Status DefaultHandler(Node* node) override {
    const bool result_contains_token = TypeHasToken(node->GetType());
    if (result_contains_token || OpIsSideEffecting(node->op())) {
      if (node->OpIn({Op::kParam, Op::kStateRead}) && !result_contains_token) {
        // Don't include non-token-containing state elements.
        return TokenProvenanceVisitor::DefaultHandler(node);
      }
      topo_sorted_token_nodes_.push_back(node);
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

  absl::flat_hash_map<Node*, LeafTypeTree<absl::flat_hash_set<Node*>>> result;
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
        "  %s : {%s}", node->GetName(),
        provenance.at(node).ToString(
            [](const absl::flat_hash_set<Node*>& sources) {
              std::vector<Node*> sorted_sources(sources.begin(), sources.end());
              absl::c_sort(sorted_sources, Node::NodeIdLessThan());
              return absl::StrJoin(sorted_sources, ", ");
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
          const absl::flat_hash_set<Node*>& child =
              provenance.at(operand).Get({});
          dag[node].insert(child.cbegin(), child.cend());
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
        for (Node* child : visitor.GetValue(operand).Get({})) {
          entry.predecessors.insert(child);
        }
      }
    }
    result.push_back(std::move(entry));
  }
  return result;
}

}  // namespace xls
