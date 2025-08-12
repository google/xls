// Copyright 2025 The XLS Authors
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

#ifndef XLS_PASSES_NODE_SOURCE_ANALYSIS_H_
#define XLS_PASSES_NODE_SOURCE_ANALYSIS_H_

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/node.h"
#include "xls/ir/type.h"
#include "xls/passes/dataflow_visitor.h"

namespace xls {

// Data-structure describing the source of leaf element of a node in the
// graph. If the source cannot be determined statically then the source of the
// leaf element is itself. Example NodeSources after dataflow analysis
//
//   x: u32 = param(...)          // NodeSource(x, {})
//   y: u32 = param(...)          // NodeSource(y, {})
//   z: (u32, u32) = param(...)   // (NodeSource(z, {0}), NodeSource(z, {1}))
//   a: u32 = identity(x)         // NodeSource(x, {})
//   b: u32 = tuple_index(z, 1)   // NodeSource(z, {1})
//   c: u32 = sel(..., {x, y})    // NodeSource(c, {})
//   d: u32 = sel(..., {x, x})    // NodeSource(x, {})
class NodeSource {
 public:
  NodeSource() = default;
  NodeSource(Node* node, std::vector<int64_t> tree_index)
      : node_(node), tree_index_(std::move(tree_index)) {}

  Node* node() const { return node_; }
  absl::Span<const int64_t> tree_index() const { return tree_index_; }

  std::string ToString() const {
    if (tree_index().empty()) {
      return node()->GetName();
    }
    return absl::StrFormat("%s{%s}", node()->GetName(),
                           absl::StrJoin(tree_index(), ","));
  }

  friend bool operator==(const NodeSource&, const NodeSource&) = default;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NodeSource& source) {
    absl::Format(&sink, "%s", source.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const NodeSource& ns) {
    return H::combine(std::move(h), ns.node(), ns.tree_index());
  }

 private:
  Node* node_;
  std::vector<int64_t> tree_index_;
};

class NodeSourceDataflowVisitor : public DataflowVisitor<NodeSource> {
 public:
  absl::Status DefaultHandler(Node* node) final {
    LeafTypeTree<NodeSource> result(node->GetType());
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachIndex(
        result.AsMutableView(), [&](Type* element_type, NodeSource& element,
                                    absl::Span<const int64_t> index) {
          element = NodeSource(node, std::vector(index.begin(), index.end()));
          return absl::OkStatus();
        }));
    return SetValue(node, std::move(result));
  }

 protected:
  absl::StatusOr<NodeSource> JoinElements(
      Type* element_type, absl::Span<const NodeSource* const> data_sources,
      absl::Span<const LeafTypeTreeView<NodeSource>> control_sources,
      Node* node, absl::Span<const int64_t> index) final {
    if (std::all_of(
            data_sources.begin(), data_sources.end(),
            [&](const NodeSource* n) { return *n == *data_sources.front(); })) {
      return *data_sources.front();
    }
    return NodeSource(node, std::vector(index.begin(), index.end()));
  }
};

}  // namespace xls

#endif  // XLS_PASSES_NODE_SOURCE_ANALYSIS_H_
