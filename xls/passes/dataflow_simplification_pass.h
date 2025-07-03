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

#ifndef XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// An optimization which uses a lattice-based dataflow analysis to find
// equivalent nodes in the graph and replace them with a simpler form. The
// analysis traces through tuples, arrays, and select operations. Optimizations
// which can be performed by this pass:
//
//    tuple_index(tuple(x, y), index=1)  =>  y
//
//    select(selector, {z, z})  =>  z
//
//    array_index(array_update(A, x, index={42}), index={42})  =>  x
class DataflowSimplificationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "dataflow";
  explicit DataflowSimplificationPass()
      : OptimizationFunctionBasePass(kName, "Dataflow Optimization") {}
  ~DataflowSimplificationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

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
  absl::Status DefaultHandler(Node* node) override {
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
      Node* node, absl::Span<const int64_t> index) override {
    if (std::all_of(
            data_sources.begin(), data_sources.end(),
            [&](const NodeSource* n) { return *n == *data_sources.front(); })) {
      return *data_sources.front();
    }
    return NodeSource(node, std::vector(index.begin(), index.end()));
  }
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_
