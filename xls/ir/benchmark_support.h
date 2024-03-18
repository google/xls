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

#ifndef XLS_IR_BENCHMARK_SUPPORT_H_
#define XLS_IR_BENCHMARK_SUPPORT_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {
namespace benchmark_support {

namespace strategy {

// Strategy interface for generating terminal nodes of a graph.
//
// A Nullary node is a terminal node without any arguments.
class NullaryNode {
 public:
  virtual ~NullaryNode() = default;
  // Create or return a single value which should be used as a terminal node.
  virtual absl::StatusOr<BValue> GenerateNullaryNode(
      FunctionBuilder& builder) const = 0;
};

// A leaf strategy for any location type that always returns a new
// 'builder.Literal(...)'
class DistinctLiteral final : public NullaryNode {
 public:
  explicit DistinctLiteral(Bits value = UBits(42, 8))
      : value_(std::move(value)) {}

  absl::StatusOr<BValue> GenerateNullaryNode(
      FunctionBuilder& builder) const final {
    return builder.Literal(value_);
  }

 private:
  Bits value_;
};

// A leaf strategy for any location type that always returns *the same* Literal
// node.
class SharedLiteral final : public NullaryNode {
 public:
  explicit SharedLiteral(Bits value = UBits(42, 8))
      : value_(std::move(value)), inst_(std::nullopt) {}

  absl::StatusOr<BValue> GenerateNullaryNode(
      FunctionBuilder& builder) const final {
    if (inst_.has_value()) {
      return *inst_;
    }
    inst_ = builder.Literal(value_);
    return *inst_;
  }

 private:
  Bits value_;
  mutable std::optional<BValue> inst_;
};

// Strategy that determines how to create a node with given inputs in the graph.
//
// This is called to generate all non-terminal nodes in the graph.
class NaryNode {
 public:
  virtual ~NaryNode() = default;
  // Create a new node that represents the use of all the inputs.
  virtual absl::StatusOr<BValue> GenerateInteriorPoint(
      FunctionBuilder& builder, absl::Span<BValue> inputs) const = 0;
};

// Strategy which just fills a select for each node. The first input is
// taken to be the selector.
//
// Example:
// FullSelectStrategy([v1, v2, v3, vN]) ->
//   Select[selector = v1, cases = [v3, ..., vN], default = v2]
class FullSelect : public NaryNode {
 public:
  absl::StatusOr<BValue> GenerateInteriorPoint(
      FunctionBuilder& builder, absl::Span<BValue> inputs) const final {
    if (inputs.size() - 1 == 1 << inputs.front().BitCountOrDie()) {
      return builder.Select(inputs[0], inputs.subspan(1));
    }
    return builder.Select(inputs[0], inputs.subspan(2), inputs[1]);
  }
};

// Strategy which just fills the cases of a select for each node. A separate
// NullaryNodeStrategy is used to create the selector value.
//
// Example:
// FullSelectStrategy(NullaryNodeGenerator gen, [v1, v2, v3, vN]) ->
//   Select[selector = gen.Generate(), cases = [v2, ..., vN], default = v1]
class CaseSelect : public NaryNode {
 public:
  explicit CaseSelect(NullaryNode& selector_strategy)
      : selector_strategy_(selector_strategy) {}
  absl::StatusOr<BValue> GenerateInteriorPoint(
      FunctionBuilder& builder, absl::Span<BValue> inputs) const final {
    XLS_ASSIGN_OR_RETURN(BValue selector,
                         selector_strategy_.GenerateNullaryNode(builder));
    if (inputs.size() == 1 << selector.BitCountOrDie()) {
      return builder.Select(selector, inputs);
    }
    return builder.Select(selector, inputs.subspan(1), inputs[0]);
  }

 private:
  NullaryNode& selector_strategy_;
};

// Strategy which uses a binary add as each node in the graph.
//
// Example BinaryAddStrategy[v1, v2] -> Add[v1, v2]
class BinaryAdd : public NaryNode {
 public:
  absl::StatusOr<BValue> GenerateInteriorPoint(
      FunctionBuilder& builder, absl::Span<BValue> inputs) const final {
    XLS_RET_CHECK_EQ(inputs.size(), 2);
    return builder.Add(inputs[0], inputs[1]);
  }
};
}  // namespace strategy

// Generates a graph of fully connected layers of a given width and depth.
//
// This graph is made up of 'depth' layers where each node on layer 'N' takes as
// input all values on layer 'N + 1'. The lowest (depth) layer is made up of
// Nullary nodes created by the terminal_node_strategy.
//
// A single NAry node is used to create the return value so all nodes in the
// graph are reachable.
//
// For example the generated graph will look something like this:
//
// Terminal.1 = <terminal_node>
// Terminal.2 = <terminal_node>
// ...
// Terminal.<width> = <terminal_node>
// -- Layer 1
// NAryNode_1.1 = Operation[Terminal.1, Terminal.2, ..., Terminal.<width>]
// NAryNode_1.2 = Operation[Terminal.1, Terminal.2, ..., Terminal.<width>]
// ...
// NAryNode_1.<width> = Operation[Terminal.1, Terminal.2, ..., Terminal.<width>]
// -- Layer 2
// NAryNode_2.1 = Operation[NAryNode_1.1, NAryNode_1.2, ..., NAryNode_1.<width>]
// NAryNode_2.2 = Operation[NAryNode_1.1, NAryNode_1.2, ..., NAryNode_1.<width>]
// ...
// NAryNode_2.<width> =
//                Operation[NAryNode_1.1, NAryNode_1.2, ..., NAryNode_1.<width>]
// -- Layer 3
// ...
// -- Layer <depth>
// NAryNode_<depth>.1 =
//     Operation[NAryNode_<depth - 1>.1,
//               NAryNode_<depth - 1>.2,
//               ..., NAryNode_<depth - 1>.<width>]
// NAryNode_<depth>.2 =
//     Operation[NAryNode_<depth - 1>.1,
//               NAryNode_<depth - 1>.2,
//               ..., NAryNode_<depth - 1>.<width>]
// ...
// NAryNode_<depth>.<width> =
//     Operation[NAryNode_<depth - 1>.1,
//               NAryNode_<depth - 1>.2,
//               ..., NAryNode_<depth - 1>.<width>]
// -- Final return value
// NAryNode_RETURN =
//     Operation[NAryNode_<depth>.1,
//               NAryNode_<depth>.2,
//               ..., NAryNode_<depth>.<width>]
absl::StatusOr<Function*> GenerateFullyConnectedLayerGraph(
    Package* package, int64_t depth, int64_t width,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& terminal_node_strategy);

// Create a layer of a fully-connected graph.
//
// Makes 'width' nodes which each consume all the elements in the previous
// layer.
absl::StatusOr<std::vector<BValue>> AddFullyConnectedGraphLayer(
    FunctionBuilder& builder, int64_t width,
    const strategy::NaryNode& interior_node_strategy,
    absl::Span<BValue> previous_layer);

// Generate a balanced tree of given depth with each node having the given fan
// out. Use the interior_node_strategy to generate interior nodes and
// leaf_strategy to generate leaves.
//
// This generates graphs similar to:
// (+ (+ (+ ... ...) (+ ... ...)) (+ (+ ... ...) (+ ... ...)))
//
// For example for n-ary plus and nullary being the literal '1' with depth 2 and
// width 3 it is:
// (+ (+ (+ 1 1 1)
//       (+ 1 1 1)
//       (+ 1 1 1))
//    (+ (+ 1 1 1)
//       (+ 1 1 1)
//       (+ 1 1 1))
//    (+ (+ 1 1 1)
//       (+ 1 1 1)
//       (+ 1 1 1)))
absl::StatusOr<Function*> GenerateBalancedTree(
    Package* package, int64_t depth, int64_t fan_out,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy);

// Reduce a balanced tree closer to a unitary value.
//
// This takes N-inputs and returns N/fan_out results of calling the
// interior_node_strategy on each fan_out grouping.
//
// The length of previous_layer must be a multiple of fan_out.
//
// Example
// t1 = A
// t2 = B
// t3 = C
// t4 = D
//   REDUCE
// T5 = Combine[A, B]
// T6 = Combine[C, D]
absl::StatusOr<std::vector<BValue>> BalancedTreeReduce(
    FunctionBuilder& builder, int64_t fan_out,
    const strategy::NaryNode& interior_node_strategy,
    absl::Span<BValue> previous_layer);

// Generate a 'chain' of given depth with each node having the given fan
// out.
//
// A chain tree is a tree with one argument that continues the tree structure
// and all other arguments being nullary/terminal/leaf nodes.
//
// Use the interior_node_strategy to generate interior nodes and leaf_strategy
// to generate leaves. The first 'input' is the one that continues.
//
// This generates graphs similar to
// (+
//   (+
//     (+
//       (+ ... <nullary-node>)
//       <nullary-node>)
//     <nullary-node>)
//   <nullary-node>)
absl::StatusOr<Function*> GenerateChain(
    Package* package, int64_t depth, int64_t num_children,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy);

// Add a new layer onto a chain graph.
//
// previous_layer is what was the head of the chain. This adds a new layer with
// 'layer_width' 'nullary-nodes' and returns it.
//
// For example if we have:
// ...
// previous_layer = <something>
//
// then AddChainLayer(width = 3, ...)
//
// previous_layer = <something>
// term_2 = <nullary_node>
// term_3 = <nullary_node>
// current_layer = Combine[previous_layer, term_2, term_3]
absl::StatusOr<BValue> ChainReduce(
    FunctionBuilder& builder, int64_t layer_width,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy, BValue previous_layer);

}  // namespace benchmark_support
}  // namespace xls

#endif  // XLS_IR_BENCHMARK_SUPPORT_H_
