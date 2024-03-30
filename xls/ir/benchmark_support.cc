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

#include "xls/ir/benchmark_support.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls {
namespace benchmark_support {

absl::StatusOr<BValue> ChainReduce(
    FunctionBuilder& builder, int64_t layer_width,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy, BValue previous_layer) {
  std::vector<BValue> inputs;
  inputs.reserve(layer_width);
  inputs.push_back(previous_layer);
  for (int64_t i = 1; i < layer_width; ++i) {
    XLS_ASSIGN_OR_RETURN(BValue terminal,
                         leaf_strategy.GenerateNullaryNode(builder));
    inputs.push_back(terminal);
  }
  return interior_node_strategy.GenerateInteriorPoint(builder,
                                                      absl::MakeSpan(inputs));
}

absl::StatusOr<Function*> GenerateChain(
    Package* package, int64_t depth, int64_t num_children,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy) {
  XLS_RET_CHECK_GE(depth, 1);
  FunctionBuilder fb("ladder_tree", package);
  XLS_ASSIGN_OR_RETURN(auto result,
                       GenerateChain(fb, depth, num_children,
                                     interior_node_strategy, leaf_strategy));
  return fb.BuildWithReturnValue(result);
}

absl::StatusOr<BValue> GenerateChain(
    FunctionBuilder& fb, int64_t depth, int64_t num_children,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy) {
  XLS_ASSIGN_OR_RETURN(BValue current_root,
                       leaf_strategy.GenerateNullaryNode(fb));
  for (int64_t i = 0; i < depth; ++i) {
    XLS_ASSIGN_OR_RETURN(current_root,
                         ChainReduce(fb, num_children, interior_node_strategy,
                                     leaf_strategy, current_root));
  }
  return current_root;
}

absl::StatusOr<std::vector<BValue>> BalancedTreeReduce(
    FunctionBuilder& builder, int64_t fan_out,
    const strategy::NaryNode& interior_node_strategy,
    absl::Span<BValue> previous_layer) {
  XLS_RET_CHECK_GT(fan_out, 1);
  XLS_RET_CHECK_EQ(previous_layer.size() % fan_out, 0);
  std::vector<BValue> result;
  result.reserve(previous_layer.size() / fan_out);
  absl::Span<BValue> previous = absl::MakeSpan(previous_layer);
  for (int64_t i = 0; i < previous_layer.size(); i += fan_out) {
    XLS_ASSIGN_OR_RETURN(BValue res,
                         interior_node_strategy.GenerateInteriorPoint(
                             builder, previous.subspan(i, fan_out)));
    result.push_back(res);
  }
  return result;
}

absl::StatusOr<Function*> GenerateBalancedTree(
    Package* package, int64_t depth, int64_t fan_out,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy) {
  FunctionBuilder fb("balanced-tree", package);
  XLS_ASSIGN_OR_RETURN(
      auto result, GenerateBalancedTree(fb, depth, fan_out,
                                        interior_node_strategy, leaf_strategy));
  return fb.BuildWithReturnValue(result);
}

absl::StatusOr<BValue> GenerateBalancedTree(
    FunctionBuilder& fb, int64_t depth, int64_t fan_out,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& leaf_strategy) {
  XLS_RET_CHECK_GE(depth, 1);
  XLS_RET_CHECK_GT(fan_out, 1);
  std::vector<BValue> base;
  base.reserve(static_cast<size_t>(std::pow(fan_out, depth)));
  for (int64_t i = 0; static_cast<double>(i) < std::pow(fan_out, depth); ++i) {
    XLS_ASSIGN_OR_RETURN(BValue v, leaf_strategy.GenerateNullaryNode(fb));
    base.push_back(v);
  }
  while (base.size() != 1) {
    XLS_ASSIGN_OR_RETURN(base,
                         BalancedTreeReduce(fb, fan_out, interior_node_strategy,
                                            absl::MakeSpan(base)));
  }
  return base.front();
}

absl::StatusOr<std::vector<BValue>> AddFullyConnectedGraphLayer(
    FunctionBuilder& builder, int64_t width,
    const strategy::NaryNode& interior_node_strategy,
    absl::Span<BValue> previous_layer) {
  std::vector<BValue> result;
  result.reserve(width);
  for (int64_t i = 0; i < width; ++i) {
    XLS_ASSIGN_OR_RETURN(BValue v, interior_node_strategy.GenerateInteriorPoint(
                                       builder, previous_layer));
    result.push_back(v);
  }
  return result;
}

absl::StatusOr<Function*> GenerateFullyConnectedLayerGraph(
    Package* package, int64_t depth, int64_t width,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& terminal_node_strategy) {
  FunctionBuilder fb("lattice", package);
  XLS_ASSIGN_OR_RETURN(
      auto result,
      GenerateFullyConnectedLayerGraph(fb, depth, width, interior_node_strategy,
                                       terminal_node_strategy));
  return fb.BuildWithReturnValue(result);
}

absl::StatusOr<BValue> GenerateFullyConnectedLayerGraph(
    FunctionBuilder& fb, int64_t depth, int64_t width,
    const strategy::NaryNode& interior_node_strategy,
    const strategy::NullaryNode& terminal_node_strategy) {
  XLS_RET_CHECK_GE(depth, 1);
  XLS_RET_CHECK_GT(width, 1);
  std::vector<BValue> current_layer;
  current_layer.reserve(width);
  for (int64_t j = 0; j < width; ++j) {
    XLS_ASSIGN_OR_RETURN(BValue leaf,
                         terminal_node_strategy.GenerateNullaryNode(fb));
    current_layer.push_back(leaf);
  }
  for (int64_t i = 0; i < depth; ++i) {
    XLS_ASSIGN_OR_RETURN(current_layer, AddFullyConnectedGraphLayer(
                                            fb, width, interior_node_strategy,
                                            absl::MakeSpan(current_layer)));
  }
  // Make sure we have a single return value so there are no dead nodes.
  XLS_ASSIGN_OR_RETURN(BValue return_value,
                       interior_node_strategy.GenerateInteriorPoint(
                           fb, absl::MakeSpan(current_layer)));
  return return_value;
}
}  // namespace benchmark_support
}  // namespace xls
