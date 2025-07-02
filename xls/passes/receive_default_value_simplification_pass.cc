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

#include "xls/passes/receive_default_value_simplification_pass.h"

#include <optional>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

struct ReceiveData {
  Receive* receive;
  Node* data;
};

// Check if the node is a binary select or a binary priority select
bool IsBinarySelectOrPrioritySelect(Node *node) {
  // Handle Select
  if (node->Is<Select>()) {
    Select *select = node->As<Select>();
    return IsBinarySelect(select);
  }

  // Handle PrioritySelect
  if (node->Is<PrioritySelect>()) {
    PrioritySelect *select = node->As<PrioritySelect>();
    return IsBinaryPrioritySelect(select);
  }

  return false;
}

// Matches node against the data of a receive node. Returns the receive
// operation and the data value if the match succeeds.
std::optional<ReceiveData> MatchReceiveDataValue(Node* node) {
  if (node->Is<TupleIndex>() && node->As<TupleIndex>()->index() == 1 &&
      node->operand(0)->Is<Receive>()) {
    return ReceiveData{.receive = node->operand(0)->As<Receive>(),
                       .data = node};
  }
  return std::nullopt;
}

bool IsValidBitOfNonblockingReceive(Node* node) {
  return node->Is<TupleIndex>() && node->As<TupleIndex>()->index() == 2 &&
         node->operand(0)->Is<Receive>() &&
         !node->operand(0)->As<Receive>()->is_blocking();
}

// Matches `node` against a useless binary select between the data value of a
// conditional or non-blocking receive node and a constant 0. There are two
// patterns. For the conditional case:
//
//                            predicate
//                               |
//               receive <-------+
//                  |            |
//                  |            |
//  constant(0)  tuple_index(1)  |
//          \    /               |
//           \  /                |
//          select <-------------+
//
// For the non-blocking case:
//
//                   receive
//                      |
//                      +---------------+
//                      |               |
//  constant(0)  tuple_index(1)   tuple_index(2)
//          \    /                      |
//           \  /                       |
//          select <--------------------+
//
// In these case, the select is equivalent to the data value of the receive.
std::optional<ReceiveData> MatchUselessSelectAfterReceive(
    Node *select, Node *selector, Node *on_true_input, Node *on_false_input,
    QueryEngine &query_engine) {
  std::optional<ReceiveData> receive_data =
      MatchReceiveDataValue(on_true_input);
  if (!receive_data.has_value()) {
    return std::nullopt;
  }

  std::optional<Value> constant = query_engine.KnownValue(on_false_input);
  if (!constant.has_value() || !constant->IsAllZeros()) {
    return std::nullopt;
  }

  // Test for the conditional receive case.
  if (receive_data->receive->predicate().has_value() &&
      receive_data->receive->predicate().value() == selector) {
    return receive_data;
  }

  // Test for the non-blocking receive case.
  if (!receive_data->receive->is_blocking() &&
      IsValidBitOfNonblockingReceive(selector)) {
    return receive_data;
  }

  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> ReceiveDefaultValueSimplificationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  StatelessQueryEngine query_engine;

  bool changed = false;
  for (Node* node : context.TopoSort(proc)) {
    if (!IsBinarySelectOrPrioritySelect(node)) {
      continue;
    }
    std::optional<ReceiveData> receive_data;
    if (node->Is<Select>()) {
      Select *select = node->As<Select>();
      receive_data = MatchUselessSelectAfterReceive(
          node, select->selector(), select->get_case(1), select->get_case(0),
          query_engine);

    } else {
      PrioritySelect *select = node->As<PrioritySelect>();
      receive_data = MatchUselessSelectAfterReceive(
          node, select->selector(), select->get_case(0),
          select->default_value(), query_engine);
    }
    if (receive_data.has_value()) {
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(receive_data->data));
      changed = true;
    }
  }

  return changed;
}

}  // namespace xls
