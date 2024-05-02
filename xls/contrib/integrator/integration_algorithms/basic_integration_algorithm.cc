// Copyright 2020 Google LLC
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
// limitations under the License

#include "xls/contrib/integrator/integration_algorithms/basic_integration_algorithm.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls {

void BasicIntegrationAlgorithm::EnqueueNodeIfReady(Node* node) {
  if (!queued_nodes_.contains(node) &&
      integration_function_->AllOperandsHaveMapping(node)) {
    ready_nodes_.push_back(node);
    queued_nodes_.insert(node);
  }
}

absl::Status BasicIntegrationAlgorithm::Initialize() {
  // Make integration function.
  XLS_ASSIGN_OR_RETURN(integration_function_, NewIntegrationFunction());

  // ID initial nodes with all operands ready.
  for (const Function* func : source_functions_) {
    for (Node* node : func->nodes()) {
      if (node->op() != Op::kParam) {
        EnqueueNodeIfReady(node);
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<IntegrationFunction>>
BasicIntegrationAlgorithm::Run() {
  while (!ready_nodes_.empty()) {
    std::optional<BasicIntegrationMove> move;
    for (auto node_itr = ready_nodes_.begin(); node_itr != ready_nodes_.end();
         ++node_itr) {
      // Check insertion cost.
      XLS_ASSIGN_OR_RETURN(int64_t insert_cost,
                           integration_function_->GetInsertNodeCost(*node_itr));
      if (!move.has_value() || insert_cost < move.value().cost) {
        move = MakeInsertMove(node_itr, insert_cost);
      }

      // Check merge cost.
      for (Node* internal_node : integration_function_->function()->nodes()) {
        // TODO(jbaileyhandle): Relax this requirement so that
        // it only applies to integration-generated muxes.
        if (!integration_function_->IsMappingTarget(internal_node)) {
          continue;
        }

        // Check if mergeable
        XLS_ASSIGN_OR_RETURN(
            std::optional<int64_t> merge_cost,
            integration_function_->GetMergeNodesCost(*node_itr, internal_node));
        if (!merge_cost.has_value()) {
          continue;
        }

        // Check if lowest cost.
        XLS_RET_CHECK(move.has_value());
        if (merge_cost < move.value().cost) {
          move = MakeMergeMove(node_itr, internal_node, merge_cost.value());
        }
      }
    }

    // Execute lowest-cost move.
    XLS_RET_CHECK(move.has_value());
    XLS_RETURN_IF_ERROR(
        ExecuteMove(integration_function_.get(), move.value()).status());

    // Update ready_nodes_.
    ready_nodes_.erase(move.value().node_itr);
    for (Node* user : move.value().node->users()) {
      EnqueueNodeIfReady(user);
    }
  }

  // Finalize.
  XLS_RETURN_IF_ERROR(integration_function_->MakeTupleReturnValue().status());
  return std::move(integration_function_);
}

// Explicit instantiation.
template class IntegrationAlgorithm<BasicIntegrationAlgorithm>;

}  // namespace xls
