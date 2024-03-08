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
// limitations under the License.
// A abstract class for an algorithm to integrate multiple functions.

#ifndef XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_BASIC_INTEGRATION_ALGORITHM_H_
#define XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_BASIC_INTEGRATION_ALGORITHM_H_

#include <list>
#include <memory>

#include "xls/contrib/integrator/integration_algorithms/integration_algorithm.h"

namespace xls {

// A naive merging algorithm.  A node is eligible to be added to the
// integration function when all of its operands have already been added.
// At each step, adds the eligible node for which the cost of adding it to
// the function (either by inserting or merging with any integeration function
// node) is the lowest.
class BasicIntegrationAlgorithm
    : public IntegrationAlgorithm<BasicIntegrationAlgorithm> {
 public:
  BasicIntegrationAlgorithm(const BasicIntegrationAlgorithm& other) = delete;
  void operator=(const BasicIntegrationAlgorithm& other) = delete;

 private:
  // IntegrationAlgorithm::IntegrateFunctions needs to be able to call derived
  // class constructor.
  friend class IntegrationAlgorithm;

  BasicIntegrationAlgorithm(
      Package* package, absl::Span<const Function* const> source_functions,
      const IntegrationOptions& options = IntegrationOptions())
      : IntegrationAlgorithm(package, source_functions, options) {}

  // Represents a possible modification to the integration function.
  struct BasicIntegrationMove : IntegrationMove {
    std::list<Node*>::iterator node_itr;
  };

  // Make a BasicIntegrationMove for an insert.
  inline BasicIntegrationMove MakeInsertMove(
      std::list<Node*>::iterator node_itr, int64_t cost) {
    return BasicIntegrationMove{{.node = *node_itr,
                                 .move_type = IntegrationMoveType::kInsert,
                                 .cost = cost},
                                node_itr};
  }

  // Make a BasicIntegrationMove for a merge.
  inline BasicIntegrationMove MakeMergeMove(std::list<Node*>::iterator node_itr,
                                            Node* merge_node, int64_t cost) {
    return BasicIntegrationMove{{.node = *node_itr,
                                 .move_type = IntegrationMoveType::kMerge,
                                 .merge_node = merge_node,
                                 .cost = cost},
                                node_itr};
  }

  // Initialize member fields.
  absl::Status Initialize() override;

  // Returns a function that integrates the functions in source_functions_.
  // Runs after Initialize.
  absl::StatusOr<std::unique_ptr<IntegrationFunction>> Run() override;

  // Get the IntegrationOptions::Algorithm value corresponding to
  // this algoirthm.
  IntegrationOptions::Algorithm get_corresponding_algorithm_option() override {
    return IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm;
  }

  // Queue node for processing if all its operands are mapped
  // and node has not already been queued for processing.
  void EnqueueNodeIfReady(Node* node);

  // Track nodes for which all operands are already mapped and
  // are ready to be added to the integration_function_
  std::list<Node*> ready_nodes_;

  // Track all nodes that have ever been inserted into 'ready_nodes_'.
  absl::flat_hash_set<Node*> queued_nodes_;

  // Function combining the source functions.
  std::unique_ptr<IntegrationFunction> integration_function_;
};

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_BASIC_INTEGRATION_ALGORITHM_H_
