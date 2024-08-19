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

#ifndef XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_H_
#define XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/contrib/integrator/ir_integrator.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {

// An abstract class representing an algorithm to merge 2 or more
// xls ir Functions.
template <class AlgorithmType>
class IntegrationAlgorithm {
 public:
  IntegrationAlgorithm(const IntegrationAlgorithm& other) = delete;
  void operator=(const IntegrationAlgorithm& other) = delete;

  // Returns a function that integrates the functions in source_functions.
  static absl::StatusOr<std::unique_ptr<IntegrationFunction>>
  IntegrateFunctions(Package* package,
                     absl::Span<const Function* const> source_functions,
                     const IntegrationOptions& options = IntegrationOptions());

 protected:
  IntegrationAlgorithm(Package* package,
                       absl::Span<const Function* const> source_functions,
                       const IntegrationOptions& options = IntegrationOptions())
      : package_(package), integration_options_(options) {
    source_functions_.reserve(source_functions.size());
    for (const auto* func : source_functions) {
      source_functions_.push_back(func);
    }
  }
  virtual ~IntegrationAlgorithm() = default;

  // Get the IntegrationOptions::Algorithm value corresponding to
  // this algoirthm.
  virtual IntegrationOptions::Algorithm
  get_corresponding_algorithm_option() = 0;

  // Initialize any member fields.
  virtual absl::Status Initialize() = 0;

  // Returns a function that integrates the functions in source_functions_.
  // Runs after Initialize.
  virtual absl::StatusOr<std::unique_ptr<IntegrationFunction>> Run() = 0;

  // Represents a possible modification to an integration function.
  enum class IntegrationMoveType { kInsert, kMerge };
  struct IntegrationMove {
    Node* node;
    IntegrationMoveType move_type;
    Node* merge_node = nullptr;
    int64_t cost;
  };

  // Perform the modification to 'integration_function' described by 'move'.
  absl::StatusOr<std::vector<Node*>> ExecuteMove(
      IntegrationFunction* integration_function, const IntegrationMove& move);

  // Create and return a new IntegrationFunction.
  absl::StatusOr<std::unique_ptr<IntegrationFunction>>
  NewIntegrationFunction() {
    return IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
        package_, source_functions_, integration_options_);
  }

  std::vector<const Function*> source_functions_;
  Package* package_;
  const IntegrationOptions integration_options_;
};

// -- Implementation

template <class AlgorithmType>
absl::StatusOr<std::unique_ptr<IntegrationFunction>>
IntegrationAlgorithm<AlgorithmType>::IntegrateFunctions(
    Package* package, absl::Span<const Function* const> source_functions,
    const IntegrationOptions& options) {
  // Setup.
  XLS_RET_CHECK_GT(source_functions.size(), 1);
  AlgorithmType algorithm(package, source_functions, options);
  XLS_RETURN_IF_ERROR(algorithm.Initialize());
  XLS_RET_CHECK_EQ(algorithm.get_corresponding_algorithm_option(),
                   options.algorithm());

  // Integrate.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<IntegrationFunction> integration_func,
                       algorithm.Run());
  XLS_RETURN_IF_ERROR(VerifyFunction(integration_func->function()));
  return integration_func;
}

template <class AlgorithmType>
absl::StatusOr<std::vector<Node*>>
IntegrationAlgorithm<AlgorithmType>::ExecuteMove(
    IntegrationFunction* integration_function, const IntegrationMove& move) {
  if (move.move_type == IntegrationMoveType::kInsert) {
    XLS_ASSIGN_OR_RETURN(Node * node,
                         integration_function->InsertNode(move.node));
    return std::vector<Node*>({node});
  }
  return integration_function->MergeNodes(move.node, move.merge_node);
}

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_H_
