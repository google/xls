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

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/contrib/integrator/ir_integrator.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

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

}  // namespace xls

#include "xls/contrib/integrator/integration_algorithms/integration_algorithm_implementation.h"

#endif  // XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_H_
