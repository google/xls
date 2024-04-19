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

#ifndef XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_IMPLEMENTATION_H_
#define XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_IMPLEMENTATION_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/integrator/integration_algorithms/integration_algorithm.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/contrib/integrator/ir_integrator.h"
#include "xls/ir/verifier.h"

namespace xls {

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
  } else {
    return integration_function->MergeNodes(move.node, move.merge_node);
  }
}

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_INTEGRATION_ALGORITHMS_INTEGRATION_ALGORITHM_IMPLEMENTATION_H_
