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

#include "xls/contrib/integrator/integration_builder.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/integrator/integration_algorithms/basic_integration_algorithm.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/contrib/integrator/ir_integrator.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xls {

absl::StatusOr<Function*> IntegrationBuilder::CloneFunctionRecursive(
    const Function* function,
    absl::flat_hash_map<const Function*, Function*>* call_remapping) {
  // Collect callee functions.
  std::vector<const Function*> callee_funcs;
  for (const Node* node : function->nodes()) {
    switch (node->op()) {
      case Op::kCountedFor:
        callee_funcs.push_back(node->As<CountedFor>()->body());
        break;
      case Op::kMap:
        callee_funcs.push_back(node->As<Map>()->to_apply());
        break;
      case Op::kInvoke:
        callee_funcs.push_back(node->As<Invoke>()->to_apply());
        break;
      default:
        break;
    }
  }

  // Clone and call_remapping callees.
  for (const Function* callee : callee_funcs) {
    if (!call_remapping->contains(callee)) {
      XLS_ASSIGN_OR_RETURN(Function * callee_clone,
                           CloneFunctionRecursive(callee, call_remapping));
      (*call_remapping)[callee] = callee_clone;
    }
  }

  std::string clone_name =
      function_name_uniquer_.GetSanitizedUniqueName(function->name());
  return function->Clone(clone_name, package_.get(), *call_remapping);
}

absl::Status IntegrationBuilder::CopySourcesToIntegrationPackage() {
  source_functions_.reserve(original_package_source_functions_.size());
  for (const Function* source : original_package_source_functions_) {
    absl::flat_hash_map<const Function*, Function*> call_remapping;
    XLS_ASSIGN_OR_RETURN(Function * clone_func,
                         CloneFunctionRecursive(source, &call_remapping));
    source_functions_.push_back(clone_func);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<IntegrationBuilder>> IntegrationBuilder::Build(
    absl::Span<const Function* const> input_functions,
    const IntegrationOptions& options) {
  XLS_RET_CHECK_GT(input_functions.size(), 1);

  // Build builder.
  auto builder =
      absl::WrapUnique(new IntegrationBuilder(input_functions, options));

  // Add sources to common package.
  XLS_RETURN_IF_ERROR(builder->CopySourcesToIntegrationPackage());

  // Integrate the functions.
  std::unique_ptr<IntegrationFunction> merged_func;
  switch (options.algorithm()) {
    case IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm:
      XLS_ASSIGN_OR_RETURN(merged_func,
                           BasicIntegrationAlgorithm::IntegrateFunctions(
                               builder->package(), builder->source_functions(),
                               *builder->integration_options()));
      break;
  }
  builder->set_integrated_function(std::move(merged_func));

  return std::move(builder);
}

}  // namespace xls
