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

#include "xls/contrib/integrator/ir_integrator.h"

#include "xls/ir/ir_parser.h"

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
    absl::Span<const Function* const> input_functions) {
  auto builder = absl::WrapUnique(new IntegrationBuilder(input_functions));

  // Add sources to common package.
  XLS_RETURN_IF_ERROR(builder->CopySourcesToIntegrationPackage());

  switch (builder->source_functions_.size()) {
    case 0:
      return absl::InternalError(
          "No source functions provided for integration");
    case 1:
      builder->integrated_function_ = builder->source_functions_.front();
      break;
    default:
      return absl::InternalError("Integration not yet implemented.");
  }

  return std::move(builder);
}

}  // namespace xls
