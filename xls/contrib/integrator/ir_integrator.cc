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

absl::StatusOr<std::unique_ptr<IntegrationFunction>>
IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
    Package* package, absl::Span<const Function* const> source_functions) {
  // Create integration function object.
  std::unique_ptr<IntegrationFunction> integration_function =
      absl::make_unique<IntegrationFunction>();
  integration_function->package_ = package;

  // Create ir function.
  // TODO(jbaileyhandle): Make function name an optional argument.
  static int64 integration_functions_count = 0;
  std::string function_name = std::string("IntegrationFunction") +
                              std::to_string(integration_functions_count++);
  integration_function->function_ =
      absl::make_unique<Function>(function_name, package);

  // Package source function parameters as tuple parameters to integration
  // function.
  for (const auto* source_func : source_functions) {
    // Add tuple paramter for source function.
    std::vector<Type*> arg_types;
    for (const Node* param : source_func->params()) {
      arg_types.push_back(param->GetType());
    }
    Type* args_tuple_type = package->GetTupleType(arg_types);
    std::string tuple_name = source_func->name() + std::string("ParamTuple");
    XLS_ASSIGN_OR_RETURN(
        Node * args_tuple,
        integration_function->function_->MakeNodeWithName<Param>(
            /*loc=*/std::nullopt, tuple_name, args_tuple_type));

    // Add TupleIndex nodes inside function to unpack tuple parameter.
    int64 paramter_index = 0;
    for (const Node* param : source_func->params()) {
      XLS_ASSIGN_OR_RETURN(
          Node * tuple_index,
          integration_function->function_->MakeNode<TupleIndex>(
              /*loc=*/std::nullopt, args_tuple, paramter_index));
      XLS_RETURN_IF_ERROR(
          integration_function->SetNodeMapping(param, tuple_index));
      paramter_index++;
    }
  }

  return std::move(integration_function);
}

absl::Status IntegrationFunction::SetNodeMapping(const Node* source,
                                                 Node* map_target) {
  // Validate map pairing.
  if (source == map_target) {
    return absl::InternalError("Tried to map a node to itself");
  }
  if (!IntegrationFunctionOwnsNode(map_target)) {
    return absl::InternalError(
        "Tried to map to a node not owned by the integration function");
  }
  // TODO(jbaileyhandle): Reasonable assumption for short-term use cases.  May
  // be worth relaxing to enable some optimizations of some sort?
  if (IntegrationFunctionOwnsNode(source) && !IsMappingTarget(source)) {
    return absl::InternalError(
        "Tried to map an integration function node that is not itself a "
        "mapping target");
  }

  // 'original' is itself a member of the integrated function.
  if (IntegrationFunctionOwnsNode(source)) {
    absl::flat_hash_set<const Node*>& nodes_that_map_to_source =
        integrated_node_to_original_nodes_map_.at(source);

    // Nodes that previously mapped to original now map to map_target.
    for (const Node* original_node : nodes_that_map_to_source) {
      integrated_node_to_original_nodes_map_[map_target].insert(original_node);
      XLS_RET_CHECK(HasMapping(original_node));
      original_node_to_integrated_node_map_[original_node] = map_target;
    }

    // No nodes map to source anymore.
    integrated_node_to_original_nodes_map_.erase(source);

    // 'source' is an external node.
  } else {
    original_node_to_integrated_node_map_[source] = map_target;
    integrated_node_to_original_nodes_map_[map_target].insert(source);
  }

  return absl::OkStatus();
}

absl::StatusOr<Node*> IntegrationFunction::GetNodeMapping(
    const Node* original) const {
  XLS_RET_CHECK(!IntegrationFunctionOwnsNode(original));
  if (!HasMapping(original)) {
    return absl::InternalError("No mapping found for original node");
  }
  return original_node_to_integrated_node_map_.at(original);
}

absl::StatusOr<const absl::flat_hash_set<const Node*>*>
IntegrationFunction::GetNodesMappedToNode(const Node* map_target) const {
  XLS_RET_CHECK(IntegrationFunctionOwnsNode(map_target));
  if (!IsMappingTarget(map_target)) {
    return absl::InternalError("No mappings found for map target node");
  }
  return &integrated_node_to_original_nodes_map_.at(map_target);
}

bool IntegrationFunction::HasMapping(const Node* node) const {
  return original_node_to_integrated_node_map_.contains(node);
}

bool IntegrationFunction::IsMappingTarget(const Node* node) const {
  return integrated_node_to_original_nodes_map_.find(node) !=
         integrated_node_to_original_nodes_map_.end();
}

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
