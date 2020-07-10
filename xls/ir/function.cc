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

#include "xls/ir/function.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"

using absl::StrAppend;

namespace xls {

std::string Function::DumpIr(bool recursive) const {
  std::string nested_funcs = "";
  std::string res = "fn " + name() + "(";
  std::vector<std::string> param_strings;
  for (Param* param : params_) {
    param_strings.push_back(param->name() + ": " +
                            param->GetType()->ToString());
  }
  StrAppend(&res, absl::StrJoin(param_strings, ", "));
  StrAppend(&res, ") -> ");

  if (return_value() != nullptr) {
    StrAppend(&res, return_value()->GetType()->ToString());
  }
  StrAppend(&res, " {\n");

  for (Node* node : TopoSort(const_cast<Function*>(this))) {
    if (node->op() == OP_PARAM && node == return_value()) {
      absl::StrAppendFormat(&res, "  ret param.%d: %s = param(name=%s)\n",
                            node->id(), node->GetType()->ToString(),
                            node->As<Param>()->name());
      continue;
    }
    if (node->op() == OP_PARAM) {
      continue;  // Already accounted for in the signature.
    }
    if (recursive && (node->op() == OP_COUNTED_FOR)) {
      nested_funcs += node->As<CountedFor>()->body()->DumpIr() + "\n";
    }
    if (recursive && (node->op() == OP_MAP)) {
      nested_funcs += node->As<Map>()->to_apply()->DumpIr() + "\n";
    }
    if (recursive && (node->op() == OP_INVOKE)) {
      nested_funcs += node->As<Invoke>()->to_apply()->DumpIr() + "\n";
    }
    StrAppend(&res, "  ", node == return_value() ? "ret " : "",
              node->ToString(), "\n");
  }

  StrAppend(&res, "}\n");
  return nested_funcs + res;
}

FunctionType* Function::GetType() {
  std::vector<Type*> arg_types;
  for (Param* param : params()) {
    arg_types.push_back(param->GetType());
  }
  XLS_CHECK(return_value() != nullptr);
  return package_->GetFunctionType(arg_types, return_value()->GetType());
}

xabsl::StatusOr<Param*> Function::GetParamByName(
    absl::string_view param_name) const {
  for (Param* param : params()) {
    if (param->name() == param_name) {
      return param;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Function '%s' does not have a paramater named '%s'",
                      name(), param_name));
}

xabsl::StatusOr<int64> Function::GetParamIndex(Param* param) const {
  auto it = std::find(params_.begin(), params_.end(), param);
  if (it == params_.end()) {
    return absl::InvalidArgumentError(
        "Given param is not a member of this function: " + param->ToString());
  }
  return std::distance(params_.begin(), it);
}

xabsl::StatusOr<Node*> Function::GetNode(absl::string_view standard_node_name) {
  for (Node* node : nodes()) {
    if (node->GetName() == standard_node_name) {
      return node;
    }
  }
  for (auto& param : params()) {
    if (param->As<Param>()->name() == standard_node_name) {
      return param;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("GetNode(%s) failed.", standard_node_name));
}

absl::Status Function::RemoveNode(Node* node, bool remove_param_ok) {
  XLS_RET_CHECK(node->users().empty());
  XLS_RET_CHECK_NE(node, return_value());
  if (node->Is<Param>()) {
    XLS_RET_CHECK(remove_param_ok)
        << "Attempting to remove parameter when !remove_param_ok: " << *node;
  }
  std::vector<Node*> unique_operands;
  for (Node* operand : node->operands()) {
    if (!absl::c_linear_search(unique_operands, operand)) {
      unique_operands.push_back(operand);
    }
  }
  for (Node* operand : unique_operands) {
    operand->RemoveUser(node);
  }
  auto node_it = node_iterators_.find(node);
  XLS_RET_CHECK(node_it != node_iterators_.end());
  nodes_.erase(node_it->second);
  node_iterators_.erase(node_it);
  if (remove_param_ok) {
    params_.erase(std::remove(params_.begin(), params_.end(), node),
                  params_.end());
  }

  return absl::OkStatus();
}

absl::Status Function::Accept(DfsVisitor* visitor) {
  for (Node* node : nodes()) {
    if (node->users().empty()) {
      XLS_RETURN_IF_ERROR(node->Accept(visitor));
    }
  }
  return absl::OkStatus();
}

xabsl::StatusOr<Function*> Function::Clone(absl::string_view new_name) const {
  absl::flat_hash_map<Node*, Node*> original_to_clone;
  Function* cloned_function =
      package()->AddFunction(absl::make_unique<Function>(new_name, package()));
  for (Node* node : TopoSort(const_cast<Function*>(this))) {
    std::vector<Node*> cloned_operands;
    for (Node* operand : node->operands()) {
      cloned_operands.push_back(original_to_clone.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(original_to_clone[node],
                         node->Clone(cloned_operands, cloned_function));
  }
  cloned_function->set_return_value(original_to_clone.at(return_value()));
  return cloned_function;
}

// Helper function for IsDefinitelyEqualTo. Recursively compares 'node' and
// 'other_node' and their operands using Node::IsDefinitelyEqualTo.
// 'matched_pairs' is used to memoize the result of the comparison.
static bool IsEqualRecurse(
    const Node* node, const Node* other_node,
    absl::flat_hash_map<const Node*, const Node*>* matched_pairs) {
  auto it = matched_pairs->find(node);
  if (it != matched_pairs->end()) {
    return it->second == other_node;
  }

  if (!node->IsDefinitelyEqualTo(other_node)) {
    XLS_VLOG(2) << absl::StrFormat(
        "Function %s != %s: node %s != %s", node->function()->name(),
        other_node->function()->name(), node->GetName(), other_node->GetName());
    return false;
  }

  for (int64 i = 0; i < node->operand_count(); ++i) {
    if (!IsEqualRecurse(node->operand(i), other_node->operand(i),
                        matched_pairs)) {
      return false;
    }
  }
  (*matched_pairs)[node] = other_node;
  return true;
}

bool Function::IsDefinitelyEqualTo(const Function* other) const {
  if (this == other) {
    XLS_VLOG(2) << absl::StrFormat("Function %s == %s: same pointer", name(),
                                   other->name());
    return true;
  }

  // Must have the types of parameters in the same order.
  if (params().size() != other->params().size()) {
    XLS_VLOG(2) << absl::StrFormat(
        "Function %s != %s: different number of parameters (%d vs %d)", name(),
        other->name(), params().size(), other->params().size());
    return false;
  }

  absl::flat_hash_map<const Node*, const Node*> matched_pairs;
  for (int64 i = 0; i < params().size(); ++i) {
    // All we care about is the type (not the name) of the parameter so don't
    // use Param::IsDefinitelyEqualTo.
    if (!param(i)->GetType()->IsEqualTo(other->param(i)->GetType())) {
      XLS_VLOG(2) << absl::StrFormat(
          "Function %s != %s: type of parameter %d not the same (%s vs %s)",
          name(), other->name(), i, param(i)->GetType()->ToString(),
          other->param(i)->GetType()->ToString());
      return false;
    }
    matched_pairs[param(i)] = other->param(i);
  }

  bool result =
      IsEqualRecurse(return_value(), other->return_value(), &matched_pairs);
  XLS_VLOG_IF(2, result) << absl::StrFormat("Function %s is equal to %s",
                                            name(), other->name());
  return result;
}

std::ostream& operator<<(std::ostream& os, const Function& function) {
  os << function.DumpIr();
  return os;
}

}  // namespace xls
