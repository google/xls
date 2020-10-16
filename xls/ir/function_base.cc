// Copyright 2020 The XLS Authors
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

std::string FunctionBase::DumpIr(bool recursive) const {
  std::string nested_funcs = "";
  std::string res = "fn " + name() + "(";
  std::vector<std::string> param_strings;
  for (Param* param : params_) {
    param_strings.push_back(
        absl::StrFormat("%s: %s", param->name(), param->GetType()->ToString()));
  }
  StrAppend(&res, absl::StrJoin(param_strings, ", "));
  StrAppend(&res, ") -> ");

  if (return_value() != nullptr) {
    StrAppend(&res, return_value()->GetType()->ToString());
  }
  StrAppend(&res, " {\n");

  for (Node* node : TopoSort(const_cast<FunctionBase*>(this))) {
    if (node->op() == Op::kParam && node == return_value()) {
      absl::StrAppendFormat(&res, "  ret %s: %s = param(name=%s)\n",
                            node->GetName(), node->GetType()->ToString(),
                            node->As<Param>()->name());
      continue;
    }
    if (node->op() == Op::kParam) {
      continue;  // Already accounted for in the signature.
    }
    if (recursive && (node->op() == Op::kCountedFor)) {
      nested_funcs += node->As<CountedFor>()->body()->DumpIr() + "\n";
    }
    if (recursive && (node->op() == Op::kMap)) {
      nested_funcs += node->As<Map>()->to_apply()->DumpIr() + "\n";
    }
    if (recursive && (node->op() == Op::kInvoke)) {
      nested_funcs += node->As<Invoke>()->to_apply()->DumpIr() + "\n";
    }
    StrAppend(&res, "  ", node == return_value() ? "ret " : "",
              node->ToString(), "\n");
  }

  StrAppend(&res, "}\n");
  return nested_funcs + res;
}

FunctionType* FunctionBase::GetType() {
  std::vector<Type*> arg_types;
  for (Param* param : params()) {
    arg_types.push_back(param->GetType());
  }
  XLS_CHECK(return_value() != nullptr);
  return package_->GetFunctionType(arg_types, return_value()->GetType());
}

absl::StatusOr<Param*> FunctionBase::GetParamByName(
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

absl::StatusOr<int64> FunctionBase::GetParamIndex(Param* param) const {
  auto it = std::find(params_.begin(), params_.end(), param);
  if (it == params_.end()) {
    return absl::InvalidArgumentError(
        "Given param is not a member of this function: " + param->ToString());
  }
  return std::distance(params_.begin(), it);
}

absl::StatusOr<Node*> FunctionBase::GetNode(
    absl::string_view standard_node_name) {
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

absl::Status FunctionBase::RemoveNode(Node* node, bool remove_param_ok) {
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

absl::Status FunctionBase::Accept(DfsVisitor* visitor) {
  for (Node* node : nodes()) {
    if (node->users().empty()) {
      XLS_RETURN_IF_ERROR(node->Accept(visitor));
    }
  }
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const FunctionBase& function) {
  os << function.DumpIr();
  return os;
}

}  // namespace xls
