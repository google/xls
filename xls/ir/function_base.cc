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

#include "xls/ir/function_base.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

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

absl::StatusOr<int64_t> FunctionBase::GetParamIndex(Param* param) const {
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
  return absl::NotFoundError(
      absl::StrFormat("GetNode(%s) failed.", standard_node_name));
}

absl::Status FunctionBase::RemoveNode(Node* node) {
  XLS_RET_CHECK(node->users().empty());
  XLS_RET_CHECK(!HasImplicitUse(node));
  std::vector<Node*> unique_operands;
  for (Node* operand : node->operands()) {
    if (!absl::c_linear_search(unique_operands, operand)) {
      unique_operands.push_back(operand);
    }
  }
  for (Node* operand : unique_operands) {
    operand->RemoveUser(node);
  }
  if (node->Is<Param>()) {
    params_.erase(std::remove(params_.begin(), params_.end(), node),
                  params_.end());
  }
  auto node_it = node_iterators_.find(node);
  XLS_RET_CHECK(node_it != node_iterators_.end());
  nodes_.erase(node_it->second);
  node_iterators_.erase(node_it);
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

bool FunctionBase::IsFunction() const {
  return dynamic_cast<const Function*>(this) != nullptr;
}

bool FunctionBase::IsProc() const {
  return dynamic_cast<const Proc*>(this) != nullptr;
}

bool FunctionBase::IsBlock() const {
  return dynamic_cast<const Block*>(this) != nullptr;
}

Function* FunctionBase::AsFunctionOrDie() {
  XLS_CHECK(IsFunction());
  return down_cast<Function*>(this);
}

Proc* FunctionBase::AsProcOrDie() {
  XLS_CHECK(IsProc());
  return down_cast<Proc*>(this);
}

Block* FunctionBase::AsBlockOrDie() {
  XLS_CHECK(IsBlock());
  return down_cast<Block*>(this);
}

std::ostream& operator<<(std::ostream& os, const FunctionBase& function) {
  os << function.DumpIr();
  return os;
}

}  // namespace xls
