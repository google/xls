// Copyright 2021 The XLS Authors
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

#include "xls/ir/block.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"

namespace xls {

std::string Block::DumpIr(bool recursive) const {
  // TODO(meheff): Remove recursive argument. Recursively dumping multiple
  // functions should be a method at the Package level, not the function/proc
  // level.
  XLS_CHECK(!recursive);

  std::string res = absl::StrFormat("block %s {\n", name());

  for (Register* reg : GetRegisters()) {
    if (reg->reset_value().has_value()) {
      absl::StrAppendFormat(&res, "  reg %s(%s, reset_value=%s)\n", reg->name(),
                            reg->type()->ToString(),
                            reg->reset_value().value().ToHumanString());
    } else {
      absl::StrAppendFormat(&res, "  reg %s(%s)\n", reg->name(),
                            reg->type()->ToString());
    }
  }

  for (Node* node : TopoSort(const_cast<Block*>(this))) {
    absl::StrAppend(&res, "  ", node->ToString(), "\n");
  }
  absl::StrAppend(&res, "}\n");
  return res;
}

absl::StatusOr<InputPort*> Block::AddInputPort(
    absl::string_view name, Type* type, absl::optional<SourceLocation> loc) {
  if (ports_by_name_.contains(name)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s already contains a port named %s", this->name(), name));
  }
  InputPort* port =
      AddNode(absl::make_unique<InputPort>(loc, name, type, this));
  if (name != port->GetName()) {
    // The name uniquer changed the given name of the input port to preserve
    // name uniqueness so a node with this name must already exist.
    return absl::InvalidArgumentError(
        absl::StrFormat("A node already exists with name %s", name));
  }

  ports_by_name_[name] = port;
  ports_.push_back(port);
  input_ports_.push_back(port);
  return port;
}

absl::StatusOr<OutputPort*> Block::AddOutputPort(
    absl::string_view name, Node* operand, absl::optional<SourceLocation> loc) {
  if (ports_by_name_.contains(name)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s already contains a port named %s", this->name(), name));
  }
  OutputPort* port =
      AddNode(absl::make_unique<OutputPort>(loc, operand, name, this));

  if (name != port->GetName()) {
    // The name uniquer changed the given name of the input port to preserve
    // name uniqueness so a node with this name must already exist.
    return absl::InvalidArgumentError(
        absl::StrFormat("A node already exists with name %s", name));
  }
  ports_by_name_[name] = port;
  ports_.push_back(port);
  output_ports_.push_back(port);
  return port;
}

absl::StatusOr<Register*> Block::AddRegister(
    absl::string_view name, Type* type, absl::optional<Value> reset_value) {
  if (registers_.contains(name)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Register already exists with name %s", name));
  }
  if (reset_value.has_value()) {
    if (type != package()->GetTypeForValue(reset_value.value())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Reset value %s for register %s is not of type %s",
          reset_value.value().ToString(), name, type->ToString()));
    }
  }
  registers_[name] =
      absl::make_unique<Register>(std::string(name), type, reset_value, this);
  register_vec_.push_back(registers_[name].get());
  return register_vec_.back();
}

absl::Status Block::RemoveRegister(Register* reg) {
  if (reg->block() != this) {
    return absl::InvalidArgumentError("Register is not owned by block.");
  }
  if (!registers_.contains(reg->name())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block %s has no register named %s", name(), reg->name()));
  }

  XLS_RET_CHECK(registers_.at(reg->name()).get() == reg);

  auto it = std::find(register_vec_.begin(), register_vec_.end(), reg);
  XLS_RET_CHECK(it != register_vec_.end());
  register_vec_.erase(it);
  registers_.erase(reg->name());
  return absl::OkStatus();
}

absl::StatusOr<Register*> Block::GetRegister(absl::string_view name) const {
  if (!registers_.contains(name)) {
    return absl::NotFoundError(absl::StrFormat(
        "Block %s has no register named %s", this->name(), name));
  }
  return registers_.at(name).get();
}

}  // namespace xls
