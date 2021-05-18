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

}  // namespace xls
