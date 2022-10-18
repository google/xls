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

#include "xls/ir/instantiation.h"

#include "absl/strings/str_format.h"
#include "xls/ir/block.h"

namespace xls {

std::string InstantiationKindToString(InstantiationKind kind) {
  switch (kind) {
    case InstantiationKind::kBlock:
      return "block";
    case InstantiationKind::kFifo:
      return "fifo";
    case InstantiationKind::kExtern:
      return "extern";
  }
  XLS_LOG(FATAL) << "Invalid instantiation kind: "
                 << static_cast<int64_t>(kind);
}

absl::StatusOr<InstantiationKind> StringToInstantiationKind(
    std::string_view str) {
  if (str == "block") {
    return InstantiationKind::kBlock;
  } else if (str == "fifo") {
    return InstantiationKind::kFifo;
  } else if (str == "extern") {
    return InstantiationKind::kExtern;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid instantiation kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, InstantiationKind kind) {
  os << InstantiationKindToString(kind);
  return os;
}

std::string BlockInstantiation::ToString() const {
  return absl::StrFormat("instantiation %s(block=%s, kind=block)", name(),
                         instantiated_block()->name());
}

absl::StatusOr<InstantiationPort> BlockInstantiation::GetInputPort(
    std::string_view name) {
  for (InputPort* input_port : instantiated_block()->GetInputPorts()) {
    if (input_port->GetName() == name) {
      return InstantiationPort{std::string{name}, input_port->GetType()};
    }
  }
  return absl::NotFoundError(absl::StrFormat("No such input port `%s`", name));
}

absl::StatusOr<InstantiationPort> BlockInstantiation::GetOutputPort(
    std::string_view name) {
  for (OutputPort* output_port : instantiated_block()->GetOutputPorts()) {
    if (output_port->GetName() == name) {
      return InstantiationPort{std::string{name},
                               output_port->operand(0)->GetType()};
    }
  }
  return absl::NotFoundError(absl::StrFormat("No such output port `%s`", name));
}

}  // namespace xls
