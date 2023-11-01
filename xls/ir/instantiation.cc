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

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "re2/re2.h"

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

absl::StatusOr<InstantiationPort> ExternInstantiation::GetInputPort(
    std::string_view name) {
  XLS_ASSIGN_OR_RETURN(Param * param, function_->GetParamByName(name));
  return InstantiationPort{std::string{name}, param->GetType()};
}

absl::StatusOr<InstantiationPort> ExternInstantiation::GetOutputPort(
    std::string_view name) {
  static const LazyRE2 kReMatchTupleId{"return\\.([0-9]+)"};
  Type* const return_type = function_->GetType()->return_type();

  switch (return_type->kind()) {
    case TypeKind::kBits:
      if (name == "return") {
        return InstantiationPort{"return", function_->GetType()->return_type()};
      }
      break;
    case TypeKind::kTuple: {
      int64_t tuple_index = 0;
      if (!RE2::FullMatch(name, *kReMatchTupleId, &tuple_index)) {
        return absl::NotFoundError(
            absl::StrFormat("%s: Expected return value parameter to be of form "
                            "return.<tuple-index>; got `%s`",
                            function()->name(), name));
      }
      TupleType* const tuple = return_type->AsTupleOrDie();
      if (tuple_index < 0 || tuple_index >= tuple->size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s: Invalid index into tuple with `return.%d`; expected to be in "
            "range 0..%d",
            function()->name(), tuple_index, tuple->size() - 1));
      }
      Type* const element_type = tuple->element_type(tuple_index);
      if (!element_type->IsBits()) {
        return absl::UnimplementedError("Not supporting nested tuples yet");
      }
      return InstantiationPort{std::string{name}, element_type};
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Can not represent FFI types yet other than bits and tuples");
  }

  // Issue in template.
  return absl::NotFoundError(absl::StrFormat("No such output port `%s`", name));
}

std::string ExternInstantiation::ToString() const {
  return absl::StrFormat("instantiation %s(foreign_function=%s, kind=extern)",
                         name(), function_->name());
}

FifoInstantiation::FifoInstantiation(std::string_view inst_name,
                                     FifoConfig fifo_config, Type* data_type,
                                     std::optional<int64_t> channel_id,
                                     Package* package)
    : Instantiation(inst_name, InstantiationKind::kFifo),
      fifo_config_(fifo_config),
      data_type_(data_type),
      channel_id_(channel_id),
      package_(package) {
  XLS_CHECK(package->IsOwnedType(data_type));
}

absl::StatusOr<InstantiationPort> FifoInstantiation::GetInputPort(
    std::string_view name) {
  if (name == "push_data") {
    return InstantiationPort{std::string{name}, data_type()};
  }
  if (name == "push_valid") {
    return InstantiationPort{std::string{name}, package_->GetBitsType(1)};
  }
  if (name == "pop_ready") {
    return InstantiationPort{std::string{name}, package_->GetBitsType(1)};
  }
  return absl::NotFoundError(
      absl::Substitute("No such input port `$0`: must be one of push_data, "
                       "push_valid, or pop_ready.",
                       name));
}

absl::StatusOr<InstantiationPort> FifoInstantiation::GetOutputPort(
    std::string_view name) {
  if (name == "pop_data") {
    return InstantiationPort{std::string{name}, data_type()};
  }
  if (name == "pop_valid") {
    return InstantiationPort{std::string{name}, package_->GetBitsType(1)};
  }
  if (name == "push_ready") {
    return InstantiationPort{std::string{name}, package_->GetBitsType(1)};
  }
  return absl::NotFoundError(absl::StrFormat("No such output port `%s`", name));
}

std::string FifoInstantiation::ToString() const {
  std::string channel_str;
  if (channel_id_.has_value()) {
    channel_str = absl::StrFormat("channel_id=%d, ", *channel_id_);
  }

  return absl::StrFormat(
      "instantiation %s(data_type=%s, depth=%d, bypass=%s, %skind=fifo)",
      name(), data_type_->ToString(), fifo_config_.depth,
      fifo_config_.bypass ? "true" : "false", channel_str);
}

}  // namespace xls
