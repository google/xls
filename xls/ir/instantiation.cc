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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
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
  LOG(FATAL) << "Invalid instantiation kind: " << static_cast<int64_t>(kind);
}

absl::StatusOr<InstantiationKind> StringToInstantiationKind(
    std::string_view str) {
  if (str == "block") {
    return InstantiationKind::kBlock;
  }
  if (str == "fifo") {
    return InstantiationKind::kFifo;
  }
  if (str == "extern") {
    return InstantiationKind::kExtern;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid instantiation kind '%s'", str));
}

std::ostream& operator<<(std::ostream& os, InstantiationKind kind) {
  os << InstantiationKindToString(kind);
  return os;
}

namespace {
absl::Status UnexpectedKind(InstantiationKind expected,
                            InstantiationKind actual) {
  CHECK_NE(expected, actual) << "Unexpected call to base As... function.";
  return absl::InternalError(absl::StrFormat(
      "Wrong instantiation kind. Expected %s but is %s",
      InstantiationKindToString(expected), InstantiationKindToString(actual)));
}
}  // namespace
absl::StatusOr<BlockInstantiation*> Instantiation::AsBlockInstantiation() {
  return UnexpectedKind(/*expected=*/InstantiationKind::kBlock,
                        /*actual=*/kind());
}
absl::StatusOr<ExternInstantiation*> Instantiation::AsExternInstantiation() {
  return UnexpectedKind(/*expected=*/InstantiationKind::kExtern,
                        /*actual=*/kind());
}
absl::StatusOr<FifoInstantiation*> Instantiation::AsFifoInstantiation() {
  return UnexpectedKind(/*expected=*/InstantiationKind::kFifo,
                        /*actual=*/kind());
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
      return InstantiationPort{.name = std::string{name},
                               .type = output_port->operand(0)->GetType()};
    }
  }
  return absl::NotFoundError(absl::StrFormat("No such output port `%s`", name));
}

// Note: these are tested in ffi_instantiation_pass_test
static absl::StatusOr<InstantiationPort> ExtractNested(
    std::string_view fn_name, std::string_view full_parameter_name,
    Type* current_type, std::string_view nested_name) {
  static const LazyRE2 kReMatchTupleId{"\\.([0-9]+)(.*)"};

  // (start_name ++ nested_name) == full_name
  const std::string_view start_name = full_parameter_name.substr(
      0, nested_name.data() - full_parameter_name.data());

  switch (current_type->kind()) {
    case TypeKind::kBits:
      if (!nested_name.empty()) {
        return absl::NotFoundError(absl::StrCat(
            "Attempting to access tuple-field `", full_parameter_name,
            "` but `", start_name, "` is already a scalar (of type ",
            current_type->ToString(), ")"));
      }
      return InstantiationPort{std::string(full_parameter_name), current_type};
      break;
    case TypeKind::kTuple: {
      TupleType* const tuple = current_type->AsTupleOrDie();
      int64_t tuple_index = 0;
      std::string_view parameter_remaining;
      if (!RE2::FullMatch(nested_name, *kReMatchTupleId, &tuple_index,
                          &parameter_remaining)) {
        return absl::NotFoundError(
            absl::StrFormat("%s: %s is a tuple (with %d fields), expected "
                            "sub-access by .<number>",
                            fn_name, start_name, tuple->size()));
      }
      if (tuple_index < 0 || tuple_index >= tuple->size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s: Invalid index into tuple `%s.%d`; expected to be in "
            "range 0..%d",
            fn_name, start_name, tuple_index, tuple->size() - 1));
      }
      Type* const element_type = tuple->element_type(tuple_index);
      return ExtractNested(fn_name, full_parameter_name, element_type,
                           parameter_remaining);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Can not represent FFI types yet other than bits and tuples");
  }

  // Issue in template.
  return absl::NotFoundError(absl::StrFormat("%s: No such output port `%s`",
                                             fn_name, full_parameter_name));
}

absl::StatusOr<InstantiationPort> ExternInstantiation::GetInputPort(
    std::string_view name) {
  std::string_view::size_type dot = name.find_first_of('.');
  std::string_view main_name =
      name.substr(0, dot);  // Up to start of tuple access
  XLS_ASSIGN_OR_RETURN(Param * param, function_->GetParamByName(main_name));
  const std::string& fn_name = function()->name();
  const std::string_view nested_name = name.substr(main_name.length());
  return ExtractNested(fn_name, name, param->GetType(), nested_name);
}

// Extern instantiation (FFI) will have names in template refer to output
// ports, such as "return" for a scalar or "return.0" for a tuple value.
absl::StatusOr<InstantiationPort> ExternInstantiation::GetOutputPort(
    std::string_view name) {
  const std::string& fn_name = function()->name();
  const std::string_view prefix = "return";
  if (!absl::StartsWith(name, prefix)) {
    return absl::NotFoundError(absl::StrFormat(
        "%s: output port reference needs to start with `%s`; got `%s`", fn_name,
        prefix, name));
  }
  return ExtractNested(fn_name, name, function()->GetType()->return_type(),
                       name.substr(prefix.length()));
}

std::string ExternInstantiation::ToString() const {
  return absl::StrFormat("instantiation %s(foreign_function=%s, kind=extern)",
                         name(), function_->name());
}

FifoInstantiation::FifoInstantiation(
    std::string_view inst_name, FifoConfig fifo_config, Type* data_type,
    std::optional<std::string_view> channel_name, Package* package)
    : Instantiation(inst_name, InstantiationKind::kFifo),
      fifo_config_(fifo_config),
      data_type_(data_type),
      channel_name_(channel_name),
      package_(package) {
  CHECK(package->IsOwnedType(data_type));
}

absl::StatusOr<InstantiationPort> FifoInstantiation::GetInputPort(
    std::string_view name) {
  if (name == kPushDataPortName) {
    return InstantiationPort{.name = std::string{kPushDataPortName},
                             .type = data_type()};
  }
  if (name == kPushValidPortName) {
    return InstantiationPort{.name = std::string{kPushValidPortName},
                             .type = package_->GetBitsType(1)};
  }
  if (name == kPopReadyPortName) {
    return InstantiationPort{.name = std::string{kPopReadyPortName},
                             .type = package_->GetBitsType(1)};
  }
  return absl::NotFoundError(
      absl::Substitute("No such input port `$0`: must be one of push_data, "
                       "push_valid, or pop_ready.",
                       name));
}

absl::StatusOr<InstantiationPort> FifoInstantiation::GetOutputPort(
    std::string_view name) {
  if (name == kPopDataPortName) {
    return InstantiationPort{.name = std::string{kPopDataPortName},
                             .type = data_type()};
  }
  if (name == kPopValidPortName) {
    return InstantiationPort{.name = std::string{kPopValidPortName},
                             .type = package_->GetBitsType(1)};
  }
  if (name == kPushReadyPortName) {
    return InstantiationPort{.name = std::string{kPushReadyPortName},
                             .type = package_->GetBitsType(1)};
  }
  return absl::NotFoundError(absl::StrFormat("No such output port `%s`", name));
}

std::string FifoInstantiation::ToString() const {
  std::string channel_str;
  if (channel_name_.has_value()) {
    channel_str = absl::StrFormat("channel=%s, ", *channel_name_);
  }

  return absl::StrFormat(
      "instantiation %s(data_type=%s, depth=%d, bypass=%s, "
      "register_push_outputs=%s, register_pop_outputs=%s, %skind=fifo)",
      name(), data_type_->ToString(), fifo_config_.depth(),
      fifo_config_.bypass() ? "true" : "false",
      fifo_config_.register_push_outputs() ? "true" : "false",
      fifo_config_.register_pop_outputs() ? "true" : "false", channel_str);
}

}  // namespace xls
