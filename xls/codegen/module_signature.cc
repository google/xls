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

#include "xls/codegen/module_signature.h"

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"

namespace xls {
namespace verilog {

ModuleSignatureBuilder& ModuleSignatureBuilder::WithClock(
    std::string_view name) {
  XLS_CHECK(!proto_.has_clock_name());
  proto_.set_clock_name(ToProtoString(name));
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithReset(
    std::string_view name, bool asynchronous, bool active_low) {
  XLS_CHECK(!proto_.has_reset());
  ResetProto* reset = proto_.mutable_reset();
  reset->set_name(ToProtoString(name));
  reset->set_asynchronous(asynchronous);
  reset->set_active_low(active_low);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithFixedLatencyInterface(
    int64_t latency) {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  FixedLatencyInterface* interface = proto_.mutable_fixed_latency();
  interface->set_latency(latency);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithCombinationalInterface() {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_combinational();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithUnknownInterface() {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_unknown();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithPipelineInterface(
    int64_t latency, int64_t initiation_interval,
    std::optional<PipelineControl> pipeline_control) {
  XLS_CHECK_EQ(proto_.interface_oneof_case(),
               ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  PipelineInterface* interface = proto_.mutable_pipeline();
  interface->set_latency(latency);
  interface->set_initiation_interval(initiation_interval);
  if (pipeline_control.has_value()) {
    *interface->mutable_pipeline_control() = *pipeline_control;
  }
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataInput(
    std::string_view name, Type* type) {
  PortProto* port = proto_.add_data_ports();
  port->set_direction(DIRECTION_INPUT);
  port->set_name(ToProtoString(name));
  port->set_width(type->GetFlatBitCount());
  *port->mutable_type() = type->ToProto();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataOutput(
    std::string_view name, Type* type) {
  PortProto* port = proto_.add_data_ports();
  port->set_direction(DIRECTION_OUTPUT);
  port->set_name(ToProtoString(name));
  port->set_width(type->GetFlatBitCount());
  *port->mutable_type() = type->ToProto();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataInputAsBits(
    std::string_view name, int64_t width) {
  BitsType bits_type(width);
  return AddDataInput(name, &bits_type);
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataOutputAsBits(
    std::string_view name, int64_t width) {
  BitsType bits_type(width);
  return AddDataOutput(name, &bits_type);
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddSingleValueChannel(
    std::string_view name, ChannelOps supported_ops,
    std::string_view port_name) {
  ChannelProto* channel = proto_.add_data_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_SINGLE_VALUE);

  if (supported_ops == ChannelOps::kSendOnly) {
    channel->set_supported_ops(CHANNEL_OPS_SEND_ONLY);
  } else if (supported_ops == ChannelOps::kReceiveOnly) {
    channel->set_supported_ops(CHANNEL_OPS_RECEIVE_ONLY);
  } else {
    channel->set_supported_ops(CHANNEL_OPS_SEND_RECEIVE);
  }

  channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  channel->set_data_port_name(ToProtoString(port_name));

  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddStreamingChannel(
    std::string_view name, ChannelOps supported_ops, FlowControl flow_control,
    std::optional<int64_t> fifo_depth, std::string_view port_name,
    std::optional<std::string_view> valid_port_name,
    std::optional<std::string_view> ready_port_name) {
  ChannelProto* channel = proto_.add_data_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_STREAMING);

  if (supported_ops == ChannelOps::kSendOnly) {
    channel->set_supported_ops(CHANNEL_OPS_SEND_ONLY);
  } else if (supported_ops == ChannelOps::kReceiveOnly) {
    channel->set_supported_ops(CHANNEL_OPS_RECEIVE_ONLY);
  } else {
    channel->set_supported_ops(CHANNEL_OPS_SEND_RECEIVE);
  }

  if (flow_control == FlowControl::kReadyValid) {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_READY_VALID);
  } else {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  }

  if (fifo_depth.has_value()) {
    channel->set_fifo_depth(fifo_depth.value());
  }

  channel->set_data_port_name(ToProtoString(port_name));

  if (ready_port_name.has_value()) {
    channel->set_ready_port_name(ToProtoString(ready_port_name.value()));
  }

  if (valid_port_name.has_value()) {
    channel->set_valid_port_name(ToProtoString(valid_port_name.value()));
  }

  return *this;
}

absl::Status ModuleSignatureBuilder::RemoveStreamingChannel(
    std::string_view name) {
  auto channel_itr =
      std::find_if(proto_.data_channels().begin(), proto_.data_channels().end(),
                   [name](const ChannelProto& channel) -> bool {
                     return channel.name() == ToProtoString(name);
                   });
  if (channel_itr == proto_.mutable_data_channels()->end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel with name %s could not be found in the ModuleSignature.",
        name));
  }
  proto_.mutable_data_channels()->erase(channel_itr);
  return absl::OkStatus();
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddRamRWPort(
    std::string_view ram_name, std::string_view req_name,
    std::string_view resp_name, int64_t address_width, int64_t data_width,
    std::string_view address_name, std::string_view read_enable_name,
    std::string_view write_enable_name, std::string_view read_data_name,
    std::string_view write_data_name) {
  RamProto* ram = proto_.add_rams();
  ram->set_name(ToProtoString(ram_name));

  Ram1RWProto* ram_1rw = ram->mutable_ram_1rw();
  RamRWPortProto* rw_port = ram_1rw->mutable_rw_port();

  RamRWRequestProto* req = rw_port->mutable_request();
  RamRWResponseProto* resp = rw_port->mutable_response();

  req->set_name(ToProtoString(req_name));
  resp->set_name(ToProtoString(resp_name));

  auto* address_proto = req->mutable_address();
  address_proto->set_name(ToProtoString(address_name));
  address_proto->set_direction(DIRECTION_OUTPUT);
  address_proto->set_width(address_width);

  auto* read_enable_proto = req->mutable_read_enable();
  read_enable_proto->set_name(ToProtoString(read_enable_name));
  read_enable_proto->set_direction(DIRECTION_OUTPUT);
  read_enable_proto->set_width(1);

  auto* write_enable_proto = req->mutable_write_enable();
  write_enable_proto->set_name(ToProtoString(write_enable_name));
  write_enable_proto->set_direction(DIRECTION_OUTPUT);
  write_enable_proto->set_width(1);

  auto* write_data_proto = req->mutable_write_data();
  write_data_proto->set_name(ToProtoString(write_data_name));
  write_data_proto->set_direction(DIRECTION_OUTPUT);
  write_data_proto->set_width(data_width);

  auto* read_data_proto = resp->mutable_read_data();
  read_data_proto->set_name(ToProtoString(read_data_name));
  read_data_proto->set_direction(DIRECTION_INPUT);
  read_data_proto->set_width(data_width);

  return *this;
}

static absl::Status ValidateProto(const ModuleSignatureProto& proto) {
  // TODO(meheff): do more validation here.
  // Validate widths/number of function type.
  if (proto.has_pipeline() && !proto.has_clock_name()) {
    return absl::InvalidArgumentError("Missing clock signal");
  }
  for (const PortProto& port : proto.data_ports()) {
    if (!port.has_name() || port.name().empty()) {
      return absl::InvalidArgumentError("A name is required for all ports.");
    }
    if (port.direction() == DirectionProto::DIRECTION_INVALID) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' has an invalid port direction.", port.name()));
    }
  }
  auto data_ports = proto.data_ports();
  absl::flat_hash_map<std::string, int64_t> name_data_ports_map;
  for (int64_t index = 0; index < data_ports.size(); ++index) {
    std::string_view port_name = data_ports[index].name();
    if (name_data_ports_map.contains(port_name)) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Port name '%s' found more than once in signature proto.",
          port_name));
    }
    name_data_ports_map[data_ports[index].name()] = index;
  }

  absl::flat_hash_map<std::string, std::string> port_name_to_channel_reference;
  for (const ChannelProto& channel : proto.data_channels()) {
    if (!channel.has_name() || channel.name().empty()) {
      return absl::InvalidArgumentError("A name is required for all channels.");
    }

    if (channel.supported_ops() == ChannelOpsProto::CHANNEL_OPS_INVALID) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel '%s' has an invalid entry for support ops.",
                          channel.name()));
    }

    if (channel.kind() == ChannelKindProto::CHANNEL_KIND_INVALID) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel '%s' has an invalid kind.", channel.name()));
    }

    if (!channel.has_data_port_name() || channel.data_port_name().empty()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "The data port of channel '%s' is not assigned.", channel.name()));
    }

    // Ensure the channel names exist in the port list.
    if (channel.has_data_port_name() &&
        !name_data_ports_map.contains(channel.data_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' of channel '%s' is not present in the port list.",
          channel.data_port_name(), channel.name()));
    }
    if (channel.has_valid_port_name() &&
        !name_data_ports_map.contains(channel.valid_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' of channel '%s' is not present in the port list.",
          channel.valid_port_name(), channel.name()));
    }
    if (channel.has_ready_port_name() &&
        !name_data_ports_map.contains(channel.ready_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' of channel '%s' is not present in the port list.",
          channel.ready_port_name(), channel.name()));
    }

    // Ensure that ports are referenced by at most one channel.
    if (port_name_to_channel_reference.contains(channel.data_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' referenced by more than one channel: found in channel "
          "'%s' and channel '%s'.",
          channel.data_port_name(),
          port_name_to_channel_reference[channel.data_port_name()],
          channel.name()));
    }
    if (channel.has_valid_port_name() &&
        port_name_to_channel_reference.contains(channel.valid_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' referenced by more than one channel: found in channel "
          "'%s' and channel '%s'.",
          channel.valid_port_name(),
          port_name_to_channel_reference[channel.valid_port_name()],
          channel.name()));
    }
    if (channel.has_ready_port_name() &&
        port_name_to_channel_reference.contains(channel.ready_port_name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' referenced by more than one channel: found in channel "
          "'%s' and channel '%s'.",
          channel.ready_port_name(),
          port_name_to_channel_reference[channel.ready_port_name()],
          channel.name()));
    }

    // TODO (vmirian): 10-30-2022 Investigate that channel port directions can
    // be derived from channel ops.
    int64_t data_port_index = name_data_ports_map[channel.data_port_name()];
    if (channel.has_valid_port_name()) {
      int64_t valid_port_index = name_data_ports_map[channel.valid_port_name()];
      PortProto data_port = proto.data_ports(data_port_index);
      PortProto valid_port = proto.data_ports(valid_port_index);
      if (data_port.direction() != valid_port.direction()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "For channel '%s', data port '%s' and valid port '%s' must have "
            "the same direction.",
            data_port.name(), valid_port.name(), channel.name()));
      }
      if (valid_port.width() != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "For channel '%s', valid port '%s' must have a width of 1 bit.",
            valid_port.name(), channel.name()));
      }
    }
    if (channel.has_ready_port_name()) {
      int64_t ready_port_index = name_data_ports_map[channel.valid_port_name()];
      PortProto data_port = proto.data_ports(data_port_index);
      PortProto ready_port = proto.data_ports(ready_port_index);
      if (data_port.direction() != ready_port.direction()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "For channel '%s', data port '%s' and ready port '%s' must have "
            "the same direction.",
            data_port.name(), ready_port.name(), channel.name()));
      }
      if (ready_port.width() != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "For channel '%s', ready port '%s' must have a width of 1 bit.",
            ready_port.name(), channel.name()));
      }
    }

    port_name_to_channel_reference[channel.data_port_name()] = channel.name();
    if (channel.has_valid_port_name()) {
      port_name_to_channel_reference[channel.valid_port_name()] =
          channel.name();
    }
    if (channel.has_ready_port_name()) {
      port_name_to_channel_reference[channel.ready_port_name()] =
          channel.name();
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ModuleSignature> ModuleSignatureBuilder::Build() {
  XLS_RETURN_IF_ERROR(ValidateProto(proto_));
  return ModuleSignature::FromProto(proto_);
}

/*static*/ absl::StatusOr<ModuleSignature> ModuleSignature::FromProto(
    const ModuleSignatureProto& proto) {
  XLS_RETURN_IF_ERROR(ValidateProto(proto));

  ModuleSignature signature;
  signature.proto_ = proto;
  for (const PortProto& port : proto.data_ports()) {
    if (port.direction() == DIRECTION_INPUT) {
      signature.data_inputs_.push_back(port);
    } else if (port.direction() == DIRECTION_OUTPUT) {
      signature.data_outputs_.push_back(port);
    } else {
      return absl::InvalidArgumentError("Invalid port direction.");
    }
  }

  // Makes a name-port map from the port list.
  auto port_map_builder = [&](const std::vector<PortProto>& ports)
      -> absl::StatusOr<absl::flat_hash_map<std::string, int64_t>> {
    absl::flat_hash_map<std::string, int64_t> name_port_map;
    for (int64_t i = 0; i < ports.size(); ++i) {
      name_port_map[ports[i].name()] = i;
    }
    return name_port_map;
  };

  XLS_ASSIGN_OR_RETURN(signature.input_port_map_,
                       port_map_builder(signature.data_inputs_));
  XLS_ASSIGN_OR_RETURN(signature.output_port_map_,
                       port_map_builder(signature.data_outputs_));

  for (const ChannelProto& channel : proto.data_channels()) {
    if (channel.kind() == CHANNEL_KIND_SINGLE_VALUE) {
      signature.single_value_channels_.push_back(channel);
    } else if (channel.kind() == CHANNEL_KIND_STREAMING) {
      signature.streaming_channels_.push_back(channel);
    } else {
      return absl::InvalidArgumentError("Invalid channel kind.");
    }
  }

  for (const ChannelProto& channel : proto.data_channels()) {
    if (signature.input_port_map_.contains(channel.data_port_name())) {
      signature.input_channels_.push_back(channel);
    } else {
      signature.output_channels_.push_back(channel);
    }
  }
  // Creates a name-channel map from the channel list.
  //
  // TODO (vmirian) : 10-28-2022 There may be an efficient way to implement
  // the following by using channel ops.
  auto channel_map_builder = [&](const std::vector<ChannelProto>& channels)
      -> absl::StatusOr<absl::flat_hash_map<std::string, int64_t>> {
    absl::flat_hash_map<std::string, int64_t> name_channel_map;
    for (std::int64_t i = 0; i < channels.size(); ++i) {
      std::string_view channel_name = channels[i].name();
      if (name_channel_map.contains(channel_name)) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "Channel name '%s' found more than once in signature proto.",
            channel_name));
      }
      name_channel_map[channel_name] = i;
    }
    return name_channel_map;
  };

  XLS_ASSIGN_OR_RETURN(signature.input_channel_map_,
                       channel_map_builder(signature.input_channels_));
  XLS_ASSIGN_OR_RETURN(signature.output_channel_map_,
                       channel_map_builder(signature.output_channels_));

  for (const ChannelProto& channel : proto.data_channels()) {
    signature.port_name_to_channel_name[channel.data_port_name()] =
        channel.name();
    if (channel.has_valid_port_name()) {
      signature.port_name_to_channel_name[channel.valid_port_name()] =
          channel.name();
    }
    if (channel.has_ready_port_name()) {
      signature.port_name_to_channel_name[channel.ready_port_name()] =
          channel.name();
    }
  }

  for (const RamProto& ram_ports : proto.rams()) {
    signature.rams_.push_back(ram_ports);
  }

  return signature;
}

int64_t ModuleSignature::TotalDataInputBits() const {
  int64_t total = 0;
  for (const PortProto& port : data_inputs()) {
    total += port.width();
  }
  return total;
}

int64_t ModuleSignature::TotalDataOutputBits() const {
  int64_t total = 0;
  for (const PortProto& port : data_outputs()) {
    total += port.width();
  }
  return total;
}

// Checks that the given inputs match one-to-one to the input ports (matched
// by name). Returns a vector containing the inputs in the same order as the
// input ports.
template <typename T>
static absl::StatusOr<std::vector<const T*>> CheckAndReturnOrderedInputs(
    absl::Span<const PortProto> input_ports,
    const absl::flat_hash_map<std::string, T>& inputs) {
  absl::flat_hash_set<std::string> port_names;
  std::vector<const T*> ordered_inputs;
  for (const PortProto& port : input_ports) {
    port_names.insert(port.name());

    if (!inputs.contains(port.name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input '%s' was not passed as an argument.", port.name()));
    }
    ordered_inputs.push_back(&inputs.at(port.name()));
  }

  // Verify every passed in input is accounted for.
  for (const auto& pair : inputs) {
    if (!port_names.contains(pair.first)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unexpected input value named '%s'.", pair.first));
    }
  }
  return ordered_inputs;
}

absl::Status ModuleSignature::ValidateInputs(
    const absl::flat_hash_map<std::string, Bits>& input_bits) const {
  XLS_ASSIGN_OR_RETURN(std::vector<const Bits*> ordered_inputs,
                       CheckAndReturnOrderedInputs(data_inputs(), input_bits));
  for (int64_t i = 0; i < ordered_inputs.size(); ++i) {
    const PortProto& port = data_inputs()[i];
    const Bits* input = ordered_inputs[i];
    if (port.width() != input->bit_count()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected input '%s' to have width %d, has width %d",
                          port.name(), port.width(), input->bit_count()));
    }
  }
  return absl::OkStatus();
}

static std::string TypeProtoToString(const TypeProto& proto) {
  // Create a dummy package for creating Type*'s from a proto.
  // TODO(meheff): Find a better way to manage types. We need types disconnected
  // from any IR package.
  Package p("dummy_package");
  auto type_status = p.GetTypeFromProto(proto);
  if (!type_status.ok()) {
    return "<invalid>";
  }
  return type_status.value()->ToString();
}

static absl::StatusOr<bool> TypeProtosEqual(const TypeProto& a,
                                            const TypeProto& b) {
  // Create a dummy package for creating Type*'s from a proto.
  // TODO(meheff): Find a better way to manage types. We need types disconnected
  // from any IR package.
  Package p("dummy_package");
  XLS_ASSIGN_OR_RETURN(Type * a_type, p.GetTypeFromProto(a));
  XLS_ASSIGN_OR_RETURN(Type * b_type, p.GetTypeFromProto(b));
  return a_type == b_type;
}

absl::Status ModuleSignature::ValidateInputs(
    const absl::flat_hash_map<std::string, Value>& input_values) const {
  XLS_ASSIGN_OR_RETURN(
      std::vector<const Value*> ordered_inputs,
      CheckAndReturnOrderedInputs(data_inputs(), input_values));
  for (int64_t i = 0; i < ordered_inputs.size(); ++i) {
    const Value* input = ordered_inputs[i];
    const TypeProto& expected_type_proto = data_inputs()[i].type();
    XLS_ASSIGN_OR_RETURN(TypeProto value_type_proto, input->TypeAsProto());
    XLS_ASSIGN_OR_RETURN(bool types_equal, TypeProtosEqual(expected_type_proto,
                                                           value_type_proto));
    if (!types_equal) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input value '%s' is wrong type. Expected '%s', got '%s'",
          data_inputs()[i].name(), TypeProtoToString(expected_type_proto),
          TypeProtoToString(value_type_proto)));
    }
  }
  return absl::OkStatus();
}

absl::Status ModuleSignature::ValidateChannelBitsInputs(
    std::string_view channel_name, absl::Span<const Bits> values) const {
  if (!input_channel_map_.contains(channel_name)) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Channel '%s' is not an input channel.", channel_name));
  }
  const ChannelProto& channel =
      input_channels_[input_channel_map_.at(channel_name)];
  PortProto port = data_inputs_[input_port_map_.at(channel.data_port_name())];
  for (const Bits& value : values) {
    if (port.width() != value.bit_count()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected input '%s' to have width %d, has width %d",
                          port.name(), port.width(), value.bit_count()));
    }
  }
  return absl::OkStatus();
}

absl::Status ModuleSignature::ValidateChannelValueInputs(
    std::string_view channel_name, absl::Span<const Value> values) const {
  if (!input_channel_map_.contains(channel_name)) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Channel '%s' is not an input channel.", channel_name));
  }
  const ChannelProto& channel =
      input_channels_[input_channel_map_.at(channel_name)];
  TypeProto expected_type_proto =
      data_inputs_[input_port_map_.at(channel.data_port_name())].type();
  for (const Value& value : values) {
    XLS_ASSIGN_OR_RETURN(TypeProto value_type_proto, value.TypeAsProto());
    XLS_ASSIGN_OR_RETURN(bool types_equal, TypeProtosEqual(expected_type_proto,
                                                           value_type_proto));
    if (!types_equal) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input value '%s' is wrong type. Expected '%s', got '%s'",
          channel.data_port_name(), TypeProtoToString(expected_type_proto),
          TypeProtoToString(value_type_proto)));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
ModuleSignature::ToKwargs(absl::Span<const Value> inputs) const {
  if (inputs.size() != data_inputs().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d arguments, got %d.", data_inputs().size(), inputs.size()));
  }
  absl::flat_hash_map<std::string, Value> kwargs;
  for (int64_t i = 0; i < data_inputs().size(); ++i) {
    kwargs[data_inputs()[i].name()] = inputs[i];
  }
  return kwargs;
}

absl::StatusOr<PortProto> ModuleSignature::GetInputPortProtoByName(
    std::string_view name) const {
  if (!input_port_map_.contains(name)) {
    return absl::NotFoundError(
        absl::StrFormat("Port '%s' is not an input port.", name));
  }
  return data_inputs_[input_port_map_.at(name)];
}

absl::StatusOr<PortProto> ModuleSignature::GetOutputPortProtoByName(
    std::string_view name) const {
  if (!output_port_map_.contains(name)) {
    return absl::NotFoundError(
        absl::StrFormat("Port '%s' is not an output port.", name));
  }
  return data_outputs_[output_port_map_.at(name)];
}

absl::StatusOr<ChannelProto> ModuleSignature::GetInputChannelProtoByName(
    std::string_view name) const {
  if (!input_channel_map_.contains(name)) {
    return absl::NotFoundError(
        absl::StrFormat("Channel '%s' is not an input channel.", name));
  }
  return input_channels_[input_channel_map_.at(name)];
}

absl::StatusOr<ChannelProto> ModuleSignature::GetOutputChannelProtoByName(
    std::string_view name) const {
  if (!output_channel_map_.contains(name)) {
    return absl::NotFoundError(
        absl::StrFormat("Channel '%s' is not an output channel.", name));
  }
  return output_channels_[output_channel_map_.at(name)];
}

absl::StatusOr<std::string> ModuleSignature::GetChannelNameWith(
    std::string_view port_name) const {
  if (!port_name_to_channel_name.contains(port_name)) {
    return absl::NotFoundError(absl::StrFormat(
        "Port name '%s' is not referenced by a channel.", port_name));
  }
  return port_name_to_channel_name.at(port_name);
}

absl::Status ModuleSignature::ReplaceBlockMetrics(
    BlockMetricsProto block_metrics) {
  *proto_.mutable_metrics()->mutable_block_metrics() = std::move(block_metrics);
  return absl::OkStatus();
}

std::ostream& operator<<(std::ostream& os, const ModuleSignature& signature) {
  os << signature.ToString();
  return os;
}

}  // namespace verilog
}  // namespace xls
