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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace verilog {

ModuleSignatureBuilder& ModuleSignatureBuilder::WithClock(
    std::string_view name) {
  CHECK(!proto_.has_clock_name());
  proto_.set_clock_name(ToProtoString(name));
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithReset(std::string_view name,
                                                          bool asynchronous,
                                                          bool active_low) {
  CHECK(!proto_.has_reset());
  ResetProto* reset = proto_.mutable_reset();
  reset->set_name(ToProtoString(name));
  reset->set_asynchronous(asynchronous);
  reset->set_active_low(active_low);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithFixedLatencyInterface(
    int64_t latency) {
  CHECK_EQ(proto_.interface_oneof_case(),
           ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  FixedLatencyInterface* interface = proto_.mutable_fixed_latency();
  interface->set_latency(latency);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithCombinationalInterface() {
  CHECK_EQ(proto_.interface_oneof_case(),
           ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_combinational();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithUnknownInterface() {
  CHECK_EQ(proto_.interface_oneof_case(),
           ModuleSignatureProto::INTERFACE_ONEOF_NOT_SET);
  proto_.mutable_unknown();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::WithPipelineInterface(
    int64_t latency, int64_t initiation_interval,
    std::optional<PipelineControl> pipeline_control) {
  CHECK_EQ(proto_.interface_oneof_case(),
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
  port->set_direction(PORT_DIRECTION_INPUT);
  port->set_name(ToProtoString(name));
  port->set_width(type->GetFlatBitCount());
  *port->mutable_type() = type->ToProto();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddDataOutput(
    std::string_view name, Type* type) {
  PortProto* port = proto_.add_data_ports();
  port->set_direction(PORT_DIRECTION_OUTPUT);
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

absl::Status ModuleSignatureBuilder::RemoveData(std::string_view name) {
  auto port_itr =
      std::find_if(proto_.data_ports().begin(), proto_.data_ports().end(),
                   [name](const PortProto& port) -> bool {
                     return port.name() == ToProtoString(name);
                   });
  if (port_itr == proto_.data_ports().end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Port with name %s could not be found in the ModuleSignature.", name));
  }
  proto_.mutable_data_ports()->erase(port_itr);
  return absl::OkStatus();
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddSingleValueChannel(
    std::string_view name, Type* type) {
  ChannelProto* channel = proto_.add_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_SINGLE_VALUE);
  *channel->mutable_type() = type->ToProto();
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddStreamingChannel(
    std::string_view name, Type* type, FlowControl flow_control,
    std::optional<FifoConfig> fifo_config) {
  ChannelProto* channel = proto_.add_channels();
  channel->set_name(ToProtoString(name));
  channel->set_kind(CHANNEL_KIND_STREAMING);
  if (flow_control == FlowControl::kReadyValid) {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_READY_VALID);
  } else {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  }
  *channel->mutable_type() = type->ToProto();
  if (fifo_config.has_value()) {
    *channel->mutable_fifo_config() =
        fifo_config->ToProto(type->GetFlatBitCount());
  }
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddSingleValueChannelInterface(
    std::string_view name, ChannelDirectionProto direction, Type* type,
    std::string_view data_port_name, FlopKindProto flop_kind) {
  ChannelInterfaceProto* interface = proto_.add_channel_interfaces();
  interface->set_channel_name(ToProtoString(name));
  interface->set_direction(direction);
  *interface->mutable_type() = type->ToProto();
  interface->set_kind(CHANNEL_KIND_SINGLE_VALUE);
  interface->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  interface->set_data_port_name(ToProtoString(data_port_name));
  interface->set_flop_kind(flop_kind);
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddStreamingChannelInterface(
    std::string_view name, ChannelDirectionProto direction, Type* type,
    FlowControl flow_control, std::optional<std::string> data_port_name,
    std::optional<std::string> ready_port_name,
    std::optional<std::string> valid_port_name, FlopKindProto flop_kind) {
  ChannelInterfaceProto* interface = proto_.add_channel_interfaces();
  interface->set_channel_name(ToProtoString(name));
  interface->set_direction(direction);
  *interface->mutable_type() = type->ToProto();
  interface->set_kind(CHANNEL_KIND_STREAMING);
  if (flow_control == FlowControl::kReadyValid) {
    interface->set_flow_control(CHANNEL_FLOW_CONTROL_READY_VALID);
  } else {
    interface->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  }
  interface->set_flop_kind(flop_kind);
  if (data_port_name.has_value()) {
    interface->set_data_port_name(ToProtoString(*data_port_name));
  }
  if (ready_port_name.has_value()) {
    interface->set_ready_port_name(ToProtoString(*ready_port_name));
  }
  if (valid_port_name.has_value()) {
    interface->set_valid_port_name(ToProtoString(*valid_port_name));
  }
  return *this;
}

absl::Status ModuleSignatureBuilder::RemoveChannel(std::string_view name) {
  auto channel_itr =
      std::find_if(proto_.channels().begin(), proto_.channels().end(),
                   [name](const ChannelProto& channel) -> bool {
                     return channel.name() == ToProtoString(name);
                   });
  if (channel_itr == proto_.mutable_channels()->end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Internal channel with name %s could not be found in the "
        "ModuleSignature.",
        name));
  }
  proto_.mutable_channels()->erase(channel_itr);
  return absl::OkStatus();
}

absl::Status ModuleSignatureBuilder::RemoveChannelInterface(
    std::string_view name) {
  auto channel_itr = std::find_if(
      proto_.channel_interfaces().begin(), proto_.channel_interfaces().end(),
      [name](const ChannelInterfaceProto& interface) -> bool {
        return interface.channel_name() == ToProtoString(name);
      });
  if (channel_itr == proto_.mutable_channel_interfaces()->end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Interface channel with name %s could not be found in "
                        "the ModuleSignature.",
                        name));
  }
  proto_.mutable_channel_interfaces()->erase(channel_itr);
  return absl::OkStatus();
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddFifoInstantiation(
    Package* package, std::string_view instance_name,
    std::optional<std::string_view> channel_name, const Type* data_type,
    FifoConfig fifo_config) {
  InstantiationProto* instantiation = proto_.add_instantiations();
  FifoInstantiationProto* fifo_instantiation =
      instantiation->mutable_fifo_instantiation();
  fifo_instantiation->set_instance_name(ToProtoString(instance_name));
  if (channel_name.has_value()) {
    fifo_instantiation->set_channel_name(ToProtoString(channel_name.value()));
  }
  *fifo_instantiation->mutable_fifo_config() =
      fifo_config.ToProto(data_type->GetFlatBitCount());
  *fifo_instantiation->mutable_type() = data_type->ToProto();

  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddBlockInstantiation(
    Package* package, std::string_view block_name,
    std::string_view instance_name) {
  InstantiationProto* instantiation = proto_.add_instantiations();
  BlockInstantiationProto* block_instantiation =
      instantiation->mutable_block_instantiation();
  block_instantiation->set_block_name(ToProtoString(block_name));
  block_instantiation->set_instance_name(ToProtoString(instance_name));
  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddRam1RW(
    const Ram1RWArgs& args) {
  RamProto* ram = proto_.add_rams();
  int64_t data_width = args.data_type->GetFlatBitCount();

  ram->set_name(ToProtoString(args.ram_name));

  Ram1RWProto* ram_1rw = ram->mutable_ram_1rw();
  RamRWPortProto* rw_port = ram_1rw->mutable_rw_port();

  RamRWRequestProto* req = rw_port->mutable_request();
  RamRWResponseProto* resp = rw_port->mutable_response();

  req->set_name(ToProtoString(args.req_name));
  resp->set_name(ToProtoString(args.resp_name));

  auto* address_proto = req->mutable_address();
  address_proto->set_name(ToProtoString(args.address_name));
  address_proto->set_direction(PORT_DIRECTION_OUTPUT);
  address_proto->set_width(args.address_width);
  *address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* read_enable_proto = req->mutable_read_enable();
  read_enable_proto->set_name(ToProtoString(args.read_enable_name));
  read_enable_proto->set_direction(PORT_DIRECTION_OUTPUT);
  read_enable_proto->set_width(1);
  *read_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* write_enable_proto = req->mutable_write_enable();
  write_enable_proto->set_name(ToProtoString(args.write_enable_name));
  write_enable_proto->set_direction(PORT_DIRECTION_OUTPUT);
  write_enable_proto->set_width(1);
  *write_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* write_data_proto = req->mutable_write_data();
  write_data_proto->set_name(ToProtoString(args.write_data_name));
  write_data_proto->set_direction(PORT_DIRECTION_OUTPUT);
  write_data_proto->set_width(data_width);
  *write_data_proto->mutable_type() = args.data_type->ToProto();

  auto* read_data_proto = resp->mutable_read_data();
  read_data_proto->set_name(ToProtoString(args.read_data_name));
  read_data_proto->set_direction(PORT_DIRECTION_INPUT);
  read_data_proto->set_width(data_width);
  *read_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.write_mask_width > 0) {
    auto* write_mask_proto = req->mutable_write_mask();
    write_mask_proto->set_name(ToProtoString(args.write_mask_name));
    write_mask_proto->set_direction(PORT_DIRECTION_OUTPUT);
    write_mask_proto->set_width(args.write_mask_width);
    *write_mask_proto->mutable_type() =
        args.package->GetBitsType(args.write_mask_width)->ToProto();
  }

  if (args.read_mask_width > 0) {
    auto* read_mask_proto = req->mutable_read_mask();
    read_mask_proto->set_name(ToProtoString(args.read_mask_name));
    read_mask_proto->set_direction(PORT_DIRECTION_OUTPUT);
    read_mask_proto->set_width(args.read_mask_width);
    *read_mask_proto->mutable_type() =
        args.package->GetBitsType(args.read_mask_width)->ToProto();
  }

  return *this;
}

ModuleSignatureBuilder& ModuleSignatureBuilder::AddRam1R1W(
    const Ram1R1WArgs& args) {
  int64_t data_width = args.data_type->GetFlatBitCount();
  RamProto* ram = proto_.add_rams();
  ram->set_name(ToProtoString(args.ram_name));

  Ram1R1WProto* ram_1r1w = ram->mutable_ram_1r1w();

  RamRPortProto* r_port = ram_1r1w->mutable_r_port();
  RamWPortProto* w_port = ram_1r1w->mutable_w_port();

  RamRRequestProto* r_req = r_port->mutable_request();
  RamRResponseProto* r_resp = r_port->mutable_response();
  RamWRequestProto* w_req = w_port->mutable_request();

  r_req->set_name(ToProtoString(args.rd_req_name));
  r_resp->set_name(ToProtoString(args.rd_resp_name));
  w_req->set_name(ToProtoString(args.wr_req_name));

  auto* rd_address_proto = r_req->mutable_address();
  rd_address_proto->set_name(ToProtoString(args.read_address_name));
  rd_address_proto->set_direction(PORT_DIRECTION_OUTPUT);
  rd_address_proto->set_width(args.address_width);
  *rd_address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* rd_enable_proto = r_req->mutable_enable();
  rd_enable_proto->set_name(ToProtoString(args.read_enable_name));
  rd_enable_proto->set_direction(PORT_DIRECTION_OUTPUT);
  rd_enable_proto->set_width(1);
  *rd_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* rd_data_proto = r_resp->mutable_data();
  rd_data_proto->set_name(ToProtoString(args.read_data_name));
  rd_data_proto->set_direction(PORT_DIRECTION_INPUT);
  rd_data_proto->set_width(data_width);
  *rd_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.read_mask_width > 0) {
    auto* read_mask_proto = w_req->mutable_mask();
    read_mask_proto->set_name(ToProtoString(args.read_mask_name));
    read_mask_proto->set_direction(PORT_DIRECTION_OUTPUT);
    read_mask_proto->set_width(args.read_mask_width);
    *read_mask_proto->mutable_type() =
        args.package->GetBitsType(args.read_mask_width)->ToProto();
  }

  auto* wr_address_proto = w_req->mutable_address();
  wr_address_proto->set_name(ToProtoString(args.write_address_name));
  wr_address_proto->set_direction(PORT_DIRECTION_OUTPUT);
  wr_address_proto->set_width(args.address_width);
  *wr_address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* wr_data_proto = w_req->mutable_data();
  wr_data_proto->set_name(ToProtoString(args.write_data_name));
  wr_data_proto->set_direction(PORT_DIRECTION_OUTPUT);
  wr_data_proto->set_width(data_width);
  *wr_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.write_mask_width > 0) {
    auto* write_mask_proto = w_req->mutable_mask();
    write_mask_proto->set_name(ToProtoString(args.write_mask_name));
    write_mask_proto->set_direction(PORT_DIRECTION_OUTPUT);
    write_mask_proto->set_width(args.write_mask_width);
    *write_mask_proto->mutable_type() =
        args.package->GetBitsType(args.write_mask_width)->ToProto();
  }

  auto* wr_enable_proto = w_req->mutable_enable();
  wr_enable_proto->set_name(ToProtoString(args.write_enable_name));
  wr_enable_proto->set_direction(PORT_DIRECTION_OUTPUT);
  wr_enable_proto->set_width(1);
  *wr_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

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
    if (port.direction() == PORT_DIRECTION_INVALID) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Port '%s' has an invalid port direction.", port.name()));
    }
  }
  absl::flat_hash_map<std::string, PortProto> name_data_ports_map;
  for (const PortProto& data_port : proto.data_ports()) {
    auto [_, inserted] =
        name_data_ports_map.insert({data_port.name(), data_port});
    if (!inserted) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Port name '%s' found more than once in signature proto.",
          data_port.name()));
    }
  }
  absl::flat_hash_map<std::string,
                      std::pair<const FifoConfigProto*, const TypeProto*>>
      fifo_channels;
  for (const InstantiationProto& instantiation : proto.instantiations()) {
    switch (instantiation.instantiation_oneof_case()) {
      case InstantiationProto::kBlockInstantiationFieldNumber:
      case InstantiationProto::kExternInstantiationFieldNumber:
      case InstantiationProto::INSTANTIATION_ONEOF_NOT_SET: {
        break;
      }
      case InstantiationProto::kFifoInstantiationFieldNumber: {
        auto [_, inserted] = fifo_channels.insert(
            {instantiation.fifo_instantiation().channel_name(),
             std::make_pair(&instantiation.fifo_instantiation().fifo_config(),
                            &instantiation.fifo_instantiation().type())});
        if (!inserted) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Multiple FIFOs for channel %s",
              instantiation.fifo_instantiation().channel_name()));
        }
      }
    }
  }

  auto is_same_direction = [](ChannelDirectionProto a, PortDirectionProto b) {
    return (a == CHANNEL_DIRECTION_RECEIVE && b == PORT_DIRECTION_INPUT) ||
           (a == CHANNEL_DIRECTION_SEND && b == PORT_DIRECTION_OUTPUT);
  };
  auto is_opposite_direction = [](ChannelDirectionProto a,
                                  PortDirectionProto b) {
    return (a == CHANNEL_DIRECTION_RECEIVE && b == PORT_DIRECTION_OUTPUT) ||
           (a == CHANNEL_DIRECTION_SEND && b == PORT_DIRECTION_INPUT);
  };

  absl::flat_hash_set<std::string> channel_ports_seen;
  for (const ChannelInterfaceProto& channel : proto.channel_interfaces()) {
    if (!channel.has_channel_name() || channel.channel_name().empty()) {
      return absl::InvalidArgumentError("A name is required for all channels.");
    }

    if (channel.kind() == ChannelKindProto::CHANNEL_KIND_INVALID) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel '%s' has an invalid kind.", channel.channel_name()));
    }

    // Ensure the specified ports for the channel exist in the port list.
    if (channel.has_data_port_name()) {
      if (!name_data_ports_map.contains(channel.data_port_name())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Port '%s' of channel '%s' is not present in the port list.",
            channel.data_port_name(), channel.channel_name()));
      }
      if (!is_same_direction(
              channel.direction(),
              name_data_ports_map.at(channel.data_port_name()).direction())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Data port '%s' of channel '%s' is not the correct direction",
            channel.data_port_name(), channel.channel_name()));
      }
      auto [_, inserted] = channel_ports_seen.insert(channel.data_port_name());
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Port '%s' is used by multiple channels.",
                            channel.data_port_name()));
      }
    }
    if (channel.has_valid_port_name()) {
      if (!name_data_ports_map.contains(channel.valid_port_name())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Port '%s' of channel '%s' is not present in the port list.",
            channel.valid_port_name(), channel.channel_name()));
      }
      if (!is_same_direction(
              channel.direction(),
              name_data_ports_map.at(channel.valid_port_name()).direction())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Valid port '%s' of channel '%s' is not the correct direction",
            channel.valid_port_name(), channel.channel_name()));
      }
      auto [_, inserted] = channel_ports_seen.insert(channel.valid_port_name());
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Port '%s' is used by multiple channels.",
                            channel.valid_port_name()));
      }
    }
    if (channel.has_ready_port_name()) {
      if (!name_data_ports_map.contains(channel.ready_port_name())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Port '%s' of channel '%s' is not present in the port list.",
            channel.ready_port_name(), channel.channel_name()));
      }
      if (!is_opposite_direction(
              channel.direction(),
              name_data_ports_map.at(channel.ready_port_name()).direction())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Ready port '%s' of channel '%s' is not the correct direction",
            channel.ready_port_name(), channel.channel_name()));
      }
      auto [_, inserted] = channel_ports_seen.insert(channel.ready_port_name());
      if (!inserted) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Port '%s' is used by multiple channels.",
                            channel.ready_port_name()));
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<ModuleSignature> ModuleSignatureBuilder::Build() {
  XLS_RETURN_IF_ERROR(ValidateProto(proto_));
  return ModuleSignature::FromProto(proto_);
}

/* static */ absl::StatusOr<ModuleSignature> ModuleSignature::FromProto(
    const ModuleSignatureProto& proto) {
  XLS_RETURN_IF_ERROR(ValidateProto(proto));

  ModuleSignature signature;
  signature.proto_ = proto;
  for (const PortProto& port : proto.data_ports()) {
    if (port.direction() == PORT_DIRECTION_INPUT) {
      signature.data_inputs_.push_back(port);
    } else if (port.direction() == PORT_DIRECTION_OUTPUT) {
      signature.data_outputs_.push_back(port);
    } else {
      return absl::InvalidArgumentError("Invalid port direction.");
    }
  }

  for (const RamProto& ram_ports : proto.rams()) {
    signature.rams_.push_back(ram_ports);
  }

  for (const InstantiationProto& instantiation : proto.instantiations()) {
    signature.instantiations_.push_back(instantiation);
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
  auto iter = absl::c_find_if(proto_.channel_interfaces(),
                              [&](const ChannelInterfaceProto& channel) {
                                return channel.channel_name() == channel_name;
                              });
  if (iter == proto_.channel_interfaces().end() ||
      iter->direction() != CHANNEL_DIRECTION_RECEIVE) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Channel '%s' is not an input channel.", channel_name));
  }
  XLS_ASSIGN_OR_RETURN(PortProto port,
                       GetInputPortByName(iter->data_port_name()));
  for (const Bits& value : values) {
    if (port.width() != value.bit_count()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected input '%s' to have width %d, has width % d ", port.name(),
          port.width(), value.bit_count()));
    }
  }
  return absl::OkStatus();
}

absl::Status ModuleSignature::ValidateChannelValueInputs(
    std::string_view channel_name, absl::Span<const Value> values) const {
  auto iter = absl::c_find_if(proto_.channel_interfaces(),
                              [&](const ChannelInterfaceProto& channel) {
                                return channel.channel_name() == channel_name;
                              });
  if (iter == proto_.channel_interfaces().end() ||
      iter->direction() != CHANNEL_DIRECTION_RECEIVE) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Channel '%s' is not an input channel.", channel_name));
  }
  const TypeProto& expected_type_proto = iter->type();
  for (const Value& value : values) {
    XLS_ASSIGN_OR_RETURN(TypeProto value_type_proto, value.TypeAsProto());
    XLS_ASSIGN_OR_RETURN(bool types_equal, TypeProtosEqual(expected_type_proto,
                                                           value_type_proto));
    if (!types_equal) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input value '%s' is wrong type. Expected '%s', got '%s'",
          iter->data_port_name(), TypeProtoToString(expected_type_proto),
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

absl::StatusOr<PortProto> ModuleSignature::GetInputPortByName(
    std::string_view name) const {
  auto iter = absl::c_find_if(
      data_inputs_, [&](const PortProto& port) { return port.name() == name; });
  if (iter == data_inputs_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Port '%s' is not an input port.", name));
  }
  return *iter;
}

absl::StatusOr<PortProto> ModuleSignature::GetOutputPortByName(
    std::string_view name) const {
  auto iter = absl::c_find_if(data_outputs_, [&](const PortProto& port) {
    return port.name() == name;
  });
  if (iter == data_outputs_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Port '%s' is not an output port.", name));
  }
  return *iter;
}

std::vector<ChannelInterfaceProto> ModuleSignature::GetInputChannelInterfaces()
    const {
  std::vector<ChannelInterfaceProto> channels;
  for (const ChannelInterfaceProto& channel : proto_.channel_interfaces()) {
    if (channel.direction() == CHANNEL_DIRECTION_RECEIVE) {
      channels.push_back(channel);
    }
  }
  return channels;
}

std::vector<ChannelInterfaceProto> ModuleSignature::GetOutputChannelInterfaces()
    const {
  std::vector<ChannelInterfaceProto> channels;
  for (const ChannelInterfaceProto& channel : proto_.channel_interfaces()) {
    if (channel.direction() == CHANNEL_DIRECTION_SEND) {
      channels.push_back(channel);
    }
  }
  return channels;
}

absl::StatusOr<ChannelInterfaceProto>
ModuleSignature::GetChannelInterfaceByName(
    std::string_view channel_name) const {
  auto iter =
      absl::c_find_if(proto_.channel_interfaces(),
                      [&](const ChannelInterfaceProto& channel_interface) {
                        return channel_interface.channel_name() == channel_name;
                      });
  if (iter == proto_.channel_interfaces().end()) {
    return absl::NotFoundError(
        absl::StrFormat("Interface channel '%s' not found.", channel_name));
  }
  return *iter;
}

absl::StatusOr<std::string> ModuleSignature::GetChannelInterfaceNameForPort(
    std::string_view port_name) const {
  auto iter = absl::c_find_if(
      proto_.channel_interfaces(),
      [&](const ChannelInterfaceProto& channel_interface) {
        return channel_interface.data_port_name() == port_name ||
               channel_interface.ready_port_name() == port_name ||
               channel_interface.valid_port_name() == port_name;
      });
  if (iter == proto_.channel_interfaces().end()) {
    return absl::NotFoundError(absl::StrFormat(
        "No port named `%s` or port is not associated with a channel.",
        port_name));
  }
  return iter->channel_name();
}

std::vector<ChannelProto> ModuleSignature::GetChannels() {
  std::vector<ChannelProto> channels;
  for (const ChannelProto& channel : proto_.channels()) {
    channels.push_back(channel);
  }
  return channels;
}

std::vector<ChannelInterfaceProto> ModuleSignature::GetChannelInterfaces() {
  std::vector<ChannelInterfaceProto> channels;
  for (const ChannelInterfaceProto& channel : proto_.channel_interfaces()) {
    channels.push_back(channel);
  }
  return channels;
}

absl::StatusOr<ChannelProto> ModuleSignature::GetChannel(
    std::string_view channel_name) const {
  auto iter =
      absl::c_find_if(proto_.channels(), [&](const ChannelProto& channel) {
        return channel.name() == channel_name;
      });
  if (iter == proto_.channels().end()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel named `%s`.", channel_name));
  }
  return *iter;
}

absl::StatusOr<FifoInstantiationProto> ModuleSignature::GetFifoInstantiation(
    std::string_view instance_name) {
  auto iter = absl::c_find_if(
      proto_.instantiations(), [&](const InstantiationProto& instantiation) {
        return instantiation.has_fifo_instantiation() &&
               instantiation.fifo_instantiation().instance_name() ==
                   instance_name;
      });
  if (iter == proto_.instantiations().end()) {
    return absl::NotFoundError(
        absl::StrFormat("No fifo instance named `%s`.", instance_name));
  }
  return iter->fifo_instantiation();
}

absl::StatusOr<BlockInstantiationProto> ModuleSignature::GetBlockInstantiation(
    std::string_view instance_name) {
  auto iter = absl::c_find_if(
      proto_.instantiations(), [&](const InstantiationProto& instantiation) {
        return instantiation.has_block_instantiation() &&
               instantiation.block_instantiation().instance_name() ==
                   instance_name;
      });
  if (iter == proto_.instantiations().end()) {
    return absl::NotFoundError(
        absl::StrFormat("No fifo instance named `%s`.", instance_name));
  }
  return iter->block_instantiation();
}

std::ostream& operator<<(std::ostream& os, const ModuleSignature& signature) {
  os << signature.ToString();
  return os;
}

}  // namespace verilog
}  // namespace xls
