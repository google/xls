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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
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
    Type* type, std::optional<FifoConfig> fifo_config,
    std::string_view port_name, std::optional<std::string_view> valid_port_name,
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
  *channel->mutable_type() = type->ToProto();

  if (flow_control == FlowControl::kReadyValid) {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_READY_VALID);
  } else {
    channel->set_flow_control(CHANNEL_FLOW_CONTROL_NONE);
  }

  if (fifo_config.has_value()) {
    *channel->mutable_fifo_config() =
        fifo_config->ToProto(type->GetFlatBitCount());
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
  address_proto->set_direction(DIRECTION_OUTPUT);
  address_proto->set_width(args.address_width);
  *address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* read_enable_proto = req->mutable_read_enable();
  read_enable_proto->set_name(ToProtoString(args.read_enable_name));
  read_enable_proto->set_direction(DIRECTION_OUTPUT);
  read_enable_proto->set_width(1);
  *read_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* write_enable_proto = req->mutable_write_enable();
  write_enable_proto->set_name(ToProtoString(args.write_enable_name));
  write_enable_proto->set_direction(DIRECTION_OUTPUT);
  write_enable_proto->set_width(1);
  *write_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* write_data_proto = req->mutable_write_data();
  write_data_proto->set_name(ToProtoString(args.write_data_name));
  write_data_proto->set_direction(DIRECTION_OUTPUT);
  write_data_proto->set_width(data_width);
  *write_data_proto->mutable_type() = args.data_type->ToProto();

  auto* read_data_proto = resp->mutable_read_data();
  read_data_proto->set_name(ToProtoString(args.read_data_name));
  read_data_proto->set_direction(DIRECTION_INPUT);
  read_data_proto->set_width(data_width);
  *read_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.write_mask_width > 0) {
    auto* write_mask_proto = req->mutable_write_mask();
    write_mask_proto->set_name(ToProtoString(args.write_mask_name));
    write_mask_proto->set_direction(DIRECTION_OUTPUT);
    write_mask_proto->set_width(args.write_mask_width);
    *write_mask_proto->mutable_type() =
        args.package->GetBitsType(args.write_mask_width)->ToProto();
  }

  if (args.read_mask_width > 0) {
    auto* read_mask_proto = req->mutable_read_mask();
    read_mask_proto->set_name(ToProtoString(args.read_mask_name));
    read_mask_proto->set_direction(DIRECTION_OUTPUT);
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
  rd_address_proto->set_direction(DIRECTION_OUTPUT);
  rd_address_proto->set_width(args.address_width);
  *rd_address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* rd_enable_proto = r_req->mutable_enable();
  rd_enable_proto->set_name(ToProtoString(args.read_enable_name));
  rd_enable_proto->set_direction(DIRECTION_OUTPUT);
  rd_enable_proto->set_width(1);
  *rd_enable_proto->mutable_type() = args.package->GetBitsType(1)->ToProto();

  auto* rd_data_proto = r_resp->mutable_data();
  rd_data_proto->set_name(ToProtoString(args.read_data_name));
  rd_data_proto->set_direction(DIRECTION_INPUT);
  rd_data_proto->set_width(data_width);
  *rd_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.read_mask_width > 0) {
    auto* read_mask_proto = w_req->mutable_mask();
    read_mask_proto->set_name(ToProtoString(args.read_mask_name));
    read_mask_proto->set_direction(DIRECTION_OUTPUT);
    read_mask_proto->set_width(args.read_mask_width);
    *read_mask_proto->mutable_type() =
        args.package->GetBitsType(args.read_mask_width)->ToProto();
  }

  auto* wr_address_proto = w_req->mutable_address();
  wr_address_proto->set_name(ToProtoString(args.write_address_name));
  wr_address_proto->set_direction(DIRECTION_OUTPUT);
  wr_address_proto->set_width(args.address_width);
  *wr_address_proto->mutable_type() =
      args.package->GetBitsType(args.address_width)->ToProto();

  auto* wr_data_proto = w_req->mutable_data();
  wr_data_proto->set_name(ToProtoString(args.write_data_name));
  wr_data_proto->set_direction(DIRECTION_OUTPUT);
  wr_data_proto->set_width(data_width);
  *wr_data_proto->mutable_type() = args.data_type->ToProto();

  if (args.write_mask_width > 0) {
    auto* write_mask_proto = w_req->mutable_mask();
    write_mask_proto->set_name(ToProtoString(args.write_mask_name));
    write_mask_proto->set_direction(DIRECTION_OUTPUT);
    write_mask_proto->set_width(args.write_mask_width);
    *write_mask_proto->mutable_type() =
        args.package->GetBitsType(args.write_mask_width)->ToProto();
  }

  auto* wr_enable_proto = w_req->mutable_enable();
  wr_enable_proto->set_name(ToProtoString(args.write_enable_name));
  wr_enable_proto->set_direction(DIRECTION_OUTPUT);
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
    if (port.direction() == DirectionProto::DIRECTION_INVALID) {
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
    // Ignore channel that have FIFOs as they won't be in the port list
    if (!fifo_channels.contains(channel.name())) {
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
      const PortProto& data_port =
          name_data_ports_map[channel.data_port_name()];
      if (channel.has_valid_port_name()) {
        const PortProto& valid_port =
            name_data_ports_map[channel.valid_port_name()];
        if (data_port.direction() != valid_port.direction()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "For channel '%s', data port '%s' and valid port '%s' must have "
              "the same direction.",
              channel.name(), data_port.name(), valid_port.name()));
        }
        if (valid_port.width() != 1) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "For channel '%s', valid port '%s' must have a width of 1 bit "
              "(got "
              "%d).",
              channel.name(), channel.valid_port_name(), valid_port.width()));
        }
      }
      if (channel.has_ready_port_name()) {
        const PortProto& ready_port =
            name_data_ports_map[channel.valid_port_name()];
        if (data_port.direction() != ready_port.direction()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "For channel '%s', data port '%s' and ready port '%s' must have "
              "the same direction.",
              channel.name(), data_port.name(), ready_port.name()));
        }
        if (ready_port.width() != 1) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "For channel '%s', ready port '%s' must have a width of 1 bit.",
              channel.name(), ready_port.name()));
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
