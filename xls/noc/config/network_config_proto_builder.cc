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

#include "xls/noc/config/network_config_proto_builder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/noc/config/link_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/port_config_proto_builder.h"
#include "xls/noc/config/router_config_proto_builder.h"
#include "xls/noc/config/virtual_channel_config_proto_builder.h"

namespace xls::noc {

NetworkConfigProtoBuilder::NetworkConfigProtoBuilder(std::string_view name) {
  proto_.set_name(xls::ToProtoString(name));
}

NetworkConfigProtoBuilder& NetworkConfigProtoBuilder::WithDescription(
    std::string_view description) {
  proto_.set_description(xls::ToProtoString(description));
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultLinkPhitBitWidth(
    std::optional<int64_t> phit_bit_width) {
  link_phit_bit_width_ = phit_bit_width;
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultLinkSourceSinkPipelineStage(
    std::optional<int64_t> pipeline_stage) {
  link_source_sink_pipeline_stage_ = pipeline_stage;
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultLinkSinkSourcePipelineStage(
    std::optional<int64_t> pipeline_stage) {
  link_sink_source_pipeline_stage_ = pipeline_stage;
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultVirtualChannelFlitBitWidth(
    std::optional<int64_t> flit_bit_width) {
  virtual_channel_flit_bit_width_ = flit_bit_width;
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultVirtualChannelDepth(
    std::optional<int64_t> depth) {
  virtual_channel_depth_ = depth;
  return *this;
}

PortConfigProtoBuilder NetworkConfigProtoBuilder::WithPort(
    std::string_view name) {
  PortConfigProto* port = proto_.add_ports();
  port->set_name(xls::ToProtoString(name));
  return PortConfigProtoBuilder(port);
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultVirtualChannelsForRouterInputPort(
    std::optional<std::vector<std::string>> virtual_channels) {
  virtual_channels_for_input_ = virtual_channels;
  return *this;
}

NetworkConfigProtoBuilder&
NetworkConfigProtoBuilder::SetDefaultVirtualChannelsForRouterOutputPort(
    std::optional<std::vector<std::string>> virtual_channels) {
  virtual_channels_for_output_ = virtual_channels;
  return *this;
}

RouterConfigProtoBuilder NetworkConfigProtoBuilder::WithRouter(
    std::string_view name) {
  RouterConfigProto* router = proto_.add_routers();
  router->set_name(xls::ToProtoString(name));
  RouterConfigProtoBuilder router_builder(router);
  router_builder
      .SetDefaultVirtualChannelsForInputPort(virtual_channels_for_input_)
      .SetDefaultVirtualChannelsForOutputPort(virtual_channels_for_output_);
  return router_builder;
}

LinkConfigProtoBuilder NetworkConfigProtoBuilder::WithLink(
    std::string_view name) {
  LinkConfigProto* link = proto_.add_links();
  link->set_name(xls::ToProtoString(name));
  LinkConfigProtoBuilder link_builder(link);
  if (link_phit_bit_width_.has_value()) {
    link_builder.WithPhitBitWidth(link_phit_bit_width_.value());
  }
  if (link_source_sink_pipeline_stage_.has_value()) {
    link_builder.WithSourceSinkPipelineStage(
        link_source_sink_pipeline_stage_.value());
  }
  if (link_sink_source_pipeline_stage_.has_value()) {
    link_builder.WithSinkSourcePipelineStage(
        link_sink_source_pipeline_stage_.value());
  }
  return link_builder;
}

VirtualChannelConfigProtoBuilder NetworkConfigProtoBuilder::WithVirtualChannel(
    std::string_view name) {
  VirtualChannelConfigProto* virtual_channel = proto_.add_virtual_channels();
  virtual_channel->set_name(xls::ToProtoString(name));
  VirtualChannelConfigProtoBuilder virtual_channel_builder(virtual_channel);
  if (virtual_channel_flit_bit_width_.has_value()) {
    virtual_channel_builder.WithFlitBitWidth(
        virtual_channel_flit_bit_width_.value());
  }
  if (virtual_channel_depth_.has_value()) {
    virtual_channel_builder.WithDepth(virtual_channel_depth_.value());
  }
  return virtual_channel_builder;
}

absl::StatusOr<NetworkConfigProto> NetworkConfigProtoBuilder::Build() {
  return proto_;
}

}  // namespace xls::noc
