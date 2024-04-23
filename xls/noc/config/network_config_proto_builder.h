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

#ifndef XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/noc/config/link_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/port_config_proto_builder.h"
#include "xls/noc/config/router_config_proto_builder.h"
#include "xls/noc/config/virtual_channel_config_proto_builder.h"

namespace xls::noc {

// A builder for constructing a network configuration proto.
class NetworkConfigProtoBuilder {
 public:
  NetworkConfigProtoBuilder() = default;

  explicit NetworkConfigProtoBuilder(std::string_view network_name);

  // Sets the description of the network.
  NetworkConfigProtoBuilder& WithDescription(std::string_view description);

  // Sets default flit bit width for links.
  NetworkConfigProtoBuilder& SetDefaultLinkPhitBitWidth(
      std::optional<int64_t> phit_bit_width);

  // Sets default source to sink pipeline stage for links.
  NetworkConfigProtoBuilder& SetDefaultLinkSourceSinkPipelineStage(
      std::optional<int64_t> pipeline_stage);

  // Sets default sink to source pipeline stage for links.
  NetworkConfigProtoBuilder& SetDefaultLinkSinkSourcePipelineStage(
      std::optional<int64_t> pipeline_stage);

  // Sets default flit bit width for Virtual Channels.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelFlitBitWidth(
      std::optional<int64_t> flit_bit_width);

  // Sets default depth for Virtual Channels.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelDepth(
      std::optional<int64_t> depth);

  // Adds a port to this object. Returns its builder.
  PortConfigProtoBuilder WithPort(std::string_view name);

  // Sets default Virtual Channels for input ports.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelsForRouterInputPort(
      std::optional<std::vector<std::string>> virtual_channels);

  // Sets default Virtual Channels for output ports.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelsForRouterOutputPort(
      std::optional<std::vector<std::string>> virtual_channels);

  // Adds a router to this object. Returns its builder.
  RouterConfigProtoBuilder WithRouter(std::string_view name);

  // Adds a link to this object. Returns its builder.
  LinkConfigProtoBuilder WithLink(std::string_view name);

  // Adds a virtual channel to this object. Returns its builder.
  VirtualChannelConfigProtoBuilder WithVirtualChannel(std::string_view name);

  // If the builder defines a valid proto, returns the proto. Otherwise, returns
  // an error.
  absl::StatusOr<NetworkConfigProto> Build();

 private:
  NetworkConfigProto proto_;
  // The members below are used to preserve the default state for the builder.
  // See corresponding methods above for details.
  std::optional<int64_t> link_phit_bit_width_;
  std::optional<int64_t> link_source_sink_pipeline_stage_;
  std::optional<int64_t> link_sink_source_pipeline_stage_;
  std::optional<int64_t> virtual_channel_flit_bit_width_;
  std::optional<int64_t> virtual_channel_depth_;
  std::optional<std::vector<std::string>> virtual_channels_for_input_;
  std::optional<std::vector<std::string>> virtual_channels_for_output_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_H_
