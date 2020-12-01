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

  explicit NetworkConfigProtoBuilder(absl::string_view network_name);

  // Sets the description of the network.
  NetworkConfigProtoBuilder& WithDescription(absl::string_view description);

  // Sets default flit bit width for links.
  NetworkConfigProtoBuilder& SetDefaultLinkPhitBitWidth(
      absl::optional<int64> phit_bit_width);

  // Sets default source to sink pipeline stage for links.
  NetworkConfigProtoBuilder& SetDefaultLinkSourceSinkPipelineStage(
      absl::optional<int64> pipeline_stage);

  // Sets default sink to source pipeline stage for links.
  NetworkConfigProtoBuilder& SetDefaultLinkSinkSourcePipelineStage(
      absl::optional<int64> pipeline_stage);

  // Sets default flit bit width for Virtual Channels.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelFlitBitWidth(
      absl::optional<int64> flit_bit_width);

  // Sets default depth for Virtual Channels.
  NetworkConfigProtoBuilder& SetDefaultVirtualChannelDepth(
      absl::optional<int64> depth);

  // Adds a port to this object. Returns its builder.
  PortConfigProtoBuilder WithPort(absl::string_view name);

  // Adds a router to this object. Returns its builder.
  RouterConfigProtoBuilder WithRouter(absl::string_view name);

  // Adds a link to this object. Returns its builder.
  LinkConfigProtoBuilder WithLink(absl::string_view name);

  // Adds a virtual channel to this object. Returns its builder.
  VirtualChannelConfigProtoBuilder WithVirtualChannel(absl::string_view name);

  // If the builder defines a valid proto, returns the proto. Otherwise, returns
  // an error.
  absl::StatusOr<NetworkConfigProto> Build();

 private:
  NetworkConfigProto proto_;
  // The members below are used to preserve the default state for the builder.
  // See corresponding methods above for details.
  absl::optional<int64> link_phit_bit_width_;
  absl::optional<int64> link_source_sink_pipeline_stage_;
  absl::optional<int64> link_sink_source_pipeline_stage_;
  absl::optional<int64> virtual_channel_flit_bit_width_;
  absl::optional<int64> virtual_channel_depth_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_H_
