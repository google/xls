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

// Builder definitions for the common network config options proto.

#ifndef XLS_NOC_CONFIG_COMMON_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_COMMON_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_

#include <cstdint>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_builder_options.pb.h"

namespace xls::noc {

// A builder for constructing a endpoint options proto.
class EndpointOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit EndpointOptionsProtoBuilder(EndpointOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the number of send ports.
  EndpointOptionsProtoBuilder& WithNumSendPorts(int64_t number_send_ports);

  // Adds the number of receive ports.
  EndpointOptionsProtoBuilder& WithNumRecvPorts(int64_t number_recv_ports);

 private:
  EndpointOptionsProto* proto_;
};

// A builder for constructing a data options proto.
class DataOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit DataOptionsProtoBuilder(DataOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the data bit width.
  DataOptionsProtoBuilder& WithDataBitWidth(int64_t data_bit_width);

 private:
  DataOptionsProto* proto_;
};

// A builder for constructing a flow control options proto.
class FlowControlOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit FlowControlOptionsProtoBuilder(
      LinkConfigProto::FlowControlConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // When building the flow control options proto, the last flow control
  // mechanism triggered by the user is enabled.

  // Enables the peek flow control option.
  FlowControlOptionsProtoBuilder& EnablePeekFlowControl();

  // Enables the token credit-based flow control option.
  FlowControlOptionsProtoBuilder& EnableTokenCreditBasedFlowControl();

  // Enables the total available credit-based flow control option.
  FlowControlOptionsProtoBuilder& EnableTotalCreditBasedFlowControl(
      int64_t credit_bit_width);

 private:
  LinkConfigProto::FlowControlConfigProto* proto_;
};

// A builder for constructing a link options proto.
class LinkOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit LinkOptionsProtoBuilder(LinkOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the pipeline stage for source to sink.
  LinkOptionsProtoBuilder& WithSourceSinkPipelineStage(int64_t pipeline_stage);

  // Adds the pipeline stage for sink to source.
  LinkOptionsProtoBuilder& WithSinkSourcePipelineStage(int64_t pipeline_stage);

  // Returns the flow control options proto builder of this builder.
  FlowControlOptionsProtoBuilder GetFlowControlOptionsProtoBuilder();

 private:
  LinkOptionsProto* proto_;
};

// A builder for constructing a virtual channel options proto.
class VirtualChannelOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit VirtualChannelOptionsProtoBuilder(VirtualChannelOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds a virtual channel with depth.
  VirtualChannelOptionsProtoBuilder& WithVirtualChannelDepth(int64_t depth);

 private:
  VirtualChannelOptionsProto* proto_;
};

// A builder for constructing a routing scheme options proto.
class RoutingSchemeOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit RoutingSchemeOptionsProtoBuilder(
      RouterOptionsProto::RoutingSchemeOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Enables distributed routing.
  RoutingSchemeOptionsProtoBuilder& EnableDistributedRouting();

 private:
  RouterOptionsProto::RoutingSchemeOptionsProto* proto_;
};

// A builder for constructing an arbiter scheme options proto.
class ArbiterSchemeOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit ArbiterSchemeOptionsProtoBuilder(
      RouterOptionsProto::ArbiterSchemeOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Enables distributed routing.
  ArbiterSchemeOptionsProtoBuilder& EnablePriority();

 private:
  RouterOptionsProto::ArbiterSchemeOptionsProto* proto_;
};

// A builder for constructing a router options proto.
class RouterOptionsProtoBuilder {
 public:
  // Pointer cannot be nullptr.
  explicit RouterOptionsProtoBuilder(RouterOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Returns the routing scheme options proto builder of this builder.
  RoutingSchemeOptionsProtoBuilder GetRoutingSchemeOptionsProtoBuilder();

  // Returns the arbiter scheme options proto builder of this builder.
  ArbiterSchemeOptionsProtoBuilder GetArbiterSchemeOptionsProtoBuilder();

 private:
  RouterOptionsProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_COMMON_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_
