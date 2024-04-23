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

#include "xls/noc/config/common_network_config_builder_options_proto_builder.h"

#include <cstdint>

namespace xls::noc {

ArbiterSchemeOptionsProtoBuilder&
ArbiterSchemeOptionsProtoBuilder::EnablePriority() {
  proto_->mutable_priority();
  return *this;
}

EndpointOptionsProtoBuilder& EndpointOptionsProtoBuilder::WithNumSendPorts(
    int64_t number_send_ports) {
  proto_->set_num_send_ports(number_send_ports);
  return *this;
}

EndpointOptionsProtoBuilder& EndpointOptionsProtoBuilder::WithNumRecvPorts(
    int64_t number_recv_ports) {
  proto_->set_num_recv_ports(number_recv_ports);
  return *this;
}

DataOptionsProtoBuilder& DataOptionsProtoBuilder::WithDataBitWidth(
    int64_t data_bit_width) {
  proto_->set_data_bit_width(data_bit_width);
  return *this;
}

FlowControlOptionsProtoBuilder&
FlowControlOptionsProtoBuilder::EnablePeekFlowControl() {
  proto_->mutable_peek();
  return *this;
}

FlowControlOptionsProtoBuilder&
FlowControlOptionsProtoBuilder::EnableTokenCreditBasedFlowControl() {
  proto_->mutable_token_credit_based();
  return *this;
}

FlowControlOptionsProtoBuilder&
FlowControlOptionsProtoBuilder::EnableTotalCreditBasedFlowControl(
    int64_t credit_bit_width) {
  proto_->mutable_total_credit_based()->set_credit_bit_width(credit_bit_width);
  return *this;
}

LinkOptionsProtoBuilder& LinkOptionsProtoBuilder::WithSourceSinkPipelineStage(
    int64_t pipeline_stage) {
  proto_->set_source_sink_pipeline_stage(pipeline_stage);
  return *this;
}

LinkOptionsProtoBuilder& LinkOptionsProtoBuilder::WithSinkSourcePipelineStage(
    int64_t pipeline_stage) {
  proto_->set_sink_source_pipeline_stage(pipeline_stage);
  return *this;
}

FlowControlOptionsProtoBuilder
LinkOptionsProtoBuilder::GetFlowControlOptionsProtoBuilder() {
  return FlowControlOptionsProtoBuilder(proto_->mutable_flow_control());
}

VirtualChannelOptionsProtoBuilder&
VirtualChannelOptionsProtoBuilder::WithVirtualChannelDepth(int64_t depth) {
  proto_->add_virtual_channel_depth(depth);
  return *this;
}

RoutingSchemeOptionsProtoBuilder&
RoutingSchemeOptionsProtoBuilder::EnableDistributedRouting() {
  proto_->mutable_routing_table();
  return *this;
}

RoutingSchemeOptionsProtoBuilder
RouterOptionsProtoBuilder::GetRoutingSchemeOptionsProtoBuilder() {
  return RoutingSchemeOptionsProtoBuilder(proto_->mutable_routing_scheme());
}

ArbiterSchemeOptionsProtoBuilder
RouterOptionsProtoBuilder::GetArbiterSchemeOptionsProtoBuilder() {
  return ArbiterSchemeOptionsProtoBuilder(proto_->mutable_arbiter_scheme());
}

}  //  namespace xls::noc
