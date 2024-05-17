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

#include "xls/noc/config_ng/custom_topology_options_proto_builder.h"

#include "absl/log/die_if_null.h"
#include "xls/noc/config_ng/channel_options_proto_builder.h"
#include "xls/noc/config_ng/node_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {

CustomTopologyOptionsProtoBuilder::CustomTopologyOptionsProtoBuilder(
    CustomTopologyOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

CustomTopologyOptionsProtoBuilder::CustomTopologyOptionsProtoBuilder(
    CustomTopologyOptionsProto* proto_ptr,
    const CustomTopologyOptionsProto& default_proto)
    : CustomTopologyOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

CustomTopologyOptionsProtoBuilder& CustomTopologyOptionsProtoBuilder::CopyFrom(
    const CustomTopologyOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}
CustomTopologyOptionsProtoBuilder& CustomTopologyOptionsProtoBuilder::AddNode(
    const NodeOptionsProtoBuilder& builder) {
  NodeOptionsProtoBuilder(proto_ptr_->add_nodes()).CopyFrom(builder);
  return *this;
}

NodeOptionsProtoBuilder CustomTopologyOptionsProtoBuilder::AddNode() {
  return NodeOptionsProtoBuilder(proto_ptr_->add_nodes());
}

CustomTopologyOptionsProtoBuilder&
CustomTopologyOptionsProtoBuilder::AddChannel(
    const ChannelOptionsProtoBuilder& builder) {
  ChannelOptionsProtoBuilder(proto_ptr_->add_channels()).CopyFrom(builder);
  return *this;
}

ChannelOptionsProtoBuilder CustomTopologyOptionsProtoBuilder::AddChannel() {
  return ChannelOptionsProtoBuilder(proto_ptr_->add_channels());
}

}  // namespace xls::noc
