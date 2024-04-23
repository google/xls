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

#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"

#include <cstdint>

#include "absl/log/die_if_null.h"

namespace xls::noc {

TopologyEndpointOptionsProtoBuilder::TopologyEndpointOptionsProtoBuilder(
    TopologyEndpointOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

TopologyEndpointOptionsProtoBuilder::TopologyEndpointOptionsProtoBuilder(
    TopologyEndpointOptionsProto* proto_ptr,
    const TopologyEndpointOptionsProto& default_proto)
    : TopologyEndpointOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

TopologyEndpointOptionsProtoBuilder&
TopologyEndpointOptionsProtoBuilder::CopyFrom(
    const TopologyEndpointOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

TopologyEndpointOptionsProtoBuilder&
TopologyEndpointOptionsProtoBuilder::SetSendPortCount(
    const int64_t send_port_count) {
  proto_ptr_->set_send_port_count(send_port_count);
  return *this;
}

TopologyEndpointOptionsProtoBuilder&
TopologyEndpointOptionsProtoBuilder::SetRecvPortCount(
    const int64_t recv_port_count) {
  proto_ptr_->set_recv_port_count(recv_port_count);
  return *this;
}

}  // namespace xls::noc
