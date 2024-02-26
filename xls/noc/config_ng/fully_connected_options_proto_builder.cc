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

#include "xls/noc/config_ng/fully_connected_options_proto_builder.h"

#include "absl/log/die_if_null.h"
#include "xls/common/logging/logging.h"

namespace xls::noc {

FullyConnectedOptionsProtoBuilder::FullyConnectedOptionsProtoBuilder(
    FullyConnectedOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

FullyConnectedOptionsProtoBuilder::FullyConnectedOptionsProtoBuilder(
    FullyConnectedOptionsProto* proto_ptr,
    const FullyConnectedOptionsProto& default_proto)
    : FullyConnectedOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

FullyConnectedOptionsProtoBuilder& FullyConnectedOptionsProtoBuilder::CopyFrom(
    const FullyConnectedOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

FullyConnectedOptionsProtoBuilder&
FullyConnectedOptionsProtoBuilder::SetRouterCount(const int64_t router_count) {
  proto_ptr_->set_router_count(router_count);
  return *this;
}

FullyConnectedOptionsProtoBuilder&
FullyConnectedOptionsProtoBuilder::SetEndpointOptions(
    const TopologyEndpointOptionsProtoBuilder& builder) {
  TopologyEndpointOptionsProtoBuilder(proto_ptr_->mutable_endpoint_options())
      .CopyFrom(builder);
  return *this;
}

TopologyEndpointOptionsProtoBuilder
FullyConnectedOptionsProtoBuilder::GetEndpointOptions() {
  return TopologyEndpointOptionsProtoBuilder(
      proto_ptr_->mutable_endpoint_options());
}

}  // namespace xls::noc
