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

#include "xls/noc/config_ng/bidirectional_butterfly_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {

BidirectionalButterflyOptionsProtoBuilder::
    BidirectionalButterflyOptionsProtoBuilder(
        BidirectionalButterflyOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

BidirectionalButterflyOptionsProtoBuilder::
    BidirectionalButterflyOptionsProtoBuilder(
        BidirectionalButterflyOptionsProto* proto_ptr,
        const BidirectionalButterflyOptionsProto& default_proto)
    : BidirectionalButterflyOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

BidirectionalButterflyOptionsProtoBuilder&
BidirectionalButterflyOptionsProtoBuilder::CopyFrom(
    const BidirectionalButterflyOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

BidirectionalButterflyOptionsProtoBuilder&
BidirectionalButterflyOptionsProtoBuilder::SetEndpointConnection(
    const BidirectionalButterflyOptionsProto::EndpointConnection
        endpoint_connection) {
  proto_ptr_->set_endpoint_connection(endpoint_connection);
  return *this;
}

BidirectionalButterflyOptionsProtoBuilder&
BidirectionalButterflyOptionsProtoBuilder::
    SetEndpointConnectionToConnectToFirstStage() {
  proto_ptr_->set_endpoint_connection(
      BidirectionalButterflyOptionsProto::CONNECT_TO_FIRST_STAGE);
  return *this;
}

BidirectionalButterflyOptionsProtoBuilder&
BidirectionalButterflyOptionsProtoBuilder::
    SetEndpointConnectionToConnectToLastStage() {
  proto_ptr_->set_endpoint_connection(
      BidirectionalButterflyOptionsProto::CONNECT_TO_LAST_STAGE);
  return *this;
}

}  // namespace xls::noc
