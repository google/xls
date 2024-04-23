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

#include "xls/noc/config_ng/bidirectional_tree_options_proto_builder.h"

#include <cstdint>

#include "absl/log/die_if_null.h"

namespace xls::noc {

BidirectionalTreeOptionsProtoBuilder::BidirectionalTreeOptionsProtoBuilder(
    BidirectionalTreeOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

BidirectionalTreeOptionsProtoBuilder::BidirectionalTreeOptionsProtoBuilder(
    BidirectionalTreeOptionsProto* proto_ptr,
    const BidirectionalTreeOptionsProto& default_proto)
    : BidirectionalTreeOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

BidirectionalTreeOptionsProtoBuilder&
BidirectionalTreeOptionsProtoBuilder::CopyFrom(
    const BidirectionalTreeOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

BidirectionalTreeOptionsProtoBuilder&
BidirectionalTreeOptionsProtoBuilder::SetSendPortCountAtRoot(
    const int64_t send_port_count_at_root) {
  proto_ptr_->set_send_port_count_at_root(send_port_count_at_root);
  return *this;
}

BidirectionalTreeOptionsProtoBuilder&
BidirectionalTreeOptionsProtoBuilder::SetRecvPortCountAtRoot(
    const int64_t recv_port_count_at_root) {
  proto_ptr_->set_recv_port_count_at_root(recv_port_count_at_root);
  return *this;
}

}  // namespace xls::noc
