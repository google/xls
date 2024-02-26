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

#include "xls/noc/config_ng/node_options_proto_builder.h"

#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/common/logging/logging.h"
#include "xls/common/proto_adaptor_utils.h"

namespace xls::noc {

NodeOptionsProtoBuilder::NodeOptionsProtoBuilder(NodeOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

NodeOptionsProtoBuilder::NodeOptionsProtoBuilder(
    NodeOptionsProto* proto_ptr,
    const NodeOptionsProto& default_proto)
    : NodeOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

NodeOptionsProtoBuilder& NodeOptionsProtoBuilder::CopyFrom(
    const NodeOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

NodeOptionsProtoBuilder& NodeOptionsProtoBuilder::SetName(
    std::string_view name) {
  proto_ptr_->set_name(xls::ToProtoString(name));
  return *this;
}

NodeOptionsProtoBuilder& NodeOptionsProtoBuilder::SetSendPortCount(
    const int64_t send_port_count) {
  proto_ptr_->set_send_port_count(send_port_count);
  return *this;
}

NodeOptionsProtoBuilder& NodeOptionsProtoBuilder::SetRecvPortCount(
    const int64_t recv_port_count) {
  proto_ptr_->set_recv_port_count(recv_port_count);
  return *this;
}

NodeOptionsProtoBuilder& NodeOptionsProtoBuilder::AddCoordinate(
    const CoordinateOptionsProtoBuilder& builder) {
  CoordinateOptionsProtoBuilder(proto_ptr_->add_coordinate()).CopyFrom(builder);
  return *this;
}

CoordinateOptionsProtoBuilder NodeOptionsProtoBuilder::AddCoordinate() {
  return CoordinateOptionsProtoBuilder(proto_ptr_->add_coordinate());
}

}  // namespace xls::noc
