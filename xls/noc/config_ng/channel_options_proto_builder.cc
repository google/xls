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

#include "xls/noc/config_ng/channel_options_proto_builder.h"

#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/common/logging/logging.h"
#include "xls/common/proto_adaptor_utils.h"

namespace xls::noc {

ChannelOptionsProtoBuilder::ChannelOptionsProtoBuilder(
    ChannelOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

ChannelOptionsProtoBuilder::ChannelOptionsProtoBuilder(
    ChannelOptionsProto* proto_ptr, const ChannelOptionsProto& default_proto)
    : ChannelOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

ChannelOptionsProtoBuilder& ChannelOptionsProtoBuilder::CopyFrom(
    const ChannelOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

ChannelOptionsProtoBuilder& ChannelOptionsProtoBuilder::SetSourceNodeName(
    std::string_view source_node_name) {
  proto_ptr_->set_source_node_name(xls::ToProtoString(source_node_name));
  return *this;
}

ChannelOptionsProtoBuilder& ChannelOptionsProtoBuilder::SetSinkNodeName(
    std::string_view sink_node_name) {
  proto_ptr_->set_sink_node_name(xls::ToProtoString(sink_node_name));
  return *this;
}

}  // namespace xls::noc
