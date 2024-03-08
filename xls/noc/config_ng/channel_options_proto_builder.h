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

#ifndef XLS_NOC_CONFIG_NG_CHANNEL_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NG_CHANNEL_OPTIONS_PROTO_BUILDER_H_

#include <string_view>

#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {

// A builder to aid in constructing a channel options proto.
class ChannelOptionsProtoBuilder {
 public:
  // Constructor storing the proto pointer as a class member.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  explicit ChannelOptionsProtoBuilder(ChannelOptionsProto* proto_ptr);

  // Constructor storing the proto pointer as a class member and sets the fields
  // of the proto to default_proto.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  ChannelOptionsProtoBuilder(ChannelOptionsProto* proto_ptr,
                             const ChannelOptionsProto& default_proto);

  // Copy the field values of the proto message object from another builder
  // to the builder.
  ChannelOptionsProtoBuilder& CopyFrom(
      const ChannelOptionsProtoBuilder& builder);

  // Sets the source node name of the channel options. Makes a copy of the
  // source node name.
  ChannelOptionsProtoBuilder& SetSourceNodeName(
      std::string_view source_node_name);

  // Sets the sink node name of the channel options. Makes a copy of the sink
  // node name.
  ChannelOptionsProtoBuilder& SetSinkNodeName(std::string_view sink_node_name);

 private:
  ChannelOptionsProto* proto_ptr_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_CHANNEL_OPTIONS_PROTO_BUILDER_H_
