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

#ifndef XLS_NOC_CONFIG_VIRTUAL_CHANNEL_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_VIRTUAL_CHANNEL_CONFIG_PROTO_BUILDER_H_

#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// A builder for constructing a virtual channel configuration proto.
class VirtualChannelConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit VirtualChannelConfigProtoBuilder(VirtualChannelConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the name of the virtual channel.
  VirtualChannelConfigProtoBuilder& WithName(std::string_view name);

  // Adds the flit bit width configuration to the virtual channel.
  VirtualChannelConfigProtoBuilder& WithFlitBitWidth(int64_t flit_bit_width);

  // Adds the depth configuration to the virtual channel.
  VirtualChannelConfigProtoBuilder& WithDepth(int64_t depth);

 private:
  VirtualChannelConfigProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_VIRTUAL_CHANNEL_CONFIG_PROTO_BUILDER_H_
