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

#ifndef XLS_NOC_CONFIG_PORT_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_PORT_CONFIG_PROTO_BUILDER_H_

#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// A builder for constructing a port configuration proto.
class PortConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit PortConfigProtoBuilder(PortConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the name of the port.
  PortConfigProtoBuilder& WithName(std::string_view name);

  // Sets the direction type of the port as input.
  PortConfigProtoBuilder& AsInputDirection();

  // Sets the direction type of the port as output.
  PortConfigProtoBuilder& AsOutputDirection();

  // Adds the virtual channel configuration to the port.
  PortConfigProtoBuilder& WithVirtualChannel(
      std::string_view virtual_channel_name);

 private:
  PortConfigProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_PORT_CONFIG_PROTO_BUILDER_H_
