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

#include "xls/noc/config/port_config_proto_builder.h"

#include <string_view>

#include "xls/common/proto_adaptor_utils.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

PortConfigProtoBuilder& PortConfigProtoBuilder::WithName(
    std::string_view name) {
  proto_->set_name(xls::ToProtoString(name));
  return *this;
}

PortConfigProtoBuilder& PortConfigProtoBuilder::AsInputDirection() {
  proto_->set_direction(PortConfigProto::INPUT);
  return *this;
}

PortConfigProtoBuilder& PortConfigProtoBuilder::AsOutputDirection() {
  proto_->set_direction(PortConfigProto::OUTPUT);
  return *this;
}

PortConfigProtoBuilder& PortConfigProtoBuilder::WithVirtualChannel(
    std::string_view virtual_channel_name) {
  proto_->add_virtual_channels(xls::ToProtoString(virtual_channel_name));
  return *this;
}

}  // namespace xls::noc
