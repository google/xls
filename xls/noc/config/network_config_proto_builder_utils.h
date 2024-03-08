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

#ifndef XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_UTILS_H_
#define XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_UTILS_H_

#include <string>

namespace xls::noc {

// A tuple associating a port and a virtual channel.
struct PortVirtualChannelTuple {
  std::string port_name;
  std::string virtual_channel_name;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NETWORK_CONFIG_PROTO_BUILDER_UTILS_H_
