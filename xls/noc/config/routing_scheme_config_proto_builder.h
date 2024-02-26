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

#ifndef XLS_NOC_CONFIG_ROUTING_SCHEME_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_ROUTING_SCHEME_CONFIG_PROTO_BUILDER_H_

#include "absl/log/die_if_null.h"
#include "xls/common/logging/logging.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder_utils.h"

namespace xls::noc {

// A builder for constructing a routing scheme configuration proto.
class RoutingSchemeConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit RoutingSchemeConfigProtoBuilder(
      RouterConfigProto::RoutingSchemeConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // If the distributed routing scheme is not enabled, it enables the
  // distributed routing scheme, and disables any routing scheme enabled. In
  // addition, it adds an entry to the routing table. For more information,
  // please see xls.noc.RouterConfigProto.DistributedRoutingConfig.
  RoutingSchemeConfigProtoBuilder& WithDistributedRoutingEntry(
      PortVirtualChannelTuple network_receive_port,
      PortVirtualChannelTuple router_output_port);

 private:
  RouterConfigProto::RoutingSchemeConfigProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_ROUTING_SCHEME_CONFIG_PROTO_BUILDER_H_
