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

#ifndef XLS_NOC_CONFIG_ARBITER_SCHEME_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_ARBITER_SCHEME_CONFIG_PROTO_BUILDER_H_

#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder_utils.h"

namespace xls::noc {

// A builder to aid in constructing an arbiter scheme configuration proto.
class ArbiterSchemeConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit ArbiterSchemeConfigProtoBuilder(
      RouterConfigProto::ArbiterSchemeConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // If the priority arbiter scheme is not enabled, it enables the priority
  // arbiter scheme within the proto, and disables any arbiter scheme enabled.
  // In addition, a priority list of input-port-virtual-channel tuple is
  // assigned to an output port. A higher priority is associated with the
  // element at the lower index within the list.
  ArbiterSchemeConfigProtoBuilder& WithPriorityEntry(
      std::string_view output_port_name,
      absl::Span<const PortVirtualChannelTuple> priority_list);

 private:
  RouterConfigProto::ArbiterSchemeConfigProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_ARBITER_SCHEME_CONFIG_PROTO_BUILDER_H_
