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

#ifndef XLS_NOC_CONFIG_NG_FULLY_CONNECTED_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NG_FULLY_CONNECTED_OPTIONS_PROTO_BUILDER_H_

#include <cstdint>

#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {

// A builder to aid in constructing a fully connected options proto.
class FullyConnectedOptionsProtoBuilder {
 public:
  // Constructor storing the proto pointer as a class member.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  explicit FullyConnectedOptionsProtoBuilder(
      FullyConnectedOptionsProto* proto_ptr);

  // Constructor storing the proto pointer as a class member and sets the fields
  // of the proto to default_proto.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  FullyConnectedOptionsProtoBuilder(
      FullyConnectedOptionsProto* proto_ptr,
      const FullyConnectedOptionsProto& default_proto);

  // Copy the field values of the proto message object from another builder
  // to the builder.
  FullyConnectedOptionsProtoBuilder& CopyFrom(
      const FullyConnectedOptionsProtoBuilder& builder);

  // Sets the router count of the fully connected options.
  FullyConnectedOptionsProtoBuilder& SetRouterCount(int64_t router_count);

  // Sets the endpoint options of the fully connected options.
  FullyConnectedOptionsProtoBuilder& SetEndpointOptions(
      const TopologyEndpointOptionsProtoBuilder& builder);

  // Gets the endpoint options of the fully connected options.
  TopologyEndpointOptionsProtoBuilder GetEndpointOptions();

 private:
  FullyConnectedOptionsProto* proto_ptr_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_FULLY_CONNECTED_OPTIONS_PROTO_BUILDER_H_
