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

#ifndef XLS_NOC_CONFIG_NG_DIMENSION_ORDER_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NG_DIMENSION_ORDER_OPTIONS_PROTO_BUILDER_H_

#include <cstdint>

#include "xls/noc/config_ng/dimension_order_entry_options_proto_builder.h"
#include "xls/noc/config_ng/topology_endpoint_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"

namespace xls::noc {

// A builder to aid in constructing a dimension order options proto.
class DimensionOrderOptionsProtoBuilder {
 public:
  // Constructor storing the proto pointer as a class member.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  explicit DimensionOrderOptionsProtoBuilder(
      DimensionOrderOptionsProto* proto_ptr);

  // Constructor storing the proto pointer as a class member and sets the fields
  // of the proto to default_proto.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  DimensionOrderOptionsProtoBuilder(
      DimensionOrderOptionsProto* proto_ptr,
      const DimensionOrderOptionsProto& default_proto);

  // Copy the field values of the proto message object from another builder
  // to the builder.
  DimensionOrderOptionsProtoBuilder& CopyFrom(
      const DimensionOrderOptionsProtoBuilder& builder);

  // Add a dimension to the dimension order options.
  DimensionOrderEntryOptionsProtoBuilder AddDimension();

  // Add a dimension to the dimension order options.
  DimensionOrderOptionsProtoBuilder& AddDimension(
      const DimensionOrderEntryOptionsProtoBuilder& builder);

  // Sets the endpoint options of the dimension order options.
  DimensionOrderOptionsProtoBuilder& SetEndpointOptions(
      const TopologyEndpointOptionsProtoBuilder& builder);

  // Gets the endpoint options of the dimension order options.
  TopologyEndpointOptionsProtoBuilder GetEndpointOptions() const;

  // A line topology with a user defined number of routers, and the number of
  // endpoints per router.
  DimensionOrderOptionsProtoBuilder& LineTopology(
      int64_t router_count,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A ring topology with a user defined number of routers, and the number of
  // endpoints per router.
  DimensionOrderOptionsProtoBuilder& RingTopology(
      int64_t router_count,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A symmetric torus topology with a user defined number of routers, and the
  // number of endpoints per router.
  DimensionOrderOptionsProtoBuilder& SymmetricTorusTopology(
      int64_t router_count,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A symmetric mesh topology with a user defined number of routers, and the
  // number of endpoints per router.
  DimensionOrderOptionsProtoBuilder& SymmetricMeshTopology(
      int64_t router_count,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A symmetric torus topology with a user defined number of routers for
  // dimension 0 and dimension 1, and the number of endpoints per router.
  DimensionOrderOptionsProtoBuilder& TorusTopology(
      int64_t router_count_0, int64_t router_count_1,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A mesh topology with a user defined number of routers for dimension 0 and
  // dimension 1, and the number of endpoints per router.
  DimensionOrderOptionsProtoBuilder& MeshTopology(
      int64_t router_count_0, int64_t router_count_1,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

  // A grid topology with a user defined number of routers for dimension 0 and
  // dimension 1, and the number of endpoints per router. The routers along
  // dimension 0 has a connection between the routers at the perimeter.
  DimensionOrderOptionsProtoBuilder& GridTopologyWithDimension0Loopback(
      int64_t router_count_0, int64_t router_count_1,
      const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options);

 private:
  DimensionOrderOptionsProto* proto_ptr_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_DIMENSION_ORDER_OPTIONS_PROTO_BUILDER_H_
