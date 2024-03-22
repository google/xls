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

#include "xls/noc/config_ng/dimension_order_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {
DimensionOrderOptionsProtoBuilder::DimensionOrderOptionsProtoBuilder(
    DimensionOrderOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

DimensionOrderOptionsProtoBuilder::DimensionOrderOptionsProtoBuilder(
    DimensionOrderOptionsProto* proto_ptr,
    const DimensionOrderOptionsProto& default_proto)
    : DimensionOrderOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

DimensionOrderOptionsProtoBuilder& DimensionOrderOptionsProtoBuilder::CopyFrom(
    const DimensionOrderOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder
DimensionOrderOptionsProtoBuilder::AddDimension() {
  return DimensionOrderEntryOptionsProtoBuilder(proto_ptr_->add_dimensions());
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::AddDimension(
    const DimensionOrderEntryOptionsProtoBuilder& builder) {
  DimensionOrderEntryOptionsProtoBuilder(proto_ptr_->add_dimensions())
      .CopyFrom(builder);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::SetEndpointOptions(
    const TopologyEndpointOptionsProtoBuilder& builder) {
  TopologyEndpointOptionsProtoBuilder(proto_ptr_->mutable_endpoint_options())
      .CopyFrom(builder);
  return *this;
}

TopologyEndpointOptionsProtoBuilder
DimensionOrderOptionsProtoBuilder::GetEndpointOptions() const {
  return TopologyEndpointOptionsProtoBuilder(
      proto_ptr_->mutable_endpoint_options());
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::LineTopology(
    const int64_t router_count,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count)
      .SetLoopbackToFalse();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(1)
      .SetLoopbackToFalse();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::RingTopology(
    const int64_t router_count,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count)
      .SetLoopbackToTrue();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(1)
      .SetLoopbackToFalse();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::SymmetricTorusTopology(
    const int64_t router_count,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count)
      .SetLoopbackToTrue();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(router_count)
      .SetLoopbackToTrue();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::SymmetricMeshTopology(
    const int64_t router_count,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count)
      .SetLoopbackToFalse();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(router_count)
      .SetLoopbackToFalse();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::TorusTopology(
    const int64_t router_count_0, const int64_t router_count_1,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count_0)
      .SetLoopbackToTrue();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(router_count_1)
      .SetLoopbackToTrue();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::MeshTopology(
    const int64_t router_count_0, const int64_t router_count_1,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count_0)
      .SetLoopbackToFalse();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(router_count_1)
      .SetLoopbackToFalse();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

DimensionOrderOptionsProtoBuilder&
DimensionOrderOptionsProtoBuilder::GridTopologyWithDimension0Loopback(
    const int64_t router_count_0, const int64_t router_count_1,
    const TopologyEndpointOptionsProtoBuilder& topology_endpoints_options) {
  this->AddDimension()
      .SetDimensionIndex(0)
      .SetEntityCount(router_count_0)
      .SetLoopbackToTrue();
  this->AddDimension()
      .SetDimensionIndex(1)
      .SetEntityCount(router_count_1)
      .SetLoopbackToFalse();
  this->SetEndpointOptions(topology_endpoints_options);
  return *this;
}

}  // namespace xls::noc
