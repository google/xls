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

#include "xls/noc/config_ng/butterfly_options_proto_builder.h"

#include <cstdint>

#include "absl/log/die_if_null.h"

namespace xls::noc {

ButterflyOptionsProtoBuilder::ButterflyOptionsProtoBuilder(
    ButterflyOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

ButterflyOptionsProtoBuilder::ButterflyOptionsProtoBuilder(
    ButterflyOptionsProto* proto_ptr,
    const ButterflyOptionsProto& default_proto)
    : ButterflyOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

ButterflyOptionsProtoBuilder& ButterflyOptionsProtoBuilder::CopyFrom(
    const ButterflyOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

ButterflyOptionsProtoBuilder& ButterflyOptionsProtoBuilder::SetRadixPerStage(
    const int64_t radix_per_stage) {
  proto_ptr_->set_radix_per_stage(radix_per_stage);
  return *this;
}

ButterflyOptionsProtoBuilder& ButterflyOptionsProtoBuilder::SetStageCount(
    const int64_t stage_count) {
  proto_ptr_->set_stage_count(stage_count);
  return *this;
}

ButterflyOptionsProtoBuilder& ButterflyOptionsProtoBuilder::SetFlatten(
    const bool flatten) {
  proto_ptr_->set_flatten(flatten);
  return *this;
}

ButterflyOptionsProtoBuilder&
ButterflyOptionsProtoBuilder::EnableUnidirectional(
    const UnidirectionalButterflyOptionsProtoBuilder& builder) {
  UnidirectionalButterflyOptionsProtoBuilder(
      proto_ptr_->mutable_unidirectional_butterfly())
      .CopyFrom(builder);
  return *this;
}

UnidirectionalButterflyOptionsProtoBuilder
ButterflyOptionsProtoBuilder::EnableUnidirectional() {
  return UnidirectionalButterflyOptionsProtoBuilder(
      proto_ptr_->mutable_unidirectional_butterfly());
}

ButterflyOptionsProtoBuilder& ButterflyOptionsProtoBuilder::EnableBidirectional(
    const BidirectionalButterflyOptionsProtoBuilder& builder) {
  BidirectionalButterflyOptionsProtoBuilder(
      proto_ptr_->mutable_bidirectional_butterfly())
      .CopyFrom(builder);
  return *this;
}

BidirectionalButterflyOptionsProtoBuilder
ButterflyOptionsProtoBuilder::EnableBidirectional() {
  return BidirectionalButterflyOptionsProtoBuilder(
      proto_ptr_->mutable_bidirectional_butterfly());
}

ButterflyOptionsProtoBuilder&
ButterflyOptionsProtoBuilder::UnidirectionalButterflyTopology(
    const int64_t stage_count, const int64_t radix_per_stage,
    const UnidirectionalButterflyOptionsProto::Flow flow, const bool flatten) {
  this->SetStageCount(stage_count)
      .SetRadixPerStage(radix_per_stage)
      .SetFlatten(flatten);
  this->EnableUnidirectional().SetFlow(flow);
  return *this;
}

ButterflyOptionsProtoBuilder&
ButterflyOptionsProtoBuilder::BidirectionalButterflyTopology(
    const int64_t stage_count, const int64_t radix_per_stage,
    const BidirectionalButterflyOptionsProto::EndpointConnection
        endpoint_connection,
    const bool flatten) {
  this->SetStageCount(stage_count)
      .SetRadixPerStage(radix_per_stage)
      .SetFlatten(flatten);
  this->EnableBidirectional().SetEndpointConnection(endpoint_connection);
  return *this;
}

}  // namespace xls::noc
