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

#include "xls/noc/config_ng/dimension_order_entry_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {

DimensionOrderEntryOptionsProtoBuilder::DimensionOrderEntryOptionsProtoBuilder(
    DimensionOrderEntryOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

DimensionOrderEntryOptionsProtoBuilder::DimensionOrderEntryOptionsProtoBuilder(
    DimensionOrderEntryOptionsProto* proto_ptr,
    const DimensionOrderEntryOptionsProto& default_proto)
    : DimensionOrderEntryOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::CopyFrom(
    const DimensionOrderEntryOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::SetDimensionIndex(const int64_t index) {
  proto_ptr_->set_index(index);
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::SetEntityCount(
    const int64_t entity_count) {
  proto_ptr_->set_entity_count(entity_count);
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::SetLoopback(const bool loopback) {
  proto_ptr_->set_loopback(loopback);
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::SetLoopbackToTrue() {
  proto_ptr_->set_loopback(true);
  return *this;
}

DimensionOrderEntryOptionsProtoBuilder&
DimensionOrderEntryOptionsProtoBuilder::SetLoopbackToFalse() {
  proto_ptr_->set_loopback(false);
  return *this;
}

}  // namespace xls::noc
