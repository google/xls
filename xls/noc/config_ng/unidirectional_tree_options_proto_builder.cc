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

#include "xls/noc/config_ng/unidirectional_tree_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {

UnidirectionalTreeOptionsProtoBuilder::UnidirectionalTreeOptionsProtoBuilder(
    UnidirectionalTreeOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

UnidirectionalTreeOptionsProtoBuilder::UnidirectionalTreeOptionsProtoBuilder(
    UnidirectionalTreeOptionsProto* proto_ptr,
    const UnidirectionalTreeOptionsProto& default_proto)
    : UnidirectionalTreeOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

UnidirectionalTreeOptionsProtoBuilder&
UnidirectionalTreeOptionsProtoBuilder::CopyFrom(
    const UnidirectionalTreeOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

UnidirectionalTreeOptionsProtoBuilder&
UnidirectionalTreeOptionsProtoBuilder::SetType(
    const UnidirectionalTreeOptionsProto::Type type) {
  proto_ptr_->set_type(type);
  return *this;
}

UnidirectionalTreeOptionsProtoBuilder&
UnidirectionalTreeOptionsProtoBuilder::AsAggregation() {
  proto_ptr_->set_type(UnidirectionalTreeOptionsProto::AGGREGATION);
  return *this;
}

UnidirectionalTreeOptionsProtoBuilder&
UnidirectionalTreeOptionsProtoBuilder::AsDistribution() {
  proto_ptr_->set_type(UnidirectionalTreeOptionsProto::DISTRIBUTION);
  return *this;
}

UnidirectionalTreeOptionsProtoBuilder&
UnidirectionalTreeOptionsProtoBuilder::SetPortCountAtRoot(
    const int64_t port_count_at_root) {
  proto_ptr_->set_port_count_at_root(port_count_at_root);
  return *this;
}

}  // namespace xls::noc
