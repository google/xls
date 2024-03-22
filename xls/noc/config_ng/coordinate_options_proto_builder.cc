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

#include "xls/noc/config_ng/coordinate_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {

CoordinateOptionsProtoBuilder::CoordinateOptionsProtoBuilder(
    CoordinateOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

CoordinateOptionsProtoBuilder::CoordinateOptionsProtoBuilder(
    CoordinateOptionsProto* proto_ptr,
    const CoordinateOptionsProto& default_proto)
    : CoordinateOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

CoordinateOptionsProtoBuilder& CoordinateOptionsProtoBuilder::CopyFrom(
    const CoordinateOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

CoordinateOptionsProtoBuilder& CoordinateOptionsProtoBuilder::SetIndex(
    const int64_t index) {
  proto_ptr_->set_index(index);
  return *this;
}

CoordinateOptionsProtoBuilder& CoordinateOptionsProtoBuilder::SetLocation(
    const int64_t location) {
  proto_ptr_->set_location(location);
  return *this;
}

}  // namespace xls::noc
