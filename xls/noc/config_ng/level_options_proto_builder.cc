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

#include "xls/noc/config_ng/level_options_proto_builder.h"

#include "absl/log/die_if_null.h"

namespace xls::noc {

LevelOptionsProtoBuilder::LevelOptionsProtoBuilder(LevelOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

LevelOptionsProtoBuilder::LevelOptionsProtoBuilder(
    LevelOptionsProto* proto_ptr, const LevelOptionsProto& default_proto)
    : LevelOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

LevelOptionsProtoBuilder& LevelOptionsProtoBuilder::CopyFrom(
    const LevelOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

LevelOptionsProtoBuilder& LevelOptionsProtoBuilder::SetIndex(
    const int64_t index) {
  proto_ptr_->set_index(index);
  return *this;
}

LevelOptionsProtoBuilder& LevelOptionsProtoBuilder::SetNodeCount(
    const int64_t node_count) {
  proto_ptr_->set_node_count(node_count);
  return *this;
}

}  // namespace xls::noc
