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

#include "xls/noc/config_ng/unidirectional_butterfly_options_proto_builder.h"

#include "absl/log/die_if_null.h"
#include "xls/common/logging/logging.h"

namespace xls::noc {

UnidirectionalButterflyOptionsProtoBuilder::
    UnidirectionalButterflyOptionsProtoBuilder(
        UnidirectionalButterflyOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

UnidirectionalButterflyOptionsProtoBuilder::
    UnidirectionalButterflyOptionsProtoBuilder(
        UnidirectionalButterflyOptionsProto* proto_ptr,
        const UnidirectionalButterflyOptionsProto& default_proto)
    : UnidirectionalButterflyOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

UnidirectionalButterflyOptionsProtoBuilder&
UnidirectionalButterflyOptionsProtoBuilder::CopyFrom(
    const UnidirectionalButterflyOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

UnidirectionalButterflyOptionsProtoBuilder&
UnidirectionalButterflyOptionsProtoBuilder::SetFlow(
    const UnidirectionalButterflyOptionsProto::Flow flow) {
  proto_ptr_->set_flow(flow);
  return *this;
}

UnidirectionalButterflyOptionsProtoBuilder&
UnidirectionalButterflyOptionsProtoBuilder::SetFlowFromFirstStageToLastStage() {
  proto_ptr_->set_flow(
      UnidirectionalButterflyOptionsProto::FROM_FIRST_STAGE_TO_LAST_STAGE);
  return *this;
}

UnidirectionalButterflyOptionsProtoBuilder&
UnidirectionalButterflyOptionsProtoBuilder::SetFlowFromLastStageToFirstStage() {
  proto_ptr_->set_flow(
      UnidirectionalButterflyOptionsProto::FROM_LAST_STAGE_TO_FIRST_STAGE);
  return *this;
}

}  // namespace xls::noc
