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

#include "xls/noc/config/common_network_config_builder_options_proto_builder.h"

namespace xls::noc {

LinkOptionsProtoBuilder& LinkOptionsProtoBuilder::WithSourceSinkPipelineStage(
    int64_t pipeline_stage) {
  proto_->set_source_sink_pipeline_stage(pipeline_stage);
  return *this;
}

LinkOptionsProtoBuilder& LinkOptionsProtoBuilder::WithSinkSourcePipelineStage(
    int64_t pipeline_stage) {
  proto_->set_sink_source_pipeline_stage(pipeline_stage);
  return *this;
}

FlowControlOptionsProtoBuilder
LinkOptionsProtoBuilder::GetFlowControlOptionsProtoBuilder() {
  return FlowControlOptionsProtoBuilder(proto_->mutable_flow_control());
}

}  // namespace xls::noc
