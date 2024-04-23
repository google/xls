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

#include "xls/noc/config/link_config_proto_builder.h"

#include <cstdint>
#include <string_view>

#include "xls/common/proto_adaptor_utils.h"

namespace xls::noc {

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithName(
    std::string_view name) {
  proto_->set_name(xls::ToProtoString(name));
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithSourcePort(
    std::string_view port_name) {
  proto_->set_source_port_name(xls::ToProtoString(port_name));
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithSinkPort(
    std::string_view port_name) {
  proto_->set_sink_port_name(xls::ToProtoString(port_name));
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithPhitBitWidth(
    int64_t phit_bit_width) {
  proto_->set_phit_bit_width(phit_bit_width);
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithSourceSinkPipelineStage(
    int64_t pipeline_stage) {
  proto_->set_source_sink_pipeline_stage(pipeline_stage);
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithSinkSourcePipelineStage(
    int64_t pipeline_stage) {
  proto_->set_sink_source_pipeline_stage(pipeline_stage);
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithPeekFlowControl() {
  proto_->mutable_flow_control()->mutable_peek();
  return *this;
}

LinkConfigProtoBuilder&
LinkConfigProtoBuilder::WithTokenCreditBasedFlowControl() {
  proto_->mutable_flow_control()->mutable_token_credit_based();
  return *this;
}

LinkConfigProtoBuilder& LinkConfigProtoBuilder::WithTotalCreditBasedFlowControl(
    int64_t credit_bit_width) {
  proto_->mutable_flow_control()
      ->mutable_total_credit_based()
      ->set_credit_bit_width(credit_bit_width);
  return *this;
}

}  // namespace xls::noc
