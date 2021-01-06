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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/common_network_config_builder_options_proto_builder.h"

namespace xls::noc {

// Test field values for a link option.
TEST(LinkOptionsProtoBuilderTest, FieldValues) {
  const int64 kSourceSinkPipelineStage = 4;
  const int64 kSinkSourcePipelineStage = 2;
  LinkOptionsProto proto;
  LinkOptionsProtoBuilder link_options_proto_builder(&proto);
  link_options_proto_builder.WithSourceSinkPipelineStage(
      kSourceSinkPipelineStage);
  link_options_proto_builder.WithSinkSourcePipelineStage(
      kSinkSourcePipelineStage);
  link_options_proto_builder.GetFlowControlOptionsProtoBuilder();

  EXPECT_TRUE(proto.has_source_sink_pipeline_stage());
  EXPECT_TRUE(proto.has_sink_source_pipeline_stage());
  EXPECT_TRUE(proto.has_flow_control());
}

}  // namespace xls::noc
