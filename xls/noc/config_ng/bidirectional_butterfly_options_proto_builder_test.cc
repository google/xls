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

#include "xls/noc/config_ng/bidirectional_butterfly_options_proto_builder.h"

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

// Test field values of the builder when constructed using a proto ptr.
TEST(BidirectionalButterflyOptionsProtoBuilderTest, FieldValuesForPtr) {
  BidirectionalButterflyOptionsProto proto;
  BidirectionalButterflyOptionsProtoBuilder builder(&proto);
  builder.SetEndpointConnection(
      BidirectionalButterflyOptionsProto::CONNECT_TO_LAST_STAGE);
  EXPECT_EQ(proto.endpoint_connection(),
            BidirectionalButterflyOptionsProto::CONNECT_TO_LAST_STAGE);
  builder.SetEndpointConnectionToConnectToFirstStage();
  EXPECT_EQ(proto.endpoint_connection(),
            BidirectionalButterflyOptionsProto::CONNECT_TO_FIRST_STAGE);
  builder.SetEndpointConnectionToConnectToLastStage();
  EXPECT_EQ(proto.endpoint_connection(),
            BidirectionalButterflyOptionsProto::CONNECT_TO_LAST_STAGE);
}

// Test field values of the builder when constructed using a proto ptr and a
// default proto.
TEST(BidirectionalButterflyOptionsProtoBuilderTest,
     FieldValuesForPtrWithDefault) {
  BidirectionalButterflyOptionsProto default_proto;
  BidirectionalButterflyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetEndpointConnectionToConnectToFirstStage();
  BidirectionalButterflyOptionsProto proto;
  BidirectionalButterflyOptionsProtoBuilder builder(&proto, default_proto);
  EXPECT_EQ(proto.endpoint_connection(),
            BidirectionalButterflyOptionsProto::CONNECT_TO_FIRST_STAGE);
}

// Test field values of the builder when copied from another builder.
TEST(BidirectionalButterflyOptionsProtoBuilderTest, FieldValuesForCopyFrom) {
  BidirectionalButterflyOptionsProto default_proto;
  BidirectionalButterflyOptionsProtoBuilder builder_default(&default_proto);
  builder_default.SetEndpointConnectionToConnectToFirstStage();
  BidirectionalButterflyOptionsProto proto;
  BidirectionalButterflyOptionsProtoBuilder builder(&proto);
  builder.CopyFrom(builder_default);
  EXPECT_EQ(proto.endpoint_connection(),
            BidirectionalButterflyOptionsProto::CONNECT_TO_FIRST_STAGE);
}

}  // namespace
}  // namespace xls::noc
