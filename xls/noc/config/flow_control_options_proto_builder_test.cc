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

// Test field values for a peek flow control option.
TEST(FlowControlOptionsProtoBuilderTest, PeekFlowControlFieldValues) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnablePeekFlowControl();

  EXPECT_TRUE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
}

// Test field values for a token credit flow control option.
TEST(FlowControlOptionsProtoBuilderTest, TokenCreditFlowControlFieldValues) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnableTokenCreditBasedFlowControl();

  EXPECT_FALSE(proto.has_peek());
  EXPECT_TRUE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
}

// Test field values for a total credit flow control option.
TEST(FlowControlOptionsProtoBuilderTest, TotalCreditFlowControlFieldValues) {
  const int64_t kCreditBitWidth = 42;
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnableTotalCreditBasedFlowControl(kCreditBitWidth);

  EXPECT_FALSE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_TRUE(proto.has_total_credit_based());
  EXPECT_EQ(proto.total_credit_based().credit_bit_width(), kCreditBitWidth);
}

// Test last enabled flow control option.
TEST(FlowControlOptionsProtoBuilderTest, LastEnabledFlowControl) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  EXPECT_FALSE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
  builder.EnablePeekFlowControl();
  EXPECT_TRUE(proto.has_peek());
  builder.EnableTokenCreditBasedFlowControl();
  EXPECT_TRUE(proto.has_token_credit_based());
  builder.EnableTotalCreditBasedFlowControl(1337);
  EXPECT_TRUE(proto.has_total_credit_based());
}

}  // namespace xls::noc
