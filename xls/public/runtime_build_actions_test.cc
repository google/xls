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

#include "xls/public/runtime_build_actions.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"

namespace xls {
namespace {

TEST(RuntimeBuildActionsTest, SimpleProtoToDslxConversion) {
  constexpr std::string_view kBindingName = "MY_TEST_MESSAGE";
  constexpr std::string_view kProtoDef = R"(
syntax = "proto2";

package xls_public_test;

message TestMessage {
  optional int32 test_field = 1;
}

message TestRepeatedMessage {
  repeated TestMessage messages = 1;
}
)";
  constexpr std::string_view kTextProto = R"(
messages: {
  test_field: 42
}
messages: {
  test_field: 64
}
)";
  constexpr std::string_view kMessageName =
      "xls_public_test.TestRepeatedMessage";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string dslx,
      ProtoToDslx(kProtoDef, kMessageName, kTextProto, kBindingName));
  EXPECT_EQ(dslx, R"(pub struct TestMessage {
    test_field: sN[32],
}
pub struct TestRepeatedMessage {
    messages: TestMessage[2],
    messages_count: u32,
}
pub const MY_TEST_MESSAGE = TestRepeatedMessage { messages: [TestMessage { test_field: sN[32]:42 }, TestMessage { test_field: sN[32]:64 }], messages_count: u32:2 };)");
}

TEST(RuntimeBuildActionsTest, DefaultDslxPath) {
  EXPECT_EQ(GetDefaultDslxStdlibPath(), kDefaultDslxStdlibPath);
}

}  // namespace
}  // namespace xls
