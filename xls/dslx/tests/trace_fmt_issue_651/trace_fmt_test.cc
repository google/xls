// Copyright 2024 The XLS Authors
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
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/tests/trace_fmt_issue_651/test_target_enum_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/test_target_s32_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/test_target_u16_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/test_target_u21_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;

MATCHER_P(TraceMessage, m, "") {
  return testing::ExplainMatchResult(m, arg.message, result_listener);
}
TEST(TraceFmt, LeadingOneU16) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0xff00, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(res.events.trace_msgs,
              ElementsAre(TraceMessage("1111_1111_0000_0000")));
}

TEST(TraceFmt, ZeroPaddedU16) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x70, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(res.events.trace_msgs,
              ElementsAre(TraceMessage("0000_0000_0111_0000")));
}

TEST(TraceFmt, LeadingOneU21) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto input,
      ValueBuilder::Bits(UBits(0b100010001000100010001, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(res.events.trace_msgs,
              ElementsAre(TraceMessage("1_0001_0001_0001_0001_0001")));
}

TEST(TraceFmt, ZeroPaddedU21) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x70, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(res.events.trace_msgs,
              ElementsAre(TraceMessage("0_0000_0000_0000_0111_0000")));
}

TEST(TraceFmt, LeadingOneS32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_s32::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(SBits(-32, 32)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(
      res.events.trace_msgs,
      ElementsAre(TraceMessage("1111_1111_1111_1111_1111_1111_1110_0000")));
}

TEST(TraceFmt, ZeroPaddedS32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_s32::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(SBits(0x70, 32)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(
      res.events.trace_msgs,
      ElementsAre(TraceMessage("0000_0000_0000_0000_0000_0000_0111_0000")));
}

TEST(TraceFmt, Enum) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::TraceEnum::Create());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run(absl::Span<Value>()));

  EXPECT_THAT(res.events.assert_msgs, IsEmpty());
  EXPECT_THAT(res.events.trace_msgs,
              ElementsAre(TraceMessage("0_0000_0011_0000_0011_1001")));
}
}  // namespace
}  // namespace xls
