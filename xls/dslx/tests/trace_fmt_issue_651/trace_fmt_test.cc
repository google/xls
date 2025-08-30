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

#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_enum_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_s32_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_u16_hex_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_u16_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_u21_hex_wrapper.h"
#include "xls/dslx/tests/trace_fmt_issue_651/trace_u21_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
TEST(TraceFmt, LeadingOneU16) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0xff00, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("1111_1111_0000_0000"));
}

TEST(TraceFmt, ZeroPaddedU16) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x70, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("0000_0000_0111_0000"));
}

TEST(TraceFmt, LeadingOneU21) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto input,
      ValueBuilder::Bits(UBits(0b100010001000100010001, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("1_0001_0001_0001_0001_0001"));
}

TEST(TraceFmt, ZeroPaddedU21) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x70, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("0_0000_0000_0000_0111_0000"));
}

TEST(TraceFmt, LeadingOneS32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_s32::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(SBits(-32, 32)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("1111_1111_1111_1111_1111_1111_1110_0000"));
}

TEST(TraceFmt, ZeroPaddedS32) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_s32::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(SBits(0x70, 32)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("0000_0000_0000_0000_0000_0000_0111_0000"));
}

TEST(TraceFmt, Enum) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_enum::Create());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run(absl::Span<Value>()));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("0_0000_0011_0000_0011_1001"));
}

TEST(TraceFmt, LeadingOne16Hex) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16_hex::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0xff00, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(), ElementsAre("ff00"));
}

TEST(TraceFmt, ZeroPadded16Hex) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u16_hex::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x70, 16)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(), ElementsAre("0070"));
}

TEST(TraceFmt, LeadingOne21Hex) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21_hex::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0x1fff00, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(), ElementsAre("1f_ff00"));
}

TEST(TraceFmt, ZeroPadded21Hex) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::Trace_u21_hex::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto input,
                           ValueBuilder::Bits(UBits(0xfff00, 21)).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(), ElementsAre("0f_ff00"));
}
}  // namespace
}  // namespace xls
