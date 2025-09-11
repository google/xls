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
#include "xls/common/status/matchers.h"
#include "xls/dslx/tests/trace_fmt_array/test_target_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace xls {
namespace {
using ::testing::ElementsAre;
using ::testing::IsEmpty;
TEST(TraceFmtArray, TraceHasCommas) {
  XLS_ASSERT_OK_AND_ASSIGN(auto trace, wrapped::TraceIt::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto input,
      ValueBuilder::UBitsArray({1, 2, 3, 4, 5, 6, 7, 8}, 8).Build());
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> res,
                           trace->jit()->Run({input}));

  EXPECT_EQ(res.value.bits(), UBits(1, 8));
  EXPECT_THAT(res.events.GetAssertMessages(), IsEmpty());
  EXPECT_THAT(res.events.GetTraceMessageStrings(),
              ElementsAre("The array is [1, 2, 3, 4, 5, 6, 7, 8]"));
}

}  // namespace
}  // namespace xls
