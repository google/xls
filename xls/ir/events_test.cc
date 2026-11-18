// Copyright 2026 The XLS Authors
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

#include "xls/ir/events.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(InterpreterEventsTest, EmptyStateAndPredicates) {
  InterpreterEvents events;
  EXPECT_TRUE(events.IsEmpty());
  EXPECT_FALSE(events.HasTraces());
  EXPECT_FALSE(events.HasAsserts());
  XLS_EXPECT_OK(events.ToStatus());

  events.Clear();
  EXPECT_TRUE(events.IsEmpty());
}

TEST(InterpreterEventsTest, TraceEventsRemainOkAndClearResetsState) {
  InterpreterEvents events;
  events.AddTraceStatementMessage(/*verbosity=*/0, "trace 1");
  EXPECT_FALSE(events.IsEmpty());
  EXPECT_TRUE(events.HasTraces());
  EXPECT_FALSE(events.HasAsserts());
  XLS_EXPECT_OK(events.ToStatus());

  events.Clear();
  EXPECT_TRUE(events.IsEmpty());
  EXPECT_FALSE(events.HasTraces());
  EXPECT_TRUE(events.GetTraceMessageStrings().empty());
}

TEST(InterpreterEventsTest, AssertEventsReturnFirstAssertionFailure) {
  InterpreterEvents events;
  events.AddAssertMessage("first failure");
  events.AddAssertMessage("second failure");
  EXPECT_FALSE(events.IsEmpty());
  EXPECT_FALSE(events.HasTraces());
  EXPECT_TRUE(events.HasAsserts());
  EXPECT_THAT(events.ToStatus(),
              StatusIs(absl::StatusCode::kAborted, "first failure"));

  InterpreterResult<int> failed_result{.value = 42, .events = events};
  EXPECT_THAT(InterpreterResultToStatusOrValue(failed_result),
              StatusIs(absl::StatusCode::kAborted, "first failure"));

  events.Clear();
  EXPECT_TRUE(events.IsEmpty());
  EXPECT_FALSE(events.HasAsserts());
  XLS_EXPECT_OK(events.ToStatus());

  InterpreterResult<int> ok_result{.value = 42, .events = events};
  EXPECT_THAT(InterpreterResultToStatusOrValue(ok_result), IsOkAndHolds(42));
}

}  // namespace
}  // namespace xls
