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

#include "xls/common/logging/scoped_vlog_level.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "xls/common/logging/collect_logs.h"
#include "xls/common/logging/scoped_vlog_level_test_helper.h"

namespace xls {
namespace {

using testing::AllOf;
using testing::Contains;
using testing::HasSubstr;
using testing::Not;
auto ContainsSubstr(std::string_view sv) { return Contains(HasSubstr(sv)); }
void EmitVlogMessage(std::string message) { VLOG(1) << message; }
void EmitVlog4Message(std::string message) { VLOG(4) << message; }

TEST(ScopedSetVlogLevelTest, SingleEntry) {
  LogCollectorSink sink;
  absl::AddLogSink(&sink);
  ScopedSetVlogLevel set_vlog_level("scoped_vlog_level_test", 1);
  EmitVlogMessage("main level 1");
  EmitVlog4Message("main level 4");
  EmitHelperVlogMessage("helper level 1");
  EmitHelperVlog4Message("helper level 4");
  absl::RemoveLogSink(&sink);
  EXPECT_THAT(sink.GetLogLines(), AllOf(ContainsSubstr("main level 1"),
                                        Not(ContainsSubstr("main level 4")),
                                        Not(ContainsSubstr("helper level 1")),
                                        Not(ContainsSubstr("helper level 4"))));
}
TEST(ScopedSetVlogLevelTest, MultipleEntry) {
  LogCollectorSink sink;
  absl::AddLogSink(&sink);
  ScopedSetVlogLevel set_vlog_level{{"scoped_vlog_level_test", 1},
                                    {"scoped_vlog_level_test_helper", 5}};
  EmitVlogMessage("main level 1");
  EmitVlog4Message("main level 4");
  EmitHelperVlogMessage("helper level 1");
  EmitHelperVlog4Message("helper level 4");
  absl::RemoveLogSink(&sink);
  EXPECT_THAT(sink.GetLogLines(), AllOf(ContainsSubstr("main level 1"),
                                        Not(ContainsSubstr("main level 4")),
                                        ContainsSubstr("helper level 1"),
                                        ContainsSubstr("helper level 4")));
}

}  // namespace
}  // namespace xls
