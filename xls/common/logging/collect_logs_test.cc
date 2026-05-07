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

#include "xls/common/logging/collect_logs.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"

namespace xls {
namespace {

using testing::Contains;
using testing::ElementsAre;
using testing::HasSubstr;

TEST(CollectLogsTest, GlobalSink) {
  LogCollectorSink sink;
  absl::AddLogSink(&sink);
  LOG(INFO) << "Hello";
  LOG(INFO) << "World";
  auto lines = sink.GetLogLines();
  EXPECT_THAT(lines, Contains(HasSubstr("Hello")));
  EXPECT_THAT(lines, Contains(HasSubstr("World")));
  absl::RemoveLogSink(&sink);
}

TEST(CollectLogsTest, OnlySink) {
  LogCollectorSink sink;
  LOG(INFO).ToSinkOnly(&sink) << "Hello";
  LOG(INFO).ToSinkOnly(&sink) << "World";
  auto lines = sink.GetLogLines();
  EXPECT_THAT(lines, ElementsAre(HasSubstr("Hello"), HasSubstr("World")));
}

}  // namespace
}  // namespace xls
