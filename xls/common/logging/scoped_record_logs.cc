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

#include "xls/common/logging/scoped_record_logs.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/log_sink_registry.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/collect_logs.h"

namespace xls {

ScopedRecordLogs::ScopedRecordLogs(std::string_view property)
    : sink_(std::make_unique<LogCollectorSink>()), property_(property) {
  absl::AddLogSink(sink_.get());
}

ScopedRecordLogs::~ScopedRecordLogs() {
  absl::RemoveLogSink(sink_.get());
  std::vector<std::string> log_lines = sink_->GetLogLines();
  if (!log_lines.empty()) {
    testing::Test::RecordProperty(property_, absl::StrJoin(log_lines, "\n"));
  }
}

}  // namespace xls
