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

#include "xls/common/logging/log_flags.h"

#include "absl/log/globals.h"

namespace {
// Deduces the value of stderr threshold, based on the value of flags
// FLAGS_logtostderr
// FLAGS_alsologtostderr
// `turning_on_off` indicates that we are deducing the threshold, while turning
// above flags on or off. The deduction logic differs in these cases, since
// flags may start to contradict each other.
void DeduceStderrThreshold(bool turning_on_off) {
  // Turning on case
  // set threshold to INFO
  if (turning_on_off) {
    absl::log_internal::RawSetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
    return;
  }
  // Turning off case
  // if flags contradict each other, keep current threshold
  // otherwise set threshold to at least ERROR.
  if (!absl::GetFlag(FLAGS_logtostderr) &&
      !absl::GetFlag(FLAGS_alsologtostderr)) {
    absl::log_internal::RawSetStderrThreshold(
        (std::max)(absl::LogSeverityAtLeast::kError, absl::StderrThreshold()));
  }
}
}  // namespace

ABSL_FLAG(bool, logtostderr, false,
          "log messages go to stderr instead of logfiles")
    .OnUpdate([] {
      bool turning_on_off = absl::GetFlag(FLAGS_logtostderr);
      DeduceStderrThreshold(turning_on_off);
      // TODO(rigge): when abseil logging logs to files by default, disable
      // logging to file here. Maybe on that glorious day logtostderr will be
      // part of abseil.
      // EnableLogToFiles(!turning_on_off);
    });

ABSL_FLAG(bool, alsologtostderr, false,
          "log messages go to stderr in addition to logfiles")
    .OnUpdate([] {
      bool turning_on_off = absl::GetFlag(FLAGS_alsologtostderr);
      DeduceStderrThreshold(turning_on_off);
    });
