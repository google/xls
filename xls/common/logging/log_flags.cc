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

ABSL_FLAG(int, minloglevel,
          static_cast<int>(absl::LogSeverity::kInfo),
          "Messages logged at a lower level than this don't actually "
          "get logged anywhere");

ABSL_FLAG(bool, logtostderr, false,
          "log messages go to stderr instead of logfiles");

ABSL_FLAG(bool, alsologtostderr, false,
          "log messages go to stderr in addition to logfiles");

ABSL_FLAG(int, stderrthreshold, static_cast<int>(absl::LogSeverity::kError),
          "log messages at or above this level are copied to stderr in "
          "addition to logfiles. This flag obsoletes --alsologtostderr. ");

ABSL_FLAG(bool, log_prefix, true,
          "Prepend the log prefix to the start of each log line");

namespace absl {

// Normalize the given value to a valid LogSeverityAtLeast value. Out of bounds
// values are clipped to kInfo (0) or kFatal (3).
static absl::LogSeverityAtLeast NormalizedSeverity(int parameter) {
  if (parameter < 0) {
    return absl::LogSeverityAtLeast::kInfo;
  } else if (parameter > 3) {
    return absl::LogSeverityAtLeast::kFatal;
  }
  return static_cast<absl::LogSeverityAtLeast>(parameter);
}

absl::LogSeverityAtLeast StderrThreshold() {
  if (absl::GetFlag(FLAGS_logtostderr) ||
      absl::GetFlag(FLAGS_alsologtostderr)) {
    return absl::LogSeverityAtLeast::kInfo;
  } else {
    return NormalizedSeverity(absl::GetFlag(FLAGS_stderrthreshold));
  }
}

}  // namespace absl
