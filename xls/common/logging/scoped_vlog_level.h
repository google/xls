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

#ifndef XLS_COMMON_LOGGING_SCOPED_VLOG_LEVEL_H_
#define XLS_COMMON_LOGGING_SCOPED_VLOG_LEVEL_H_

#include <string_view>

#include "absl/log/globals.h"

namespace xls {

// Set the vlog level to the given level for a scope.
class ScopedSetVlogLevel {
 public:
  ScopedSetVlogLevel(std::string_view pattern, int level)
      : pattern_(pattern), level_(absl::SetVLogLevel(pattern_, level)) {}
  ~ScopedSetVlogLevel() { absl::SetVLogLevel(pattern_, level_); }

 private:
  std::string_view pattern_;
  int level_;
};

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_SCOPED_VLOG_LEVEL_H_
