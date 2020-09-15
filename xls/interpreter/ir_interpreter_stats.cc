// Copyright 2020 Google LLC
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

#include "xls/interpreter/ir_interpreter_stats.h"

namespace xls {

std::string InterpreterStats::ToNodeReport() const {
  std::string result;
  for (const auto& item : value_profile_) {
    if (ternary_ops::AllUnknown(item.second.value())) {
      continue;
    }
    absl::StrAppendFormat(&result, " %s: %s\n", item.first,
                          ToString(item.second.value()));
  }
  return result;
}

std::string InterpreterStats::ToReport() const {
  absl::MutexLock lock(&mutex_);
  auto percent = [](int64 value, int64 all) -> double {
    if (all == 0) {
      return 100.0;
    }
    return static_cast<double>(value) / all * 100.0;
  };
  return absl::StrFormat(
             R"(Interpreter stats report:
shll:       %d
 zero:      %d (%.2f%%)
 overlarge: %d (%.2f%%)
 in-range:  %d (%.2f%%)
)",
             all_shlls_, zero_shlls_, percent(zero_shlls_, all_shlls_),
             overlarge_shlls_, percent(overlarge_shlls_, all_shlls_),
             in_range_shlls(), percent(in_range_shlls(), all_shlls_)) +
         ToNodeReport();
}

}  // namespace xls
