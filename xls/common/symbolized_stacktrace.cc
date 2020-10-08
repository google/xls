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

#include "xls/common/symbolized_stacktrace.h"

#include <array>
#include <vector>

#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "absl/strings/str_format.h"

namespace xls {

std::string GetSymbolizedStackTraceAsString(int max_depth, int skip_count,
                                            bool demangle) {
  std::string result;
  int skip_count_including_self = skip_count + 1;
  std::vector<void*> stack_trace;
  stack_trace.resize(max_depth);
  stack_trace.resize(absl::GetStackTrace(stack_trace.data(), max_depth,
                                         skip_count_including_self));
  std::array<char, 256> symbol_name_buffer;
  for (void* pc : stack_trace) {
    if (absl::Symbolize(pc, symbol_name_buffer.data(),
                        symbol_name_buffer.size())) {
      result += absl::StrFormat("%08p: %s\n", pc, symbol_name_buffer.data());
    } else {
      result += absl::StrFormat("%08p: [unknown]\n", pc);
    }
  }
  return result;
}

}  // namespace xls
