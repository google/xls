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

#ifndef XLS_COMMON_SYMBOLIZED_STACKTRACE_H_
#define XLS_COMMON_SYMBOLIZED_STACKTRACE_H_

#include <string>

namespace xls {

// Get the symbolized stack trace at most "max_depth" frames, skipping
// innermost "skip_count" frames, as a string. All symbol names will be
// simply connected with "\n". Useful for simple debug output.
//
// Example:
//     LOG(INFO) << "@@stacktrace\n" << GetSymbolizedStackTraceAsString(10);
//
std::string GetSymbolizedStackTraceAsString(int max_depth = 50,
                                            int skip_count = 0,
                                            bool demangle = true);

}  // namespace xls

#endif  // XLS_COMMON_SYMBOLIZED_STACKTRACE_H_
