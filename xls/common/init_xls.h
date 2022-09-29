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

#ifndef XLS_COMMON_INIT_XLS_H_
#define XLS_COMMON_INIT_XLS_H_

#include <vector>

#include "absl/strings/string_view.h"

namespace xls {

// Initializes global state in the binary, including things like command line
// flags, symbolizer etc. This function might exit the program, for example if
// the command line flags are invalid or if a '--help' command line argument was
// provided.
//
// Is typically called early on in main().
//
// `usage` provides a short usage message passed to
// absl::SetProgramUsageMessage(). It can be nullptr, then it's ignored.
//
// `argc` and `argv` are the command line flags to parse. This function does not
// modify `argc`.
//
// Returns a vector of the positional arguments that are not part of any
// command-line flag (or arguments to a flag), not including the program
// invocation name. (This includes positional arguments after the
// flag-terminating delimiter '--'.)
std::vector<std::string_view> InitXls(std::string_view usage, int argc,
                                       char* argv[]);

}  // namespace xls

#endif  // XLS_COMMON_INIT_XLS_H_
