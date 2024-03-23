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

#include "xls/common/init_xls.h"

#include <string_view>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

namespace xls {

std::vector<std::string_view> InitXls(std::string_view usage, int argc,
                                      char* argv[]) {
  // Comply with xls/common/init_xls.h:32
  absl::SetProgramUsageMessage(usage);
  // Copy the argv array to ensure this method doesn't clobber argv.
  std::vector<char*> arguments(argv, argv + argc);
  std::vector<char*> remaining = absl::ParseCommandLine(argc, argv);
  CHECK_GE(argc, 1);

  internal::InitXlsPostAbslFlagParse();

  return std::vector<std::string_view>(remaining.begin() + 1, remaining.end());
}

namespace internal {
void InitXlsPostAbslFlagParse() { absl::InitializeLog(); }
}  // namespace internal

}  // namespace xls
