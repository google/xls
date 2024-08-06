// Copyright 2023 The XLS Authors
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

#include "xls/common/exit_status.h"

#include <cstdlib>
#include <iostream>

#include "absl/status/status.h"

namespace xls {

int ExitStatus(const absl::Status& status, bool log_on_error) {
  if (status.ok()) {
    return EXIT_SUCCESS;
  }
  if (log_on_error) {
    std::cerr << "Error: " << status;
  }
  return EXIT_FAILURE;
}

}  // namespace xls
