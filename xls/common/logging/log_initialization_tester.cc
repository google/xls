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

// This is a test-only utility which simply logs some precanned messages at
// various log levels using LOG and VLOG. Used for testing logging.

#include <cstdlib>

#include "absl/log/log.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);

  XLS_LOG(INFO) << "INFO message";
  XLS_LOG(WARNING) << "WARNING message";
  XLS_LOG(ERROR) << "ERROR message";

  XLS_VLOG(1) << "XLS_VLOG(1) message";
  XLS_VLOG(2) << "XLS_VLOG(2) message";

  if (VLOG_IS_ON(1)) {
    XLS_LOG(INFO) << "VLOG_IS_ON(1) message\n";
  }
  if (VLOG_IS_ON(2)) {
    XLS_LOG(INFO) << "VLOG_IS_ON(2) message\n";
  }

  return EXIT_SUCCESS;
}
