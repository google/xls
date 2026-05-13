// Copyright 2026 The XLS Authors
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

#include "xls/common/logging/scoped_vlog_level_test_helper.h"

#include <string>

#include "absl/log/log.h"

namespace xls {

void EmitHelperVlogMessage(std::string message) { VLOG(1) << message; }
void EmitHelperVlog4Message(std::string message) { VLOG(4) << message; }

}  // namespace xls
