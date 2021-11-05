// Copyright 2021 The XLS Authors
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

#include "xls/ir/register.h"

#include "absl/strings/str_format.h"

namespace xls {

std::string Register::ToString() const {
  if (reset().has_value()) {
    return absl::StrFormat(
        "reg %s(%s, reset_value=%s, asynchronous=%s, active_low=%s)\n", name(),
        type()->ToString(), reset().value().reset_value.ToHumanString(),
        reset().value().asynchronous ? "true" : "false",
        reset().value().active_low ? "true" : "false");
  }
  return absl::StrFormat("reg %s(%s)", name(), type()->ToString());
}

}  // namespace xls
