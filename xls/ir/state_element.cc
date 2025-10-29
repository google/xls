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

#include "xls/ir/state_element.h"

#include <string>

#include "absl/strings/str_format.h"

namespace xls {

std::string StateElement::ToString() const {
  return absl::StrFormat("state %s(%s, initial_value=%s, non_synth=%s)\n",
                         name(), type()->ToString(),
                         initial_value().ToHumanString(),
                         non_synthesizable() ? "true" : "false");
}

}  // namespace xls
