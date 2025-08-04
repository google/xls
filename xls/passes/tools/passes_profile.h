// Copyright 2025 The XLS Authors
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

#ifndef XLS_PASSES_TOOLS_PASSES_PROFILE_H_
#define XLS_PASSES_TOOLS_PASSES_PROFILE_H_

#include <cstdint>
#include <string_view>
#include <variant>

namespace xls {

// Add pass `short_name` to the top of the stack of passes.
void RecordPassEntry(std::string_view short_name);

namespace pass_profile {
constexpr static std::string_view kConfigKey = "configuration";
constexpr static std::string_view kCompound = "compound";
constexpr static std::string_view kFixedpoint = "fixedpoint";
constexpr static std::string_view kNodeCountBefore = "node-count-before";
constexpr static std::string_view kNodeCountAfter = "node-count-after";
}  // namespace pass_profile
// Add a label containing additional details about the current pass being run.
// Each 'key' should only be used once. Prefer the common keys noted above.
void RecordPassAnnotation(std::string_view key,
                          std::variant<std::string_view, int64_t> contents);

// Mark the pass at the top of the stack as finished with either a changed or
// unchanged state.
void ExitPass(bool changed);

}  // namespace xls

#endif  // XLS_PASSES_TOOLS_PASSES_PROFILE_H_
