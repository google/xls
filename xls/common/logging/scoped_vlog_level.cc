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

#include "xls/common/logging/scoped_vlog_level.h"

#include <initializer_list>
#include <vector>

#include "absl/log/globals.h"
#include "absl/types/span.h"

namespace xls {
namespace {
template <typename VectorLike>
std::vector<ScopedSetVlogLevel::VlogLevel> SetLevels(const VectorLike& levels) {
  std::vector<ScopedSetVlogLevel::VlogLevel> original_levels;
  original_levels.reserve(levels.size());
  for (const auto& [pattern, level] : levels) {
    original_levels.push_back({pattern, absl::SetVLogLevel(pattern, level)});
  }
  return original_levels;
}
}  // namespace

ScopedSetVlogLevel::ScopedSetVlogLevel(absl::Span<const VlogLevel> levels)
    : original_levels_(SetLevels(levels)) {}

ScopedSetVlogLevel::ScopedSetVlogLevel(std::initializer_list<VlogLevel> levels)
    : original_levels_(SetLevels(levels)) {}

ScopedSetVlogLevel::~ScopedSetVlogLevel() { SetLevels(original_levels_); }

}  // namespace xls
