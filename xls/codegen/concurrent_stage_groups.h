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

#ifndef XLS_CODEGEN_CONCURRENT_STAGE_GROUPS_H_
#define XLS_CODEGEN_CONCURRENT_STAGE_GROUPS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "xls/common/logging/logging.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {

// Records which pairs of stages may be concurrently active.
//
// This also includes the notion of stages being mutually exclusive with one
// another. Two stages are mutually exclusive with each other if at most one of
// them is active on any cycle. If both stages may be active on the same cycle
// then the stages are concurrent. These two states are inverses of each other.
//
// In practice, stages are mutually exclusive if there is some input (such as a
// proc-state element value) which both of them need the same instance of in
// order to activate.
class ConcurrentStageGroups {
 public:
  using Stage = int64_t;
  ConcurrentStageGroups(ConcurrentStageGroups&&) = default;
  ConcurrentStageGroups& operator=(ConcurrentStageGroups&&) = default;
  ConcurrentStageGroups(const ConcurrentStageGroups&) = default;
  ConcurrentStageGroups& operator=(const ConcurrentStageGroups&) = default;
  // Create a group with the initial state of all stages being concurrently
  // active with each other.
  explicit ConcurrentStageGroups(int64_t num_stages) {
    concurrent_stages_.reserve(num_stages);
    for (int64_t i = 0; i < num_stages; ++i) {
      concurrent_stages_.push_back(InlineBitmap(num_stages, true));
    }
  }

  // How many stages are present.
  int64_t stage_count() const { return concurrent_stages_.size(); }

  // Record the fact that two stages may not be concurrently active.
  void MarkMutuallyExclusive(Stage a, Stage b) {
    XLS_CHECK_NE(a, b);
    concurrent_stages_[a].Set(b, false);
    concurrent_stages_[b].Set(a, false);
  }

  bool IsConcurrent(Stage a, Stage b) const {
    XLS_CHECK_EQ(concurrent_stages_[a].Get(b), concurrent_stages_[b].Get(a))
        << "a=" << a << ", b=" << b << "\n"
        << *this;
    return concurrent_stages_[a].Get(b);
  }

  bool IsMutuallyExclusive(Stage a, Stage b) const {
    return !IsConcurrent(a, b);
  }

  // Get the bitmap of all stages which are potentially concurrently active with
  // the given stage.
  const InlineBitmap& ConcurrentStagesWith(int64_t stage) const {
    return concurrent_stages_[stage];
  }

  // Get a string description of the concurrency state.
  std::string ToString() const;

 private:
  // Map of stage# -> set of stages that can be executing at the same time as
  // the stage.  If concurrent_stages_[N].Get(M) is true then stage N and M may
  // be active at the same time.
  std::vector<InlineBitmap> concurrent_stages_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ConcurrentStageGroups& cg) {
  return os << cg.ToString();
}

}  // namespace xls

#endif  // XLS_CODEGEN_CONCURRENT_STAGE_GROUPS_H_
