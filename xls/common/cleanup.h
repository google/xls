// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_COMMON_CLEANUP_H_
#define THIRD_PARTY_XLS_COMMON_CLEANUP_H_

#include <utility>

#include "absl/types/optional.h"

namespace xabsl {

// Simple RAII template to call a specified function on destruction, e.g., to
// perform "clean-up" operations when exiting a scope.
template <typename CallbackT>
class Cleanup {
 public:
  Cleanup(CallbackT&& callback)
      : callback_(std::forward<CallbackT>(callback)) {}

  ~Cleanup() {
    if (callback_.has_value()) {
      callback_.value()();
    }
  }

  void Cancel() { callback_.reset(); }

 private:
  absl::optional<CallbackT> callback_;
};

// Helper function to enable automatic type deduction, e.g.,
//   "auto x = xabsl::MakeCleanup(...);"
template <typename CallbackT>
xabsl::Cleanup<absl::decay_t<CallbackT>> MakeCleanup(CallbackT&& callback) {
  return xabsl::Cleanup<absl::decay_t<CallbackT>>(
      std::forward<CallbackT>(callback));
}

}  // namespace xabsl

#endif  // THIRD_PARTY_XLS_COMMON_CLEANUP_H_
