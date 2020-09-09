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

#ifndef XLS_COMMON_STATUS_STATUSOR_H_
#define XLS_COMMON_STATUS_STATUSOR_H_

#include "absl/status/statusor.h"

// The xabsl namespace has types that are anticipated to become available in
// Abseil reasonably soon, at which point they can be removed. These types are
// not in the xls namespace to make it easier to search/replace migrate usages
// to Abseil in the future.
namespace xabsl {

// DEPRECATED: Use absl::StatusOr directly.
template <typename T>
using StatusOr = absl::StatusOr<T>;

}  // namespace xabsl

#endif  // XLS_COMMON_STATUS_STATUSOR_H_
