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

#ifndef XLS_DATA_STRUCTURES_BINARY_SEARCH_H_
#define XLS_DATA_STRUCTURES_BINARY_SEARCH_H_

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xls/common/integral_types.h"

namespace xls {

// Finds the maximum value in the inclusive range [start, end] for which the
// given function returns true. The given function must be monotonic over the
// range (range of zero or more true values followed by zero or more false
// values). CHECK fails if no value such exists.
int64 BinarySearchMaxTrue(int64 start, int64 end,
                          absl::FunctionRef<bool(int64 i)> f);

// Finds the minimum value in the inclusive range [start, end] for which the
// given function returns true. The given function must be monotonic over the
// range (range of zero or more false values followed by zero or more true
// values). CHECK fails if no value such exists.
int64 BinarySearchMinTrue(int64 start, int64 end,
                          absl::FunctionRef<bool(int64 i)> f);

// Overloads which accept a StatusOr function.
absl::StatusOr<int64> BinarySearchMaxTrueWithStatus(
    int64 start, int64 end, absl::FunctionRef<absl::StatusOr<bool>(int64 i)> f);
absl::StatusOr<int64> BinarySearchMinTrueWithStatus(
    int64 start, int64 end, absl::FunctionRef<absl::StatusOr<bool>(int64 i)> f);

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_BINARY_SEARCH_H_
