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

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"

namespace xls {

enum class BinarySearchAssumptions {
  kNone,
  kStartKnownTrue,
  kEndKnownTrue,
};

// Finds the maximum value in the inclusive range [start, end] for which the
// given function returns true. The given function must be monotonic over the
// range (range of zero or more true values followed by zero or more false
// values). CHECK fails if no value such exists.
int64_t BinarySearchMaxTrue(
    int64_t start, int64_t end, absl::FunctionRef<bool(int64_t i)> f,
    BinarySearchAssumptions assumptions = BinarySearchAssumptions::kNone);

// Finds the minimum value in the inclusive range [start, end] for which the
// given function returns true. The given function must be monotonic over the
// range (range of zero or more false values followed by zero or more true
// values). CHECK fails if no value such exists.
int64_t BinarySearchMinTrue(
    int64_t start, int64_t end, absl::FunctionRef<bool(int64_t i)> f,
    BinarySearchAssumptions assumptions = BinarySearchAssumptions::kNone);

// Overloads which accept a StatusOr function.
absl::StatusOr<int64_t> BinarySearchMaxTrueWithStatus(
    int64_t start, int64_t end,
    absl::FunctionRef<absl::StatusOr<bool>(int64_t i)> f,
    BinarySearchAssumptions assumptions = BinarySearchAssumptions::kNone);
absl::StatusOr<int64_t> BinarySearchMinTrueWithStatus(
    int64_t start, int64_t end,
    absl::FunctionRef<absl::StatusOr<bool>(int64_t i)> f,
    BinarySearchAssumptions assumptions = BinarySearchAssumptions::kNone);

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_BINARY_SEARCH_H_
