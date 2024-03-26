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

#include "xls/data_structures/binary_search.h"

#include <cstdint>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

int64_t BinarySearchMaxTrue(int64_t start, int64_t end,
                            absl::FunctionRef<bool(int64_t i)> f,
                            BinarySearchAssumptions assumptions) {
  return BinarySearchMaxTrueWithStatus(
             start, end,
             [&](int64_t i) -> absl::StatusOr<bool> { return f(i); },
             assumptions)
      .value();
}

int64_t BinarySearchMinTrue(int64_t start, int64_t end,
                            absl::FunctionRef<bool(int64_t i)> f,
                            BinarySearchAssumptions assumptions) {
  return BinarySearchMinTrueWithStatus(
             start, end,
             [&](int64_t i) -> absl::StatusOr<bool> { return f(i); },
             assumptions)
      .value();
}

absl::StatusOr<int64_t> BinarySearchMaxTrueWithStatus(
    int64_t start, int64_t end,
    absl::FunctionRef<absl::StatusOr<bool>(int64_t i)> f,
    BinarySearchAssumptions assumptions) {
  XLS_RET_CHECK_LE(start, end);
  if (assumptions != BinarySearchAssumptions::kStartKnownTrue) {
    XLS_ASSIGN_OR_RETURN(bool f_start, f(start));
    if (!f_start) {
      return absl::InvalidArgumentError(
          "Lowest value in range fails condition of binary search.");
    }
  }
  XLS_ASSIGN_OR_RETURN(bool f_end, f(end));
  if (f_end) {
    return end;
  }
  int64_t highest_true = start;
  int64_t lowest_false = end;
  while (highest_true < lowest_false - 1) {
    int64_t middle = highest_true + (lowest_false - highest_true) / 2;
    XLS_ASSIGN_OR_RETURN(bool f_middle, f(middle));
    if (f_middle) {
      highest_true = middle;
    } else {
      lowest_false = middle;
    }
  }
  return highest_true;
}

absl::StatusOr<int64_t> BinarySearchMinTrueWithStatus(
    int64_t start, int64_t end,
    absl::FunctionRef<absl::StatusOr<bool>(int64_t i)> f,
    BinarySearchAssumptions assumptions) {
  XLS_RET_CHECK_LE(start, end);
  if (assumptions != BinarySearchAssumptions::kEndKnownTrue) {
    XLS_ASSIGN_OR_RETURN(bool f_end, f(end));
    if (!f_end) {
      return absl::InvalidArgumentError(
          "Highest value in range fails condition of binary search.");
    }
  }
  XLS_ASSIGN_OR_RETURN(bool f_start, f(start));
  if (f_start) {
    return start;
  }
  int64_t highest_false = start;
  int64_t lowest_true = end;
  while (highest_false < lowest_true - 1) {
    int64_t middle = highest_false + (lowest_true - highest_false) / 2;
    XLS_ASSIGN_OR_RETURN(bool f_middle, f(middle));
    if (f_middle) {
      lowest_true = middle;
    } else {
      highest_false = middle;
    }
  }
  return lowest_true;
}

}  // namespace xls
