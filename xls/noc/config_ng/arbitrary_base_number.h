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

#ifndef XLS_NOC_CONFIG_NG_ARBITRARY_BASE_NUMBER_H_
#define XLS_NOC_CONFIG_NG_ARBITRARY_BASE_NUMBER_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/bits.h"

namespace xls::noc {

// A representation of an arbitrary base number with fixed digits and a
// numerical base.
class ArbitraryBaseNumber {
 public:
  // digit_count: the number of digits for the number. Its value must be
  // greater than zero. numerical_base: the numerical base for the number. Its
  // value must be greater than one and less than or equal to 128.
  ArbitraryBaseNumber(int64_t digit_count, int64_t numerical_base);

  // Returns the decimal representation of the Arbitrary Base Number of the
  // digits starting from an index. The least significant digit is at index
  // zero. Example:
  //   Assume a number with four digits of numerical base 2. The value
  //   of the number is [1, 0, 1, 0], 10 in decimal. GetValue(0)
  //   returns 10. GetValue(1) returns 5. GetValue(2) returns 2. GetValue(3)
  //   returns 1.
  //
  // If start_index is out of bounds, returns nullopt.
  std::optional<int64_t> GetValue(int64_t start_index = 0) const;

  // Sets all digits of the number to zero.
  void Reset();

  // TODO(vmirian) 02-05-21 Add arbitrary value
  // Adds one to the number. Returns the status of the overflow bit after the
  // addition.
  bool AddOne();

  // Swap digits at the given indices.
  // Returns an absl::OutOfRangeError error if the indices are out of range.
  // The least significant digit is at index zero.
  absl::Status SwapDigits(int64_t first_digit_index,
                          int64_t second_digit_index);

  int64_t GetDigitCount() const;
  int64_t GetNumericalBase() const;

 private:
  std::vector<char> digits_;
  int64_t numerical_base_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_ARBITRARY_BASE_NUMBER_H_
