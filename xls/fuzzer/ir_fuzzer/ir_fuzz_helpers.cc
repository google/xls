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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Changes the bit width of an inputted BValue to a specified bit width.
// Multiple bit width modification methods can be used and specified.
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width,
                      WidthFittingMethodProto* width_fitting_method) {
  if (bvalue.BitCountOrDie() > bit_width) {
    // If the bvalue is larger than the bit width, use the decrease width method
    // to reduce the bit width.
    return DecreaseBitWidth(fb, bvalue, bit_width,
                            width_fitting_method->decrease_width_method());
  } else if (bvalue.BitCountOrDie() < bit_width) {
    // If the bvalue is smaller than the bit width, use the increase width
    // method to increase the bit width.
    return IncreaseBitWidth(fb, bvalue, bit_width,
                            width_fitting_method->increase_width_method());
  } else {
    // If the bvalue is the same bit width as the specified bit width, return
    // the bvalue as is because no change is needed.
    return bvalue;
  }
}

// Overloaded version of ChangeBitWidth that uses default width fitting methods.
BValue ChangeBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width) {
  WidthFittingMethodProto default_method;
  default_method.set_decrease_width_method(
      DecreaseWidthMethod::UNSET_DECREASE_WIDTH_METHOD);
  default_method.set_increase_width_method(
      IncreaseWidthMethod::UNSET_INCREASE_WIDTH_METHOD);
  return ChangeBitWidth(fb, bvalue, bit_width, &default_method);
}

BValue DecreaseBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width,
                        DecreaseWidthMethod decrease_width_method) {
  switch (decrease_width_method) {
    case BIT_SLICE_METHOD:
    default:
      return fb->BitSlice(bvalue, 0, bit_width);
  }
}

BValue IncreaseBitWidth(FunctionBuilder* fb, BValue bvalue, int64_t bit_width,
                        IncreaseWidthMethod increase_width_method) {
  switch (increase_width_method) {
    case SIGN_EXTEND_METHOD:
      return fb->SignExtend(bvalue, bit_width);
    case ZERO_EXTEND_METHOD:
    default:
      return fb->ZeroExtend(bvalue, bit_width);
  }
}

// Accepts a string representing a byte array, which represents an integer. This
// byte array is converted into a Bits object, then truncated or zero extended
// to the specified bit width.
Bits ChangeBytesBitWidth(std::string bytes, int64_t bit_width) {
  Bits bits = Bits::FromBytes(
      absl::MakeSpan(reinterpret_cast<const uint8_t*>(bytes.data()),
                     bytes.size()),
      bytes.size() * 8);
  if (bits.bit_count() > bit_width) {
    bits = bits_ops::Truncate(bits, bit_width);
  } else if (bits.bit_count() < bit_width) {
    bits = bits_ops::ZeroExtend(bits, bit_width);
  }
  return bits;
}

// Returns a default integer if the integer is not in bounds.
int64_t Bounded(int64_t value, int64_t left_bound, int64_t right_bound) {
  CHECK_LE(left_bound, right_bound);
  if (left_bound <= value && value <= right_bound) {
    return value;
  } else {
    return left_bound;
  }
}

// Returns a default bit width if it is not in range. Bit width cannot be
// negative, cannot be 0, and cannot be too large otherwise the test will run
// out of memory or take too long.
int64_t BoundedWidth(int64_t bit_width, int64_t left_bound,
                     int64_t right_bound) {
  return Bounded(bit_width, left_bound, right_bound);
}

}  // namespace xls
