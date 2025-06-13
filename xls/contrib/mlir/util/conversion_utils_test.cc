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

#include "xls/contrib/mlir/util/conversion_utils.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/inlined_vector.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"
#include "llvm/include/llvm/ADT/Sequence.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"

namespace mlir::xls {
namespace {
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::xls::Bits;

// Converts an APInt to a Bits. This is a naive conversion that is not efficient
// but is guaranteed to be correct.
Bits NaiveAPIntToBits(APInt apInt) {
  absl::InlinedVector<bool, 64> bits(apInt.getBitWidth());
  for (unsigned i : llvm::seq(0u, apInt.getBitWidth())) {
    bits[i] = apInt[i];
  }
  return ::xls::Bits(bits);
}

// Converts a Bits to an APInt. This is a naive conversion that is not efficient
// but is guaranteed to be correct.
APInt NaiveBitsToAPInt(Bits b) {
  APInt converted_value;
  if (b.bit_count() == 0) {
    converted_value = APInt(/*numBits=*/0, /*val=*/0, /*isSigned=*/false,
                            /*implicitTrunc=*/false);
  } else {
    uint64_t num_words = b.bitmap().word_count();
    SmallVector<uint64_t> words;
    words.reserve(num_words);
    for (uint64_t i = 0; i < num_words; i++) {
      words.push_back(b.bitmap().GetWord(i));
    }
    converted_value = APInt(b.bit_count(), ArrayRef(words));
  }
  return converted_value;
}

void CompareAPIntToBits(APInt apInt) {
  Bits bits = bitsFromAPInt(apInt);
  Bits naive = NaiveAPIntToBits(apInt);
  EXPECT_EQ(bits, naive);
}

void CompareBitsToAPInt(Bits bits) {
  APInt apInt = bitsToAPInt(bits);
  APInt naive = NaiveBitsToAPInt(bits);
  EXPECT_EQ(apInt, naive);
}

// Fuzzer domain that returns a Bits.
auto BitsDomain(int64_t maxBits = 256) {
  auto bits = fuzztest::ContainerOf<absl::InlinedVector<bool, 64>>(
      fuzztest::Arbitrary<bool>());
  return fuzztest::Map(
      [](const absl::InlinedVector<bool, 64>& bits) { return Bits(bits); },
      bits);
}

// Fuzzer domain that returns an APInt.
auto APIntDomain(int64_t maxBits = 256) {
  auto bits = fuzztest::ContainerOf<absl::InlinedVector<bool, 64>>(
      fuzztest::Arbitrary<bool>());
  return fuzztest::Map(
      [](const absl::InlinedVector<bool, 64>& bits) {
        APInt apInt(bits.size(), 0, /*isSigned=*/false,
                    /*implicitTrunc=*/false);
        for (int64_t i = 0; i < bits.size(); ++i) {
          apInt.setBitVal(i, bits[i]);
        }
        return apInt;
      },
      bits);
}

FUZZ_TEST(FuzzBitsToAPInt, CompareBitsToAPInt).WithDomains(BitsDomain());
FUZZ_TEST(FuzzAPIntToBits, CompareAPIntToBits).WithDomains(APIntDomain());

// Define extra fuzz tests that focus on small values to cover edge cases.
FUZZ_TEST(FuzzBitsToAPInt_Small, CompareBitsToAPInt)
    .WithDomains(BitsDomain(16));
FUZZ_TEST(FuzzAPIntToBits_Small, CompareAPIntToBits)
    .WithDomains(APIntDomain(16));

}  // namespace
}  // namespace mlir::xls
