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

#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/ir/bits.h"

namespace xls {
namespace {

TEST(IrFuzzHelpersTest, ChangeBytesBitWidthWithNoChange) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = ChangeBytesBitWidth(bytes, 3);
  EXPECT_EQ(bits.ToUint64().value(), 7);
}

TEST(IrFuzzHelpersTest, ChangeBytesBitWidthWithTruncate) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = ChangeBytesBitWidth(bytes, 2);
  EXPECT_EQ(bits.ToUint64().value(), 3);
}

TEST(IrFuzzHelpersTest, ChangeBytesBitWidthWithZeroExtend) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = ChangeBytesBitWidth(bytes, 4);
  EXPECT_EQ(bits.ToUint64().value(), 7);
}

TEST(IrFuzzHelpersTest, ChangeBytesBitWidthWithLargeInput) {
  std::string bytes = "\xff\xff";
  Bits bits = ChangeBytesBitWidth(bytes, 16);
  EXPECT_EQ(bits.ToUint64().value(), 65535);
}

}  // namespace
}  // namespace xls
