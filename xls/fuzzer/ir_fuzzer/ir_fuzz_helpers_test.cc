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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "cppitertools/range.hpp"
#include "cppitertools/sliding_window.hpp"
#include "cppitertools/zip.hpp"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"

namespace xls {
namespace {

using ::testing::AllOf;
using ::testing::Each;
using ::testing::Ge;
using ::testing::Le;

class IrFuzzHelpersTest : public ::testing::TestWithParam<FuzzVersion> {
 public:
  IrFuzzHelpers helpers() const { return IrFuzzHelpers(GetParam()); }
};

TEST_P(IrFuzzHelpersTest, ChangeBytesBitWidthWithNoChange) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = helpers().ChangeBytesBitWidth(bytes, 3);
  EXPECT_EQ(bits.ToUint64().value(), 7);
}

TEST_P(IrFuzzHelpersTest, Bounded) {
  std::vector<int64_t> res;
  for (int64_t i = -10; i < 11; ++i) {
    res.push_back(helpers().Bounded(i, 1, 7));
  }
  EXPECT_THAT(res, Each(AllOf(Le(7), Ge(1))));
  RecordProperty("res", testing::PrintToString(res));
  auto nxt = [](int64_t i) -> int64_t {
    if (i != 7) {
      return i + 1;
    }
    return 1;
  };
  if (GetParam() >= FuzzVersion::BOUND_WITH_MODULO_VERSION) {
    for (const auto& [idx, w] :
         iter::zip(iter::range(-10, 11), iter::sliding_window(res, 2))) {
      if (idx == -1) {
        // Discontinuity across 0.
        continue;
      }
      EXPECT_EQ(w[1], nxt(w[0])) << idx;
    }
  }
}

TEST_P(IrFuzzHelpersTest, ChangeBytesBitWidthWithTruncate) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = helpers().ChangeBytesBitWidth(bytes, 2);
  EXPECT_EQ(bits.ToUint64().value(), 3);
}

TEST_P(IrFuzzHelpersTest, ChangeBytesBitWidthWithZeroExtend) {
  std::string bytes = absl::StrFormat("%c", 7);
  Bits bits = helpers().ChangeBytesBitWidth(bytes, 4);
  EXPECT_EQ(bits.ToUint64().value(), 7);
}

TEST_P(IrFuzzHelpersTest, ChangeBytesBitWidthWithLargeInput) {
  std::string bytes = "\xff\xff";
  Bits bits = helpers().ChangeBytesBitWidth(bytes, 16);
  EXPECT_EQ(bits.ToUint64().value(), 65535);
}

INSTANTIATE_TEST_SUITE_P(
    IrFuzzHelpersTest, IrFuzzHelpersTest,
    ::testing::Values(FuzzVersion::UNSET_FUZZ_VERSION,
                      FuzzVersion::BOUND_WITH_MODULO_VERSION),
    [](const testing::TestParamInfo<FuzzVersion>& v) {
      return FuzzVersion_Name(v.param);
    });

}  // namespace
}  // namespace xls
