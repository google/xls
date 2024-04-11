// Copyright 2024 The XLS Authors
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

#include "xls/solvers/z3_extract_counterexample.h"

#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::solvers::z3 {
namespace {

// Avoids commas in macros.
using NameToValue = absl::flat_hash_map<std::string, Value>;

TEST(ExtractCounterexampleTest, SingleBitsParam) {
  constexpr std::string_view kMessage = R"(  Model:
```f -> #x0
```
)";
  Package package("test_package");
  std::vector<IrParamSpec> signature = {
      IrParamSpec{.name = "f", .type = package.GetBitsType(5)}};

  XLS_ASSERT_OK_AND_ASSIGN(NameToValue values,
                           ExtractCounterexample(kMessage, signature));
  EXPECT_EQ(values.size(), 1);
  EXPECT_EQ(values.at("f"), Value(UBits(0, 5)));
}

// The values given back are in hex.
TEST(ExtractCounterexampleTest, SingleBitsParamHexValue) {
  constexpr std::string_view kMessage = R"(  Model:
```f -> #xa
```
)";
  Package package("test_package");
  std::vector<IrParamSpec> signature = {
      IrParamSpec{.name = "f", .type = package.GetBitsType(5)}};

  XLS_ASSERT_OK_AND_ASSIGN(NameToValue values,
                           ExtractCounterexample(kMessage, signature));
  EXPECT_EQ(values.size(), 1);
  EXPECT_EQ(values.at("f"), Value(UBits(0xa, 5)));
}

TEST(ExtractCounterexampleTest, FakeMessageTooManyParams) {
  constexpr std::string_view kMessage = R"(  Model:
```f -> #xa
g -> #xa
```
)";
  Package package("test_package");
  std::vector<IrParamSpec> signature = {
      IrParamSpec{.name = "f", .type = package.GetBitsType(5)}};

  EXPECT_THAT(ExtractCounterexample(kMessage, signature),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("could not find parameter name from model "
                                     "in user-provided spec: `g`")));
}

// Note: Z3 can (and sometimes does) give us the parameters in an arbitrary
// order in its message.
TEST(ExtractCounterexampleTest, BackwardsParams) {
  constexpr std::string_view kMessage = R"(  Model:
```param_1 -> #x00
param_0 -> #xa0
```
)";
  Package package("test_package");
  std::vector<IrParamSpec> signature = {
      IrParamSpec{.name = "param_0", .type = package.GetBitsType(8)},
      IrParamSpec{.name = "param_1", .type = package.GetBitsType(8)},
  };

  XLS_ASSERT_OK_AND_ASSIGN(NameToValue values,
                           ExtractCounterexample(kMessage, signature));
  EXPECT_EQ(values.size(), 2);
  EXPECT_EQ(values.at("param_0"), Value(UBits(0xa0, 8)));
  EXPECT_EQ(values.at("param_1"), Value(UBits(0x00, 8)));
}

void DoesNotCrashWithBitsTypeParam(std::string_view input) {
  Package package("test_package");
  std::vector<IrParamSpec> signature = {
      IrParamSpec{.name = "x", .type = package.GetBitsType(2)}};
  absl::StatusOr<NameToValue> results = ExtractCounterexample(input, signature);
  if (results.ok()) {
    EXPECT_EQ(results->size(), 1);
    EXPECT_TRUE(results->contains("x"));
  } else {
    EXPECT_EQ(results.status().code(), absl::StatusCode::kInvalidArgument);
  }
}
FUZZ_TEST(ExtractCounterexampleFuzzTest, DoesNotCrashWithBitsTypeParam);

}  // namespace
}  // namespace xls::solvers::z3
