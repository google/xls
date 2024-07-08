// Copyright 2023 The XLS Authors
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

#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"

namespace {

using ::testing::Not;
using ::xls::status_testing::IsOk;
using ::xls::status_testing::StatusIs;

constexpr size_t kMaxModuleLengthInBytes = 5'000;

void ScanningGivesErrorOrConvertsToOriginal(const std::string& test_module) {
  xls::dslx::Scanner scanner("fake.x", test_module,
                             /*include_whitespace_and_comments=*/true);
  absl::StatusOr<std::vector<xls::dslx::Token>> tokens = scanner.PopAll();
  if (tokens.ok()) {
    std::string reversed = absl::StrJoin(
        tokens.value(), "", [](std::string* out, const xls::dslx::Token& t) {
          absl::StrAppend(out, t.ToString());
        });
    EXPECT_EQ(test_module, reversed);
    return;
  }

  // Check that the error is positional and not internal.
  ASSERT_THAT(tokens.status(), Not(StatusIs(absl::StatusCode::kInternal)));
  EXPECT_THAT(xls::dslx::GetPositionalErrorData(tokens.status()), IsOk());
}
FUZZ_TEST(ScanFuzzTest, ScanningGivesErrorOrConvertsToOriginal)
    .WithDomains(fuzztest::Arbitrary<std::string>().WithMaxSize(
        kMaxModuleLengthInBytes));

}  // namespace
