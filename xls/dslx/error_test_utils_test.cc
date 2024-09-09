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

#include "xls/dslx/error_test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {
namespace {

TEST(IsPosErrorTest, VariousErrors) {
  EXPECT_THAT(absl::InternalError("ScanError: test.x:1:2 message"),
              testing::Not(IsPosError("ScanError", testing::_)));
  EXPECT_THAT(
      absl::InvalidArgumentError("TypeInferenceError: test.x:1:2 message"),
      testing::Not(IsPosError("ScanError", testing::_)));
  EXPECT_THAT(
      absl::InvalidArgumentError("ScanError: <> missing positional data"),
      testing::Not(IsPosError("ScanError", testing::_)));
  FileTable file_table;
  EXPECT_THAT(ScanErrorStatus(FakeSpan(), "my message", file_table),
              IsPosError("ScanError", "my message"));
}

}  // namespace
}  // namespace xls::dslx
