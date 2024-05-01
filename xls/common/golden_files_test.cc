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

#include "xls/common/golden_files.h"

#include <filesystem>  // NOLINT
#include <string>

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"

namespace xls {
namespace {

constexpr char kTestName[] = "golden_files_test";
constexpr char kTestdataPath[] = "xls/common/testdata";

class GoldenFilesTest : public ::testing::Test {
 protected:
  std::filesystem::path GoldenFilePath() {
    return absl::StrFormat(
        "%s/%s_%s.txt", kTestdataPath, kTestName,
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
  }
};

TEST_F(GoldenFilesTest, SuccessfulComparison) {
  std::string text = R"(this
is
a
golden
file)";
  ExpectEqualToGoldenFile(GoldenFilePath(), text);
}

TEST_F(GoldenFilesTest, EmptyComparison) {
  std::string text;
  ExpectEqualToGoldenFile(GoldenFilePath(), text);
}

TEST_F(GoldenFilesTest, Miscomparison) {
  std::string text = "not the content in the file";
  EXPECT_NONFATAL_FAILURE(ExpectEqualToGoldenFile(GoldenFilePath(), text), "");
}

}  // namespace
}  // namespace xls
