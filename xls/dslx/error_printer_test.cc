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

#include "xls/dslx/error_printer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

TEST(PrintPositionalErrorTest, BasicErrorTest) {
  XLS_ASSERT_OK_AND_ASSIGN(TempFile temp,
                           TempFile::CreateWithContent(R"(line 1
line 2
line 3
line 4
line 5
line 6
line 7)",
                                                       "some_file.x"));

  const std::string filename = temp.path();
  const Pos start_pos(filename, 5, 0);
  const Pos limit_pos(filename, 5, 4);
  const Span error_span(start_pos, limit_pos);
  std::stringstream ss;
  XLS_ASSERT_OK(PrintPositionalError(error_span, "my error message", ss,
                                     /*color=*/false,
                                     /*error_context_line_count=*/3));
  std::string output = ss.str();
  std::vector<absl::string_view> output_lines = absl::StrSplit(output, '\n');
  ASSERT_EQ(output_lines.size(), 6);
  EXPECT_THAT(output_lines[0], testing::HasSubstr("some_file.x:5-7"));
  EXPECT_EQ(output_lines[1], "  0005: line 5");
  EXPECT_EQ(output_lines[2], "* 0006: line 6");
  EXPECT_EQ(output_lines[3], "  ~~~~~~^--^ my error message");
  EXPECT_EQ(output_lines[4], "  0007: line 7");
  // Because of trailing newline after displaying line 7.
  EXPECT_EQ(output_lines[5], "");
}

}  // namespace
}  // namespace xls::dslx
