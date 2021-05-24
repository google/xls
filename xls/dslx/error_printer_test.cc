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
                                     /*get_file_contents=*/nullptr,
                                     /*color=*/false,
                                     /*error_context_line_count=*/3));
  std::string output = ss.str();
  // Note: we split lines and compare instead of doing a full string comparison
  // because the leader is a tempfile path, seems a bit clearer than regex this
  // way.
  std::vector<absl::string_view> output_lines = absl::StrSplit(output, '\n');
  ASSERT_EQ(output_lines.size(), 6);
  EXPECT_THAT(output_lines[0], testing::HasSubstr("some_file.x:6:1-6:5"));
  EXPECT_EQ(output_lines[1], "0005: line 5");
  EXPECT_EQ(output_lines[2], "0006: line 6");
  EXPECT_EQ(output_lines[3], "~~~~~~^--^ my error message");
  EXPECT_EQ(output_lines[4], "0007: line 7");
  // Because of trailing newline after displaying line 7.
  EXPECT_EQ(output_lines[5], "");
}

TEST(PrintPositionalErrorTest, MultiLineErrorTest) {
  XLS_ASSERT_OK_AND_ASSIGN(TempFile temp, TempFile::CreateWithContent(
                                              R"(fn f(x: u32) -> u32 {
  match x {
    u32:42 => x
  }
})",
                                              "some_file.x"));

  const std::string filename = temp.path();
  const Pos start_pos(filename, 1, 10);
  // Note: colno 2 is where the character lives, but colno 3 is where the limit
  // of the span is.
  const Pos limit_pos(filename, 3, 3);
  const Span error_span(start_pos, limit_pos);
  std::stringstream ss;
  XLS_ASSERT_OK(PrintPositionalError(error_span, "match not exhaustive", ss,
                                     /*get_file_contents=*/nullptr,
                                     /*color=*/false,
                                     /*error_context_line_count=*/3));
  std::string output = ss.str();
  // Note: we split lines and compare instead of doing a full string comparison
  // because the leader is a tempfile path, seems a bit clearer than regex this
  // way.
  std::vector<absl::string_view> output_lines = absl::StrSplit(output, '\n');
  ASSERT_EQ(output_lines.size(), 9);
  EXPECT_THAT(output_lines[0], testing::HasSubstr("some_file.x:2:11-4:4"));
  EXPECT_EQ(output_lines[1], "0001:   fn f(x: u32) -> u32 {");
  EXPECT_EQ(output_lines[2], "0002:     match x {");
  EXPECT_EQ(output_lines[3], "       ___________^");
  EXPECT_EQ(output_lines[4], "0003: |     u32:42 => x");
  EXPECT_EQ(output_lines[5], "0004: |   }");
  EXPECT_EQ(output_lines[6], "      |___^ match not exhaustive");
  EXPECT_EQ(output_lines[7], "0005:   }");
  // Because of trailing newline after displaying line 5.
  EXPECT_EQ(output_lines[8], "");
}

}  // namespace
}  // namespace xls::dslx
