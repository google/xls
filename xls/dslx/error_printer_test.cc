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

#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/pos.h"

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
                                     /*color=*/PositionalErrorColor::kNoColor,
                                     /*error_context_line_count=*/3));
  std::string output = ss.str();
  // Note: we split lines and compare instead of doing a full string comparison
  // because the leader is a tempfile path, seems a bit clearer than regex this
  // way.
  std::vector<std::string_view> output_lines = absl::StrSplit(output, '\n');
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
                                     /*color=*/PositionalErrorColor::kNoColor,
                                     /*error_context_line_count=*/3));
  std::string output = ss.str();
  // Note: we split lines and compare instead of doing a full string comparison
  // because the leader is a tempfile path, seems a bit clearer than regex this
  // way.
  std::vector<std::string_view> output_lines = absl::StrSplit(output, '\n');
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

TEST(PrintPositionalErrorTest, ZeroLineErrorTest) {
  const std::string filename = "zero_lines.x";
  std::function<absl::StatusOr<std::string>(std::string_view)>
      get_file_contents =
          [&](std::string_view file) -> absl::StatusOr<std::string> {
    return std::string("");
  };

  // The first nonexistent character in the file.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 0, 0}, Pos{filename, 0, 0}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "zero_lines.x:1:1-1:1\n"
              "0001: \n"
              "~~~~~~^ here\n");
  }
}

TEST(PrintPositionalErrorTest, OneLineErrorTest) {
  const std::string filename = "one_line.x";
  std::function<absl::StatusOr<std::string>(std::string_view)>
      get_file_contents =
          [&](std::string_view file) -> absl::StatusOr<std::string> {
    return std::string("x\n");
  };

  // The single character in the file.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 0, 0}, Pos{filename, 0, 1}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "one_line.x:1:1-1:2\n"
              "0001: x\n"
              "~~~~~~^ here\n");
  }

  // The single character in the file and the nonexistent
  // next character (the location of an unexpected EOF, e.g.)
  // The extreme gets capped, and the result looks identical to
  // the previous run.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 0, 0}, Pos{filename, 1, 0}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "one_line.x:1:1-2:1\n"
              "0001: x\n"
              "~~~~~~^ here\n");
  }

  // Only the (zero-length) start of the nonexistent first character of the
  // nonexistent second line. This is what it might look like when something
  // expected is missing.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 1, 0}, Pos{filename, 1, 0}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "one_line.x:2:1-2:1\n"
              "0001: x\n"
              "~~~~~~~^ here\n");  // One step beyond.
  }
}

TEST(PrintPositionalErrorTest, TwoLineErrorTest) {
  std::function<absl::StatusOr<std::string>(std::string_view)>
      get_file_contents =
          [&](std::string_view file) -> absl::StatusOr<std::string> {
    return std::string("x\nyz\n");
  };
  const std::string filename = "two_line.x";

  // Normal behavior for a single-character error.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 0, 0}, Pos{filename, 0, 1}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "two_line.x:1:1-1:2\n"
              "0001: x\n"
              "~~~~~~^ here\n"
              "0002: yz\n");
  }

  // This marked range ends just before the first character of a line.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 0, 0}, Pos{filename, 1, 0}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));

    // The last line is weird, but so is the limit coordinate.
    EXPECT_EQ(ss.str(),
              "two_line.x:1:1-2:1\n"
              "0001:   x\n"
              "       _^\n"
              "0002: | yz\n"
              "      |^ here\n");
  }

  // Marked range at the end of the file.
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 1, 0}, Pos{filename, 1, 2}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));
    EXPECT_EQ(ss.str(),
              "two_line.x:2:1-2:3\n"
              "0001: x\n"
              "0002: yz\n"
              "~~~~~~^^ here\n");
  }

  // Overruns
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, 1, 1}, Pos{filename, 1, 999}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));

    // The error range is widely overshooting the actual output,
    // and is capped.
    EXPECT_EQ(ss.str(),
              "two_line.x:2:2-2:1000\n"
              "0001: x\n"
              "0002: yz\n"
              "~~~~~~~^ here\n");
  }

  // Under- and overruns
  {
    std::stringstream ss;
    XLS_ASSERT_OK(PrintPositionalError(
        Span(Pos{filename, -99, -17}, Pos{filename, 88, 999}), "here", ss,
        get_file_contents,
        /*color=*/PositionalErrorColor::kNoColor,
        /*error_context_line_count=*/3));

    // The error range is widely overshooting the actual output,
    // and is capped.
    EXPECT_EQ(ss.str(),
              "two_line.x:-98:-16-89:1000\n"
              "0001:   x\n"
              "       _^\n"
              "0002: | yz\n"
              "      |__^ here\n");
  }

  // Underruns, overruns, and in the wrong order!
  {
    std::stringstream ss;
    XLS_ASSERT_OK(
        PrintPositionalError(Span(Pos{filename, -88, 5}, Pos{filename, -77, 3}),
                             "here", ss, get_file_contents,
                             /*color=*/PositionalErrorColor::kNoColor,
                             /*error_context_line_count=*/3));

    EXPECT_EQ(ss.str(),
              "two_line.x:-87:6--76:4\n"
              "0001: x\n"
              "~~~~~~^ here\n"
              "0002: yz\n");
  }
}

TEST(PrintPositionalErrorTest, FiveLineErrorFuzzTest) {
  std::function<absl::StatusOr<std::string>(std::string_view)>
      get_file_contents =
          [&](std::string_view file) -> absl::StatusOr<std::string> {
    return std::string(
        "abcde\n"
        "fghij\n"
        "klmno\n"
        "pqrst\n"
        "uvwxyz\n");
  };
  const std::string filename = "five_line.x";

  // Whatever bogus coordinates you throw at the printer,
  // it does ... something!
  for (int from_line = -2; from_line <= 6; ++from_line) {
    for (int from_col = -1; from_col <= 7; ++from_col) {
      const Pos from_pos(filename, from_line, from_col);
      for (int to_line = from_line; to_line <= 6; ++to_line) {
        for (int to_col = (to_line == from_line ? from_col : -1); to_col <= 7;
             ++to_col) {
          std::stringstream ss;
          const Pos to_pos(filename, to_line, to_col);
          const Span span(from_pos, to_pos);
          XLS_ASSERT_OK(PrintPositionalError(
              Span(from_pos, to_pos), "here", ss, get_file_contents,
              /*color=*/PositionalErrorColor::kNoColor,
              /*error_context_line_count=*/3));
          EXPECT_THAT(ss.str(), testing::Not(testing::IsEmpty()));
        }
      }
    }
  }
}

}  // namespace
}  // namespace xls::dslx
