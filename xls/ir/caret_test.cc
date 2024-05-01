// Copyright 2022 The XLS Authors
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

#include "xls/ir/caret.h"

#include <functional>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

std::function<std::optional<std::string>(Fileno)> LookUpInPackage(Package* p) {
  return [=](Fileno file_number) { return p->GetFilename(file_number); };
}

TEST(CaretTest, Simple) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(1), Colno(20));

  std::string expected = R"( --> /foo/bar/baz.ir:1:20
  |
1 | this is a line of code that does stuff
  |                    ^
)";
  EXPECT_EQ(
      PrintCaret(LookUpInPackage(&p), loc,
                 "this is a line of code that does stuff", std::nullopt, 60),
      expected);
}

TEST(CaretTest, FileContent) {
  Package p("example");
  constexpr const char* kProgram = R"(int main(void) {
  this is a line of code that does stuff
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "baz.ir"));
  SourceLocation loc =
      p.AddSourceLocation(std::string(temp_file.path()), Lineno(2), Colno(3));

  constexpr const char* expected_format = R"( --> %s:2:3
  |
2 |   this is a line of code that does stuff
  |   ^
)";
  std::string expected = absl::StrFormat(expected_format,
                                         std::string(temp_file.path()));
  EXPECT_EQ(
      PrintCaret(LookUpInPackage(&p), loc, std::nullopt, std::nullopt, 60),
      expected);
}

TEST(CaretTest, LongLineNumber) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(123123), Colno(20));

  std::string expected = R"(      --> /foo/bar/baz.ir:123123:20
       |
123123 | this is a line of code that does stuff
       |                    ^
)";
  EXPECT_EQ(
      PrintCaret(LookUpInPackage(&p), loc,
                 "this is a line of code that does stuff", std::nullopt, 60),
      expected);
}

TEST(CaretTest, WithComment) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(123), Colno(20));

  std::string expected = R"(   --> /foo/bar/baz.ir:123:20
    |
123 | this is a line of code that does stuff
    |                    ^
    |                    |
    |                    this is a comment
)";
  EXPECT_EQ(PrintCaret(LookUpInPackage(&p), loc,
                       "this is a line of code that does stuff",
                       "this is a comment", 60),
            expected);
}

TEST(CaretTest, OverhangingLine) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(123), Colno(20));

  std::string expected = R"(   --> /foo/bar/baz.ir:123:20
    |
123 | this is a really really long line of code that does s…
    |                    ^
)";
  EXPECT_EQ(
      PrintCaret(LookUpInPackage(&p), loc,
                 "this is a really really long line of code that does stuff",
                 std::nullopt, 60),
      expected);
}

TEST(CaretTest, OverhangingComment) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(123), Colno(20));

  std::string expected = R"(   --> /foo/bar/baz.ir:123:20
    |
123 | this is a line of code that does stuff
    |                    ^
    |                    |
    |                    this is a really really quite
    |                    long comment
)";
  EXPECT_EQ(PrintCaret(LookUpInPackage(&p), loc,
                       "this is a line of code that does stuff",
                       "this is a really really quite long comment", 50),
            expected);
}

TEST(CaretTest, UnknownLine) {
  Package p("example");
  SourceLocation loc =
      p.AddSourceLocation("/foo/bar/baz.ir", Lineno(123), Colno(20));

  std::string expected = R"(   --> /foo/bar/baz.ir:123:20
    |
123 | «unknown line contents»
    |                    ^
)";
  EXPECT_EQ(PrintCaret(LookUpInPackage(&p), loc), expected);
}

}  // namespace
}  // namespace xls
