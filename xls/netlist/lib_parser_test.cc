// Copyright 2020 The XLS Authors
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

#include "xls/netlist/lib_parser.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace netlist {
namespace cell_lib {
namespace {

TEST(LibParserTest, ScanSimple) {
  std::string text = "{}()";
  XLS_ASSERT_OK_AND_ASSIGN(auto cs, CharStream::FromText(text));
  Scanner scanner(&cs);
  EXPECT_EQ(scanner.Peek().value()->kind(), TokenKind::kOpenCurl);
  EXPECT_EQ(scanner.Peek().value()->kind(), TokenKind::kOpenCurl);
  EXPECT_EQ(scanner.Pop().value().kind(), TokenKind::kOpenCurl);
  EXPECT_EQ(scanner.Peek().value()->kind(), TokenKind::kCloseCurl);
  EXPECT_EQ(scanner.Pop().value().kind(), TokenKind::kCloseCurl);

  EXPECT_EQ(scanner.Pop().value().kind(), TokenKind::kOpenParen);
  EXPECT_EQ(scanner.Pop().value().kind(), TokenKind::kCloseParen);
  EXPECT_TRUE(scanner.AtEof());
}

// Helper that parses the given text as a library block and returns the
// block structure.
absl::StatusOr<std::unique_ptr<Block>> Parse(
    std::string text,
    std::optional<absl::flat_hash_set<std::string>> allowlist =
        std::nullopt) {
  XLS_ASSIGN_OR_RETURN(auto cs, CharStream::FromText(text));
  Scanner scanner(&cs);
  Parser parser(&scanner, std::move(allowlist));
  return parser.ParseLibrary();
}

absl::StatusOr<std::string> ParseToString(std::string text) {
  XLS_ASSIGN_OR_RETURN(auto block, Parse(text));
  return block->ToString();
}

TEST(LibParserTest, EmptyLibrary) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string parsed,
                           ParseToString("library (foo) {}"));
  EXPECT_EQ(parsed, "(block library (foo) ())");
}

TEST(LibParserTest, EmptyLibraryWithComment) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string parsed,
      ParseToString("library (foo) {/* this is a comment */}"));
  EXPECT_EQ(parsed, "(block library (foo) ())");
}

TEST(LibParserTest, And2CellWithPins) {
  std::string text = R"(
library (foo) {
  cell (AND2) {
    pin (a);
    pin (b);
    pin (o) {
      function: "a&b";
    }
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string parsed, ParseToString(text));
  EXPECT_EQ(parsed,
            "(block library (foo) ((block cell (AND2) ((block pin (a) ()) "
            "(block pin (b) ()) (block pin (o) ((function \"a&b\")))))))");
  EXPECT_EQ(1, Parse(text).value()->CountEntries("cell"));
}

TEST(LibParserTest, KeyValues) {
  std::string text = R"(
library (foo) {
  some_number: 1.234;
  tiny_pi: 3.1415926535e-10;
  big_pi: 3.14e10;
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string parsed, ParseToString(text));
  EXPECT_EQ(parsed,
            "(block library (foo) ((some_number \"1.234\") (tiny_pi "
            "\"3.1415926535e-10\") (big_pi \"3.14e10\")))");
}

TEST(LibParserTest, KeyValuesNoSemis) {
  std::string text = R"(
library (foo) {
  some_number: 1.234
  tiny_pi: 3.1415926535e-10
  big_pi: 3.14e10
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string parsed, ParseToString(text));
  EXPECT_EQ(parsed,
            "(block library (foo) ((some_number \"1.234\") (tiny_pi "
            "\"3.1415926535e-10\") (big_pi \"3.14e10\")))");
  EXPECT_EQ("1.234", Parse(text).value()->GetKVOrDie("some_number"));
}

TEST(LibParserTest, SubBlockMultiValue) {
  std::string text = R"(
library (foo) {
  my_block (1.234, "string", some_ident) ;
  my_block ();
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string parsed, ParseToString(text));
  EXPECT_EQ(parsed,
            "(block library (foo) ((block my_block (1.234 string some_ident) "
            "()) (block my_block () ())))");
  EXPECT_EQ(2, Parse(text).value()->CountEntries("my_block"));
  EXPECT_EQ(2, Parse(text).value()->GetSubBlocks("my_block").size());
}

TEST(LibParserTest, AllowlistKind) {
  std::string text = R"(
library (foo) {
  foo () {
    foo_key: foo_value;
  }
  bar () {
    bar_key: bar_value;
  }
  baz () {
    baz_key: baz_value;
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Block> library,
      Parse(text, absl::flat_hash_set<std::string>{"library", "bar"}));
  EXPECT_EQ(library->ToString(),
            "(block library (foo) ("
            "(block foo () ()) "
            "(block bar () ((bar_key \"bar_value\"))) "
            "(block baz () ())"
            "))");
}

}  // namespace
}  // namespace cell_lib
}  // namespace netlist
}  // namespace xls
