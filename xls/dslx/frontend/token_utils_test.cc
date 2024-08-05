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

#include "xls/dslx/frontend/token_utils.h"

#include <string>

#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(IsScreamingSnakeCaseTest, Samples) {
  EXPECT_TRUE(IsScreamingSnakeCase("C"));
  EXPECT_FALSE(IsScreamingSnakeCase("c"));

  // Tick marks on identifiers shouldn't change the assessment.
  EXPECT_TRUE(IsScreamingSnakeCase("C'"));

  // Exclam mark on identifiers shouldn't change the assessment.
  EXPECT_TRUE(IsScreamingSnakeCase("C!"));

  EXPECT_TRUE(IsScreamingSnakeCase("C_WITH_UNDERSCORES"));
}

TEST(EscapeTest, EscapeNullCEscape) {
  EXPECT_EQ(Escape(std::string("\0", 1)), "\\0");
  EXPECT_EQ(Escape(std::string("a\0b\0\n", 5)), "a\\0b\\0\\n");
}

TEST(EscapeTest, EscapeNullCHexEscapeAsCanonical) {
  EXPECT_EQ(Escape(std::string("\x00", 1)), "\\0");
  EXPECT_EQ(Escape(std::string("a\x00"
                               "b\x00"
                               "\n",
                               5)),
            "a\\0b\\0\\n");
}

TEST(EscapeTest, EscapeDoNotSubAlreadyEscapedCHexEscape) {
  EXPECT_EQ(Escape("\\x00"), "\\\\x00");
  EXPECT_EQ(Escape("a\\x00b\\x00\n"), "a\\\\x00b\\\\x00\\n");
}

}  // namespace
}  // namespace xls::dslx
