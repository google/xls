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

#include "xls/dslx/frontend/pos.h"

#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

TEST(PosTest, PosStringRoundTrip) {
  std::string text = "/my/foo.x:1:2";
  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(Pos p, Pos::FromString(text, file_table));
  EXPECT_EQ(p.lineno(), 0);
  EXPECT_EQ(p.colno(), 1);
  EXPECT_EQ(p.GetFilename(file_table), "/my/foo.x");
  EXPECT_EQ(text, p.ToString(file_table));
}

TEST(PosTest, PosLt) {
  const Fileno fileno = Fileno(0);
  EXPECT_LT(Pos(fileno, 0, 0), Pos(fileno, 0, 1));
  EXPECT_LT(Pos(fileno, 0, 0), Pos(fileno, 1, 0));
  EXPECT_GE(Pos(fileno, 0, 0), Pos(fileno, 0, 0));
}

TEST(SpanTest, SpanContainsOther) {
  const Fileno fileno = Fileno(0);
  const Pos origin = Pos(fileno, 0, 0);
  Span line0_col0 = Span(origin, Pos(fileno, 0, 1));
  Span line0_col0to5 = Span(origin, Pos(fileno, 0, 5));
  Span line0_col1to5 = Span(Pos(fileno, 0, 1), Pos(fileno, 0, 5));
  Span line0to1_col0 = Span(origin, Pos(fileno, 1, 0));

  EXPECT_TRUE(line0_col0to5.Contains(line0_col0));
  EXPECT_FALSE(line0_col0.Contains(line0_col0to5));

  EXPECT_TRUE(line0_col0to5.Contains(line0_col1to5));
  EXPECT_FALSE(line0_col1to5.Contains(line0_col0to5));

  EXPECT_TRUE(line0to1_col0.Contains(line0_col0));
  EXPECT_TRUE(line0to1_col0.Contains(line0_col0to5));

  auto contains_self = [](const Span& s) { EXPECT_TRUE(s.Contains(s)); };
  contains_self(line0_col0);
  contains_self(line0_col0to5);
  contains_self(line0to1_col0);
}

}  // namespace
}  // namespace xls::dslx
