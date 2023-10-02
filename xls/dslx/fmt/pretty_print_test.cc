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

#include "xls/dslx/fmt/pretty_print.h"

#include <string_view>

#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(PrettyPrintTest, OneTextDoc) {
  DocArena arena;
  DocRef ref = arena.MakeText("hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 2), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 3), "hi");
}

TEST(PrettyPrintTest, OneHardLineDoc) {
  DocArena arena;
  DocRef ref = arena.hard_line();
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "\n");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "\n");
}

TEST(PrettyPrintTest, TwoTextDoc) {
  DocArena arena;
  DocRef ref0 = arena.MakeText("hi");
  DocRef ref1 = arena.MakeText("there");
  DocRef concat = arena.MakeConcat(ref0, ref1);
  EXPECT_EQ(PrettyPrint(arena, concat, 0), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 1), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 7), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 8), "hithere");
}

TEST(PrettyPrintTest, LetExample) {
  DocArena arena;
  DocRef let = arena.MakeText("let");
  DocRef break1 = arena.break1();
  DocRef x_colon = arena.MakeText("x:");
  DocRef u32 = arena.MakeText("u32");
  DocRef equals = arena.MakeText("=");
  DocRef u32_42 = arena.MakeText("u32:42");

  DocRef doc = arena.MakeGroup(
      ConcatN(arena, let,
              {break1, x_colon, break1, u32, break1, equals, break1, u32_42}));
  const std::string_view kWantBreak = R"(let
x:
u32
=
u32:42)";
  // Up to 18 chars we get the "break" mode version.
  EXPECT_EQ(PrettyPrint(arena, doc, 0), kWantBreak);
  EXPECT_EQ(PrettyPrint(arena, doc, 18), kWantBreak);

  // At 19 chars we get the "flat" mode version.
  const std::string_view kWantFlat = "let x: u32 = u32:42";
  EXPECT_EQ(kWantFlat.size(), 19);
  EXPECT_EQ(PrettyPrint(arena, doc, 19), kWantFlat);
}

TEST(PrettyPrintTest, CallExample) {
  DocArena arena;
  DocRef call_with_args = ConcatN(
      arena, arena.MakeText("foo("),
      {arena.MakeNest(ConcatN(
           arena, arena.break0(),
           {arena.MakeText("bar,"), arena.break1(), arena.MakeText("bat")})),
       arena.break0(), arena.MakeText(")")});

  DocRef doc = arena.MakeGroup(call_with_args);
  const std::string_view kWantBreak = R"(foo(
    bar,
    bat
))";
  EXPECT_EQ(PrettyPrint(arena, doc, 0), kWantBreak);
}

}  // namespace
}  // namespace xls::dslx
