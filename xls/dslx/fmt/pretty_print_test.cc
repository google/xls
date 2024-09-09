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
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.MakeText("hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 2), "hi");
  EXPECT_EQ(PrettyPrint(arena, ref, 3), "hi");

  EXPECT_EQ(arena.ToDebugString(ref), "Doc{2, \"hi\"}");
}

TEST(PrettyPrintTest, EmptyConcatIsEmptyString) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = ConcatN(arena, {});
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "");

  EXPECT_EQ(arena.ToDebugString(ref), "Doc{0, \"\"}");
}

TEST(PrettyPrintTest, ConcatOfEmpty) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.MakeConcat(arena.empty(), arena.empty());
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "");

  EXPECT_EQ(arena.ToDebugString(ref),
            "Doc{0, Concat{Doc{0, \"\"}, Doc{0, \"\"}}}");
}

TEST(PrettyPrintTest, OneHardLineDoc) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.hard_line();
  EXPECT_EQ(PrettyPrint(arena, ref, 0), "\n");
  EXPECT_EQ(PrettyPrint(arena, ref, 1), "\n");
}

TEST(PrettyPrintTest, TwoTextDoc) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref0 = arena.MakeText("hi");
  DocRef ref1 = arena.MakeText("there");
  DocRef concat = arena.MakeConcat(ref0, ref1);
  EXPECT_EQ(PrettyPrint(arena, concat, 0), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 1), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 7), "hithere");
  EXPECT_EQ(PrettyPrint(arena, concat, 8), "hithere");
}

TEST(PrettyPrintTest, LetExample) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef let = arena.MakeText("let");
  DocRef break1 = arena.break1();
  DocRef x_colon = arena.MakeText("x:");
  DocRef u32 = arena.MakeText("u32");
  DocRef equals = arena.MakeText("=");
  DocRef u32_42 = arena.MakeText("u32:42");

  DocRef doc =
      arena.MakeGroup(ConcatN(arena, {let, break1, x_colon, break1, u32, break1,
                                      equals, break1, u32_42}));
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
  FileTable file_table;
  DocArena arena(file_table);
  DocRef call_with_args = ConcatN(
      arena,
      {arena.MakeText("foo("),
       arena.MakeNest(ConcatN(arena, {arena.break0(), arena.MakeText("bar,"),
                                      arena.break1(), arena.MakeText("bat")})),
       arena.break0(), arena.MakeText(")")});

  DocRef doc = arena.MakeGroup(call_with_args);
  const std::string_view kWantBreak = R"(foo(
    bar,
    bat
))";
  EXPECT_EQ(PrettyPrint(arena, doc, 0), kWantBreak);
}

TEST(PrettyPrintTest, PrefixedReflow) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.MakePrefixedReflow(
      "// ", "this is overly long for our liking, gladly");
  EXPECT_EQ(PrettyPrint(arena, ref, 18), R"(// this is overly
// long for our
// liking, gladly)");
}

TEST(PrettyPrintTest, PrefixedReflowAfterIndentWidth18) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref0 = arena.MakePrefixedReflow(
      "// ", "this is overly long for our liking, gladly");
  DocRef ref1 = arena.MakeNest(arena.MakePrefixedReflow(
      "// ", "yet another pleasingly long line, indented"));

  DocRef ref = ConcatN(arena, {ref0, arena.hard_line(), ref1});

  EXPECT_EQ(PrettyPrint(arena, ref, 18), R"(// this is overly
// long for our
// liking, gladly
    // yet another
    // pleasingly
    // long line,
    // indented)");
}

TEST(PrettyPrintTest, PrefixedReflowAfterIndentWidth40) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref0 = arena.MakePrefixedReflow(
      "// ", "this is overly long for our liking, gladly");
  DocRef ref1 = arena.MakeNest(arena.MakePrefixedReflow(
      "// ", "yet another pleasingly long line, indented"));

  DocRef ref = ConcatN(arena, {ref0, arena.hard_line(), ref1});

  EXPECT_EQ(PrettyPrint(arena, ref, 40),
            R"(// this is overly long for our liking,
// gladly
    // yet another pleasingly long line,
    // indented)");
}

// Demonstrates that the empty line between nested elements don't have leading
// spaces.
TEST(PrettyPrintTest, NestLevelWithEmptyLine) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref0 = arena.MakeText("foo");
  DocRef ref1 = arena.MakeText("bar");

  DocRef ref = arena.MakeNest(
      ConcatN(arena, {ref0, arena.hard_line(), arena.hard_line(), ref1}));

  // Note: the second line here has no leading spaces.
  EXPECT_EQ(PrettyPrint(arena, ref, 100), R"(    foo

    bar)");
}

TEST(PrettyPrintTest, PrefixedReflowOneOverflongToken) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.MakePrefixedReflow(
      "//",
      " abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrs"
      "tuvwxyzabcdefghijklmnopqrstuvwxyz");
  EXPECT_EQ(
      PrettyPrint(arena, ref, 40),
      R"(// abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz)");
}

TEST(PrettyPrintTest, PrefixedReflowCustomSpacingBeforeToken) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef ref = arena.MakePrefixedReflow("//", "     I like this many spaces");
  EXPECT_EQ(PrettyPrint(arena, ref, 40), R"(//     I like this many spaces)");
}

// Scenario where we use NestIfFlatFits and the "on_other_ref" DOES NOT fits
// inline into the current line so we emit the "on_nested_flat_ref" into the
// subsequent line (it does fit there).
TEST(PrettyPrintTest, NestIfFlatFitsDoNest) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef six_nums = arena.MakeText("123456");
  DocRef six_chars = arena.MakeText("abcdef");
  DocRef ref = arena.MakeConcat(
      six_nums, arena.MakeNestIfFlatFits(/*on_nested_flat_ref=*/six_nums,
                                         /*on_other_ref=*/six_chars));

  EXPECT_EQ(PrettyPrint(arena, ref, 10), R"(123456
    123456)");
}

// Scenario where we use NestIfFlatFits and the "on_other_ref" fits inline into
// the current line so we don't need to emit nested.
TEST(PrettyPrintTest, NestIfFlatFitsDoFlatInline) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef six_nums = arena.MakeText("123456");
  DocRef four_chars = arena.MakeConcat(arena.break0(), arena.MakeText("abcd"));
  DocRef ref = arena.MakeConcat(
      six_nums, arena.MakeNestIfFlatFits(/*on_nested_flat_ref=*/four_chars,
                                         /*on_other_ref=*/four_chars));

  EXPECT_EQ(PrettyPrint(arena, ref, 10), "123456abcd");
}

// Scenario where we use NestIfFlatFits but we fall back to the "break mode"
// case; i.e. the on_other_ref is selected and emitted in break mode.
TEST(PrettyPrintTest, NestIfFlatFitsDoFlatBreak) {
  FileTable file_table;
  DocArena arena(file_table);
  DocRef six_nums = arena.MakeText("123456");
  DocRef seven_chars =
      arena.MakeConcat(arena.break0(), arena.MakeText("abcdefg"));
  EXPECT_EQ(arena.Deref(seven_chars).flat_requirement,
            pprint_internal::Requirement{7});

  DocRef ref = arena.MakeConcat(
      six_nums, arena.MakeNestIfFlatFits(/*on_nested_flat_ref=*/seven_chars,
                                         /*on_other_ref=*/seven_chars));

  EXPECT_EQ(PrettyPrint(arena, ref, 10), R"(123456
abcdefg)");
}

}  // namespace
}  // namespace xls::dslx
