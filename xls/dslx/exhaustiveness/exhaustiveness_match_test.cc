// Copyright 2025 The XLS Authors
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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/exhaustiveness/match_exhaustiveness_checker.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::vector<const NameDefTree*> GetPatterns(const Match& match) {
  std::vector<const NameDefTree*> patterns;
  for (const MatchArm* arm : match.arms()) {
    for (const NameDefTree* pattern : arm->patterns()) {
      patterns.push_back(pattern);
    }
  }
  return patterns;
}

void CheckExhaustiveOnlyAfterLastPattern(std::string_view program) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  std::optional<Function*> func = tm.module->GetFunction("main");
  ASSERT_TRUE(func.has_value());
  StatementBlock* body = func.value()->body();
  const Statement& statement = *body->statements().back();
  Match* match = dynamic_cast<Match*>(std::get<Expr*>(statement.wrapped()));
  ASSERT_TRUE(match != nullptr);

  std::optional<Type*> matched_type = tm.type_info->GetItem(match->matched());
  ASSERT_TRUE(matched_type.has_value());
  ASSERT_NE(matched_type.value(), nullptr);

  MatchExhaustivenessChecker checker(match->matched()->span(), import_data,
                                     *tm.type_info, *matched_type.value());

  std::vector<const NameDefTree*> patterns = GetPatterns(*match);
  for (int64_t i = 0; i < patterns.size(); ++i) {
    bool now_exhaustive = checker.AddPattern(*patterns[i]);
    // We expect it to become exhaustive with the last match arm.
    bool expect_now_exhaustive = i + 1 == patterns.size();
    EXPECT_EQ(now_exhaustive, expect_now_exhaustive)
        << "Expected match to be "
        << (expect_now_exhaustive ? "exhaustive" : "non-exhaustive")
        << " after adding pattern `" << patterns[i]->ToString() << "`";
  }
}

void CheckNonExhaustive(std::string_view program) {
  ImportData import_data = CreateImportDataForTest();
  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(program, "test.x", "test", &import_data);
  EXPECT_THAT(tm.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Match patterns are not exhaustive")));
}

void CheckExhaustiveWithRedundantPattern(std::string_view program) {
  WarningKindSet warnings =
      DisableWarning(kAllWarningsSet, WarningKind::kUnusedDefinition);
  ImportData import_data = CreateImportDataForTest(nullptr, warnings);
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test", &import_data));
  absl::Span<const WarningCollector::Entry> collected_warnings =
      tm.warnings.warnings();
  ASSERT_FALSE(collected_warnings.empty());
  for (const WarningCollector::Entry& warning : collected_warnings) {
    EXPECT_THAT(
        warning.message,
        HasSubstr(kDefaultTypeInferenceVersion ==
                          TypeInferenceVersion::kVersion2
                      ? "`match` is already exhaustive before this pattern"
                      : "Match is already exhaustive before this pattern"));
  }
}

TEST(ExhaustivenessMatchTest, MatchBoolTrueFalse) {
  constexpr std::string_view kMatch = R"(fn main(x: bool) -> u32 {
    match x {
        false => u32:42,
        true => u32:64,
    }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchBoolJustTrue) {
  constexpr std::string_view kMatch = R"(fn main(x: bool) -> u32 {
    match x {
        true => u32:42,
        _ => u32:64,
    }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchOneHotsInTuple) {
  constexpr std::string_view kMatch = R"(fn main(t: (bool, bool, bool)) -> u32 {
    match t {
        (true, _, _) => u32:42,
        (_, true, _) => u32:64,
        (_, _, true) => u32:86,
        _ => u32:0,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, SammpleFromStdlib) {
  constexpr std::string_view kMatch = R"(fn main(x: u32, y: u32) -> u32 {
    match (x[-1:], y[-1:]) {
        (u1:1, u1:1) => u32:42,
        (u1:1, u1:0) => u32:64,
        _ => u32:0,
    }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchRestOfTuple) {
  constexpr std::string_view kMatch = R"(fn main(t: (bool, bool, bool)) -> u32 {
    match t {
        (true, ..) => u32:42,
        (_, true, ..) => u32:64,
        (.., true) => u32:128,
        (false, false, false) => u32:0,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchNestedTuple) {
  // Test a nested tuple match pattern.
  constexpr std::string_view kMatch =
      R"(fn main(x: (bool, (bool, bool))) -> u32 {
    match x {
      (true, (true, _)) => u32:1,
      (false, (_, true)) => u32:2,
      (_, (_, _)) => u32:0,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchRedundantPattern) {
  // Even if one of the arms is redundant, the overall match is only complete
  // once a catch-all branch is reached.
  constexpr std::string_view kMatch = R"(fn main(t: (bool, bool)) -> u32 {
    match t {
      (true, _) => u32:1,
      (true, false) => u32:2,
      _ => u32:0,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

// Dense enum values but missing the top value in the underlying type.
TEST(ExhaustivenessMatchTest, MatchOnDenseZeroAlignedEnum) {
  constexpr std::string_view kMatch = R"(enum E : u2 {
    A = 0,
    B = 1,
    C = 2,
  }
  fn main(e: E) -> u32 {
    match e {
      E::A => u32:42,
      E::B => u32:64,
      E::C => u32:86,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

// Note that the value 1 is a gap in the enum name space.
TEST(ExhaustivenessMatchTest, MatchOnSparseEnum) {
  constexpr std::string_view kMatch = R"(enum E : u2 {
    A = 0,
    B = 2,
    C = 3,
  }
  fn main(e: E) -> u32 {
    match e {
      E::A => u32:42,
      E::B => u32:64,
      E::C => u32:86,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchWithNestedTuplesAndRestOfTupleSprinkled) {
  constexpr std::string_view kMatch =
      R"(fn main(t: (u32, (u32, u32, u32), u32, u32)) -> u32 {
    match t {
      (.., u32:1, a) => a,
      (u32:2, (u32:0, _, b), ..) => b,
      (u32:3, (.., c), ..) => c,
      (u32:4, d, ..) => d.0,
      _ => u32:0xdeadbeef,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, EmptyRestOfTupleInCenter) {
  constexpr std::string_view kMatch = R"(fn main(x: (u32, u33)) -> u32 {
  match x {
    (y, .., z) => y,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, RestOfTupleAsWildcard) {
  constexpr std::string_view kMatch = R"(fn main(t: (u32, u32)) -> u32 {
  match t {
    (u32:1, .., a) => a,
    (..) => u32:1
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, SignedNegativeAndPositiveValueRanges) {
  constexpr std::string_view kMatch = R"(fn main(x: s8) -> u32 {
  match x {
    s8:0..s8:127 => u32:42,
    s8:-128..s8:0 => u32:64,
    s8:127 => u32:128,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchWithRangeHavingStartGtLimit) {
  constexpr std::string_view kMatch = R"(fn main(x: u32) -> u32 {
  match x {
    u32:7..u32:0 => u32:42,
    _ => u32:0,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

// Range with a "zero-volume interval" / empty range.
TEST(ExhaustivenessMatchTest, MatchWithRangeHavingStartEqLimit) {
  constexpr std::string_view kMatch = R"(fn main(x: u32) -> u32 {
  match x {
    u32:7..u32:7 => u32:42,
    _ => u32:0,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, ExhaustiveU2) {
  constexpr std::string_view kMatch = R"(fn main(x: u2) -> u32 {
  match x {
    u2:0 => u32:42,
    u2:1 => u32:64,
    u2:2 => u32:128,
    u2:3 => u32:256,
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, NonExhaustiveU2) {
  constexpr std::string_view kMatch = R"(fn main(x: u2) -> u32 {
  match x {
    u2:0 => u32:42,
    u2:1 => u32:64,
    u2:2 => u32:128,
  }
})";
  CheckNonExhaustive(kMatch);
}

TEST(ExhaustivenessMatchTest, NonExhaustiveTuple) {
  constexpr std::string_view kMatch = R"(fn main(x: (u2, u2)) -> u32 {
  match x {
    (u2:0, _) => u32:42,
    (u2:1, u2:0) => u32:64,
  }
})";
  CheckNonExhaustive(kMatch);
}

TEST(ExhaustivenessMatchTest, RedundantPattern) {
  constexpr std::string_view kMatch = R"(fn main(x: u4) -> u32 {
  match x {
    _ => u32:0,
    u4:5 => u32:1,
  }
})";
  CheckExhaustiveWithRedundantPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, ComplexOverlapAndFragmentation) {
  constexpr std::string_view kMatch = R"(fn main(x: u4) -> u32 {
  match x {
    u4:0..u4:2 => u32:0,
    u4:2..u4:4 => u32:1,
    u4:4..u4:6 => u32:2,
    // Intentionally leaving a gap for values 6 and 7.
    u4:8..u4:10 => u32:3,
    _ => fail!("nonexhaustive_match", u32:0),
  }
})";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, NonExhaustiveTupleMixedElementTypes) {
  constexpr std::string_view kMatch = R"(
fn main(x: (u2, bool)) -> u32 {
  match x {
    (u2:0, true) => u32:10,
    (u2:1, false) => u32:20,
  }
}
)";
  CheckNonExhaustive(kMatch);
}

TEST(ExhaustivenessMatchTest, MultiDimensionalTuplePattern) {
  constexpr std::string_view kMatch = R"(
enum E : u2 {
  A = 0,
  B = 1,
  C = 2,
}
fn main(x: (E, bool)) -> u32 {
  match x {
    (E::A, false) => u32:10,
    (E::A, true) => u32:20,
    (E::B, _) => u32:30,
    (E::C, _) => u32:40,
  }
}
)";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MixLiteralAndRangePatterns) {
  constexpr std::string_view kMatch = R"(
fn main(x: u4) -> u32 {
  match x {
    u4:0..u4:2 => u32:0,   // Matches 0 and 1.
    u4:2 => u32:10,        // Matches 2.
    _ => u32:20,           // Covers the rest.
  }
}
)";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MultipleRedundantPatterns) {
  constexpr std::string_view kMatch = R"(
fn main(x: u4) -> u32 {
  match x {
    _ => u32:0,
    u4:1 => u32:10,
    u4:2 => u32:20,
  }
}
)";
  CheckExhaustiveWithRedundantPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, NestedMatchExpression) {
  constexpr std::string_view kProgram = R"(
fn main(x: bool, y: bool) -> u32 {
  // Outer match only has two arms; one of which returns a nested match.
  match x {
    true => match y {
      true => u32:1,
      false => u32:2,
    },
    false => u32:3,
  }
}
)";
  // Because the top-level statement is a match, we can check its
  // exhaustiveness.
  CheckExhaustiveOnlyAfterLastPattern(kProgram);
}

TEST(ExhaustivenessMatchTest, DeeplyNestedTuplePatterns) {
  // This test ensures that the exhaustiveness checker recurses into nested
  // tuples.
  constexpr std::string_view kMatch =
      R"(fn main(x: ((bool, bool), bool)) -> u32 {
    match x {
      ((true, true), true)    => u32:1,
      ((true, true), false)   => u32:2,
      ((true, false), true)   => u32:3,
      ((true, false), false)  => u32:4,
      ((false, true), true)   => u32:5,
      ((false, true), false)  => u32:6,
      ((false, false), true)  => u32:7,
      ((false, false), false) => u32:8,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, RedundantNamedWildcardPattern) {
  // Instead of a "_" wildcard, we bind a variable which covers all cases,
  // then add a redundant arm.
  constexpr std::string_view kMatch = R"(fn main(x: bool) -> u32 {
    match x {
      val => u32:42,  // 'val' binds both true and false
      false => u32:64,  // redundant arm
    }
  })";
  CheckExhaustiveWithRedundantPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, EmptyTupleMatch) {
  // Matching on the unit type is a good edge case.
  constexpr std::string_view kMatch = R"(fn main(x: ()) -> u32 {
    match x {
      () => u32:99,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, OverlappingRangesForSignedIntegers) {
  // Here we deliberately use overlapping ranges for a signed type.
  // Although the ranges overlap, the catch-all arm ensures exhaustiveness.
  constexpr std::string_view kMatch = R"(fn main(x: s4) -> u32 {
    match x {
      s4:-8..s4:-2 => u32:10,
      s4:-4..s4:1  => u32:20,
      s4:0        => u32:30,
      _           => u32:40,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MatchOnLetBoundExpression) {
  // Instead of matching directly on a parameter, match on its transformed
  // value.
  constexpr std::string_view kMatch = R"(fn main(x: bool) -> u32 {
    let y = !x;
    match y {
      true  => u32:5,
      false => u32:10,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, MiddleCatchAllRedundantSuffix) {
  constexpr std::string_view kMatch = R"(fn main(x: u4) -> u32 {
    match x {
      u4:0 => u32:10,
      _ => u32:20,  // catch-all in the middle: makes the match exhaustive here
      u4:1 => u32:30,  // redundant
      u4:2 => u32:40,  // redundant
      u4:3 => u32:50,  // redundant
    }
  })";
  CheckExhaustiveWithRedundantPattern(kMatch);
}

// Note: the second range has a partial overlap with the first range, but we
// don't flag this as a redundant pattern because we only flag redundant
// patterns once we've reached exhaustiveness.
TEST(ExhaustivenessMatchTest, RedundantRangeOverlap) {
  // The second range overlaps with the first one.
  constexpr std::string_view kMatch = R"(fn main(x: u4) -> u32 {
    match x {
      u4:0..u4:3 => u32:10,  // covers 0, 1, 2
      u4:2..u4:4 => u32:20,  // overlaps: covers 2, 3
      _ => u32:30,  // covers the rest (values 4-15)
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, RestOfTupleAtStart) {
  constexpr std::string_view kMatch = R"(fn main(t: (u32, u32, u32)) -> u32 {
    match t {
      (.., u32:3) => u32:100,
      _ => u32:200,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, RestOfTupleWithPrefixAndSuffix) {
  constexpr std::string_view kMatch =
      R"(fn main(t: (u32, u32, u32, u32)) -> u32 {
    match t {
      (u32:10, .., u32:20) => u32:111,
      (u32:30, .., u32:40) => u32:222,
      _ => u32:333,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, NestedTupleWithRestOfTuple) {
  constexpr std::string_view kMatch =
      R"(fn main(t: ((u32, u32, u32), u32)) -> u32 {
    match t {
      ((u32:5, ..), u32:7) => u32:111,
      ((u32:8, ..), u32:9) => u32:222,
      _ => u32:333,
    }
  })";
  CheckExhaustiveOnlyAfterLastPattern(kMatch);
}

TEST(ExhaustivenessMatchTest, NonExhaustiveMatchOnEnumFromImportedModule) {
  constexpr std::string_view kImported =
      "pub enum MyEnum : u1 { A = 0, B = 1 }";
  constexpr std::string_view kProgram = R"(
import imported;

fn main(e: imported::MyEnum) -> u32 {
    match e {
        imported::MyEnum::A => u32:42,
    }
}
)";
  auto vfs = std::make_unique<FakeFilesystem>(
      absl::flat_hash_map<std::filesystem::path, std::string>{
          {std::filesystem::path("/imported.x"), std::string(kImported)},
          {std::filesystem::path("/main.x"), std::string(kProgram)},
      },
      std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(kProgram, "main.x", "main", &import_data);
  // TODO(cdleary): 2025-02-09 We should make a layer where InterpValue can
  // print out resolved enum member names. This needs constexpr-eval'd
  // information on how to map bits values into enum member names.

  EXPECT_THAT(tm.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Match patterns are not exhaustive"),
                             HasSubstr("`MyEnum:1` is not covered"))));
}

}  // namespace
}  // namespace xls::dslx
