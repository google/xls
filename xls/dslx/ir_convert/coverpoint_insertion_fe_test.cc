// Copyright 2026 The XLS Authors
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

// Tests for the frontend (DSLX→IR) coverpoint insertion, gated by the
// `emit_auto_cover` convert option.
//
// Coverpoints are only emitted for functions/procs that use the implicit-token
// calling convention (the same requirement as explicit `cover!()` calls), so
// the tests below force that convention on plain functions. The labels written
// for synthesized covers omit any file path (so they are usable as Verilog
// cover-property names) and instead encode the enclosing function name, the
// branch kind, and a `line_<N>_pos_<M>` location.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter_test_utils.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

// Builds the convert options used throughout the tests: enable synthesized
// branch coverpoints and force the implicit-token calling convention so plain
// functions receive coverpoints during conversion.
ConvertOptions DefaultCoverOptions() {
  ConvertOptions options;
  options.emit_positions = false;
  options.emit_auto_cover.Insert(AutoCoverKind::kBranch);
  options.force_implicit_token_calling_convention = true;
  return options;
}

const ConvertOptions kCoverOptions = DefaultCoverOptions();

class FrontendCoverpointInsertionTest : public IrConverterTest {
 protected:
  // Package from the most recent conversion, kept alive so the returned
  // function pointer remains valid across the test body.
  std::unique_ptr<xls::Package> package_;

  // Converts `program` with coverpoints enabled, reparses the resulting IR into
  // a package, and returns the implicit-token function (named
  // `__itok__<module>__<fn>`) which holds the synthesized covers.
  //
  // If `options` is nullopt the default `kCoverOptions` are used; pass an
  // explicit set to test disabled/mutated options.
  absl::StatusOr<xls::Function*> ConvertToFunction(
      std::string_view program, std::string_view fn_name,
      const std::optional<ConvertOptions>& options = std::nullopt) {
    XLS_ASSIGN_OR_RETURN(
        std::string ir,
        ConvertModuleForTest(program, options.value_or(kCoverOptions)));
    XLS_ASSIGN_OR_RETURN(package_, Parser::ParsePackage(ir));
    std::string implicit_name =
        absl::StrFormat("__itok__test_module__%s", fn_name);
    return package_->GetFunction(implicit_name);
  }

  // Returns the labels of all `cover` nodes in `f`.
  std::vector<std::string> CoverLabels(const xls::Function* f) const {
    std::vector<std::string> labels;
    for (Node* node : f->nodes()) {
      if (node->Is<Cover>()) {
        labels.push_back(node->As<Cover>()->label());
      }
    }
    return labels;
  }

  // Returns the source locations attached to each `cover` node in `f`, as
  // `(lineno, colno)` pairs. Nodes without a source location are skipped.
  // `Lineno`/`Colno` are 0-indexed IR positions.
  std::vector<std::pair<int64_t, int64_t>> CoverLocations(
      const xls::Function* f) const {
    std::vector<std::pair<int64_t, int64_t>> locations;
    for (Node* node : f->nodes()) {
      if (node->Is<Cover>() && !node->loc().Empty()) {
        const SourceLocation& loc = node->loc().locations.front();
        locations.emplace_back(loc.lineno().value(), loc.colno().value());
      }
    }
    return locations;
  }

  // Interprets `f` on `args` and returns the labels of the covers whose
  // condition evaluated to 1, i.e. those that fired on this input. This
  // verifies behavioral firing semantics rather than just wiring structure.
  //
  // `args` must correspond to the implicit function's parameters, i.e. the
  // first two are the token and the activation bit.
  absl::StatusOr<absl::flat_hash_set<std::string>> FiredCovers(
      xls::Function* f, absl::Span<const Value> args) const {
    absl::flat_hash_map<Node*, std::string> label_by_condition;
    for (Node* node : f->nodes()) {
      if (node->Is<Cover>()) {
        label_by_condition[node->As<Cover>()->condition()] =
            node->As<Cover>()->label();
      }
    }

    CollectingEvaluationObserver observer;
    EvaluatorOptions options;
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                         InterpretFunction(f, args, options, &observer));

    absl::flat_hash_set<std::string> fired;
    for (const auto& [cond, label] : label_by_condition) {
      auto it = observer.values().find(cond);
      if (it != observer.values().end() && !it->second.empty() &&
          it->second.back().IsBits() &&
          it->second.back().bits().ToUint64().value() == 1) {
        fired.insert(label);
      }
    }
    // A cover that is structurally unreachable on this input fires never: any
    // label the interpreter touched but left at 0 is simply absent from
    // `fired`.
    return fired;
  }

  // Builds the argument list for invoking the implicit function `f`, starting
  // with the token and activation bit. User-visible arguments follow.
  std::vector<Value> ImplicitArgs(const xls::Function* f, bool activated,
                                  const std::vector<Value>& user_args) {
    std::vector<Value> args;
    args.push_back(Value::Token());
    args.push_back(Value(UBits(activated ? 1 : 0, 1)));
    args.insert(args.end(), user_args.begin(), user_args.end());
    return args;
  }
};

// Default options do not insert covers; the option is opt-in and off by
// default.
TEST_F(FrontendCoverpointInsertionTest, DisabledByDefault) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
if x > u32:4 { u32:1 } else { u32:0 }
}
)";
  const ConvertOptions options = {
      .emit_positions = false,
      .force_implicit_token_calling_convention = true,
  };
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           ConvertToFunction(program, "f", options));
  EXPECT_THAT(CoverLabels(f), IsEmpty());
}

// A trivial if/else gets one cover per branch: `_then` and `_else`, and each
// fires exactly when its branch is taken.
TEST_F(FrontendCoverpointInsertionTest, TrivialIfElse) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
let y = if x > u32:4 { x + u32:1 } else { x - u32:1 };
y
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  EXPECT_THAT(CoverLabels(f),
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_9",
                                   "__itok__test_module__f_else_line_3_pos_9"));

  // x > 4 fires the then-cover.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_gt,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(5, 32))})));
  EXPECT_THAT(fired_gt,
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_9"));

  // x <= 4 fires the else-cover.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_le,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(4, 32))})));
  EXPECT_THAT(fired_le,
              UnorderedElementsAre("__itok__test_module__f_else_line_3_pos_9"));

  // Deactivating the token suppression fires nothing.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_inactive,
      FiredCovers(f, ImplicitArgs(f, false, {Value(UBits(5, 32))})));
  EXPECT_THAT(fired_inactive, IsEmpty());
}

// `if ... else if ... else` emits a cover per branch. The `else if` is
// structurally an `if` nested in the outer else, so it yields an outer
// then/else plus an inner then/else.
TEST_F(FrontendCoverpointInsertionTest, IfElseIfElse) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
if x > u32:10 { u32:1 } else if x > u32:4 { u32:2 } else { u32:3 }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  // Outer if sits at line 3 pos 1; the inner `else if` sits at line 3 pos 30.
  EXPECT_THAT(CoverLabels(f), UnorderedElementsAre(
                                  "__itok__test_module__f_then_line_3_pos_1",
                                  "__itok__test_module__f_else_line_3_pos_1",
                                  "__itok__test_module__f_then_line_3_pos_30",
                                  "__itok__test_module__f_else_line_3_pos_30"));

  // x > 10: innermost then (outer then).
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_gt10,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(11, 32))})));
  EXPECT_THAT(fired_gt10,
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_1"));

  // 4 < x <= 10: outer else, inner then.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_outer_else,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(6, 32))})));
  EXPECT_THAT(
      fired_outer_else,
      UnorderedElementsAre("__itok__test_module__f_else_line_3_pos_1",
                           "__itok__test_module__f_then_line_3_pos_30"));

  // x <= 4: outer else, inner else.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_both_else,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(4, 32))})));
  EXPECT_THAT(
      fired_both_else,
      UnorderedElementsAre("__itok__test_module__f_else_line_3_pos_1",
                           "__itok__test_module__f_else_line_3_pos_30"));
}

// `match` arms get `_match_arm<N>` covers, and only the matching arm fires.
TEST_F(FrontendCoverpointInsertionTest, MatchArms) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
match x { u32:1 => { u32:1 }, u32:2 => { u32:2 }, _ => { u32:3 } }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  EXPECT_THAT(
      CoverLabels(f),
      UnorderedElementsAre("__itok__test_module__f_match_arm0_line_3_pos_1",
                           "__itok__test_module__f_match_arm1_line_3_pos_1",
                           "__itok__test_module__f_match_arm2_line_3_pos_1"));

  // Selector 1 matches the first arm.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_1,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(1, 32))})));
  EXPECT_THAT(fired_1, UnorderedElementsAre(
                           "__itok__test_module__f_match_arm0_line_3_pos_1"));

  // Selector 2 matches the second arm.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_2,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(2, 32))})));
  EXPECT_THAT(fired_2, UnorderedElementsAre(
                           "__itok__test_module__f_match_arm1_line_3_pos_1"));

  // Any other value takes the default (last) arm.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_def,
      FiredCovers(f, ImplicitArgs(f, true, {Value(UBits(99, 32))})));
  EXPECT_THAT(fired_def, UnorderedElementsAre(
                             "__itok__test_module__f_match_arm2_line_3_pos_1"));
}

// Inner covers are gated on the full nesting: an inner arm fires only when both
// the outer and the inner branch are taken. This exercises the
// nested-reachability gating that a flat IR-level pass cannot express.
TEST_F(FrontendCoverpointInsertionTest, NestedCascade) {
  constexpr std::string_view program = R"(
fn f(x: u32, y: u32) -> u32 {
let outer = if x > u32:4 {
if y > u32:2 { x } else { y }
} else {
u32:0
};
u32:1
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  // Outer if at line 3 pos 13; inner if at line 4 pos 1.
  EXPECT_THAT(CoverLabels(f),
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_13",
                                   "__itok__test_module__f_else_line_3_pos_13",
                                   "__itok__test_module__f_then_line_4_pos_1",
                                   "__itok__test_module__f_else_line_4_pos_1"));

  // x > 4 and y > 2: outer then + inner then.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_tt,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(5, 32)), Value(UBits(3, 32))})));
  EXPECT_THAT(fired_tt,
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_13",
                                   "__itok__test_module__f_then_line_4_pos_1"));

  // x > 4 and y <= 2: outer then + inner else.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_te,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(5, 32)), Value(UBits(2, 32))})));
  EXPECT_THAT(fired_te,
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_13",
                                   "__itok__test_module__f_else_line_4_pos_1"));

  // x <= 4: only the outer else; inner covers must NOT fire even though the
  // inner arms' own conditions could be true.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_oe,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(1, 32)), Value(UBits(3, 32))})));
  EXPECT_THAT(fired_oe, UnorderedElementsAre(
                            "__itok__test_module__f_else_line_3_pos_13"));
}

// Multiple functions in one module are each instrumented independently.
TEST_F(FrontendCoverpointInsertionTest, InstrumentsMultipleFunctions) {
  constexpr std::string_view program = R"(
fn a(x: u32) -> u32 { if x > u32:4 { u32:1 } else { u32:0 } }
fn b(y: u32) -> u32 { if y > u32:9 { u32:2 } else { u32:3 } }
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fa, ConvertToFunction(program, "a"));
  // Both functions live in the same package, so `fb` is valid alongside `fa`.
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fb,
                           package_->GetFunction("__itok__test_module__b"));

  EXPECT_THAT(
      CoverLabels(fa),
      UnorderedElementsAre("__itok__test_module__a_then_line_2_pos_23",
                           "__itok__test_module__a_else_line_2_pos_23"));
  EXPECT_THAT(
      CoverLabels(fb),
      UnorderedElementsAre("__itok__test_module__b_then_line_3_pos_23",
                           "__itok__test_module__b_else_line_3_pos_23"));
}

// The auto-generated cover nodes carry a debug source location pointing at the
// originating branch statement. This is independent of the human-readable label
// (which encodes line/pos as text) and is only emitted when `emit_positions` is
// enabled (the label's line/pos is emitted regardless).
TEST_F(FrontendCoverpointInsertionTest, CoversCarrySourceLocations) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
if x != u32:0 { x } else { u32:0 }
}
)";
  ConvertOptions options;
  options.emit_positions = true;
  options.emit_auto_cover.Insert(AutoCoverKind::kBranch);
  options.force_implicit_token_calling_convention = true;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           ConvertToFunction(program, "f", options));

  // Both branches originate at the `if` statement on line 3 (1-based),
  // column 1.
  const std::vector<std::string> labels = CoverLabels(f);
  ASSERT_EQ(labels.size(), 2);

  // 0-indexed position of the `if` keyword: line 2, column 0.
  const std::vector<std::pair<int64_t, int64_t>> locations = CoverLocations(f);
  ASSERT_EQ(locations.size(), 2);
  for (const auto& [lineno, colno] : locations) {
    EXPECT_EQ(lineno, 2);
    EXPECT_EQ(colno, 0);
  }

  // The debug location and the label's encoded position are consistent: the
  // label carries a 1-based line/pos, the `SourceInfo` carries 0-based.
  for (int i = 0; i < labels.size(); ++i) {
    EXPECT_THAT(labels[i], HasSubstr(absl::StrFormat("_line_%d_pos_%d",
                                                     locations[i].first + 1,
                                                     locations[i].second + 1)));
  }
}

// With `emit_positions` disabled the labels still carry a position, but the
// debug `SourceInfo` on the cover nodes is suppressed.
TEST_F(FrontendCoverpointInsertionTest,
       SourceLocationsDisabledWithoutPositions) {
  constexpr std::string_view program = R"(
fn f(x: u32) -> u32 {
if x != u32:0 { x } else { u32:0 }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  EXPECT_THAT(CoverLabels(f),
              UnorderedElementsAre("__itok__test_module__f_then_line_3_pos_1",
                                   "__itok__test_module__f_else_line_3_pos_1"));
  for (Node* node : f->nodes()) {
    if (node->Is<Cover>()) {
      EXPECT_TRUE(node->loc().Empty());
    }
  }
}

// Two `match` expressions starting on the same source line must still yield
// unique labels. Match arms share the whole-`match` span, so label uniqueness
// relies on the per-arm integer suffix together with the distinct column
// positions of the two matches. A collision here would emit duplicate Verilog
// cover-property names.
TEST_F(FrontendCoverpointInsertionTest, MatchLabelsUniqueAcrossSameLine) {
  constexpr std::string_view program = R"(
fn f(x: u32, y: u32) -> u32 {
let a = match x { u32:1 => { u32:1 }, _ => { u32:0 } }; let b = match y { u32:2 => { u32:2 }, _ => { u32:0 } }; a + b
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  const std::vector<std::string> labels = CoverLabels(f);
  ASSERT_EQ(labels.size(), 4);
  // Every label is distinct even though both matches start on line 3.
  EXPECT_EQ(
      absl::flat_hash_set<std::string>(labels.begin(), labels.end()).size(), 4);

  EXPECT_THAT(labels, UnorderedElementsAre(
                          "__itok__test_module__f_match_arm0_line_3_pos_9",
                          "__itok__test_module__f_match_arm1_line_3_pos_9",
                          "__itok__test_module__f_match_arm0_line_3_pos_65",
                          "__itok__test_module__f_match_arm1_line_3_pos_65"));

  // x == 1 and y == 2: both first arms fire.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_both_first,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(1, 32)), Value(UBits(2, 32))})));
  EXPECT_THAT(
      fired_both_first,
      UnorderedElementsAre("__itok__test_module__f_match_arm0_line_3_pos_9",
                           "__itok__test_module__f_match_arm0_line_3_pos_65"));
}

// A `match` whose arm body contains an `if` combines the two forms of
// auto-cover instrumentation: the arm cover fires when the arm is selected, and
// the nested branch covers fire only when that arm *and* the inner branch are
// both taken. This exercises the match-inside-if / else-if-inside-match nesting
// combination.
TEST_F(FrontendCoverpointInsertionTest, IfInsideMatchArmGating) {
  constexpr std::string_view program = R"(
fn f(x: u32, y: u32) -> u32 {
match x {
  u32:0 => if y > u32:4 { u32:1 } else { u32:0 },
  _ => { u32:2 }
}
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, ConvertToFunction(program, "f"));

  // 2 match arms + 2 branches of the nested if = 4 covers.
  EXPECT_EQ(CoverLabels(f).size(), 4);

  // x == 0, y > 4: arm 0 and the inner then-branch fire.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_then,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(0, 32)), Value(UBits(5, 32))})));
  ASSERT_EQ(fired_then.size(), 2);
  EXPECT_THAT(fired_then, UnorderedElementsAre(HasSubstr("_match_arm0_"),
                                               HasSubstr("_then_line_4_")));

  // x == 0, y <= 4: arm 0 and the inner else-branch fire.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_else,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(0, 32)), Value(UBits(3, 32))})));
  ASSERT_EQ(fired_else.size(), 2);
  EXPECT_THAT(fired_else, UnorderedElementsAre(HasSubstr("_match_arm0_"),
                                               HasSubstr("_else_line_4_")));

  // x != 0: the default arm fires, but the inner covers must NOT fire even
  // though `y > 4` could itself be true.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto fired_default,
      FiredCovers(f, ImplicitArgs(f, true,
                                  {Value(UBits(7, 32)), Value(UBits(9, 32))})));
  EXPECT_THAT(fired_default, UnorderedElementsAre(HasSubstr("_match_arm1_")));
  EXPECT_THAT(fired_default, Not(Contains(HasSubstr("_then_line_4_"))));
  EXPECT_THAT(fired_default, Not(Contains(HasSubstr("_else_line_4_"))));
}

}  // namespace
}  // namespace xls::dslx
