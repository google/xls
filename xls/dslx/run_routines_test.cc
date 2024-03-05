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

#include "xls/dslx/run_routines.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/run_comparator.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

// A fake mangled IR name for use in some direct DoQuickCheck calls (will be
// used as a JIT cache key).
constexpr std::string_view kFakeIrName = "__test__fake";

// Matcher that helps us compare against a TestResultData value and what it
// reflects as the summary result status and counts.
MATCHER_P4(IsTestResult, result, ran_count, skipped_count, failed_count, "") {
  if (result != arg.result()) {
    *result_listener << "got " << arg.result() << " want " << result;
    return false;
  }
  if (ran_count != arg.GetRanCount()) {
    *result_listener << "ran count got " << arg.GetRanCount() << " want "
                     << ran_count;
    return false;
  }
  if (skipped_count != arg.GetSkippedCount()) {
    *result_listener << "skipped count got " << arg.GetSkippedCount()
                     << " want " << skipped_count;
    return false;
  }
  if (failed_count != arg.GetFailedCount()) {
    *result_listener << "skipped count got " << arg.GetFailedCount() << " want "
                     << failed_count;
    return false;
  }
  return true;
}

}  // namespace

TEST(RunRoutinesTest, TestInvokedFunctionDoesJit) {
  constexpr const char* kProgram = R"(
fn unit() -> () { () }

#[test]
fn test_simple() { unit() }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));

  EXPECT_EQ(jit_comparator.jit_cache_.size(), 1);
  EXPECT_EQ(jit_comparator.jit_cache_.begin()->first, "__test__unit");
}

TEST(RunRoutinesTest, QuickcheckInvokedFunctionDoesJit) {
  constexpr const char* kProgram = R"(
fn id(x: bool) -> bool { x }

#[quickcheck(test_count=1024)]
fn trivial(x: u5) -> bool { id(true) }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  options.seed = int64_t{2};
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));

  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));

  ASSERT_EQ(jit_comparator.jit_cache_.size(), 1);
  EXPECT_EQ(jit_comparator.jit_cache_.begin()->first, "__test__trivial");
}

TEST(RunRoutinesTest, NoSeedStillQuickChecks) {
  constexpr const char* kProgram = R"(
fn id(x: bool) -> bool { x }

#[quickcheck(test_count=1024)]
fn trivial(x: u5) -> bool { id(true) }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));

  ASSERT_EQ(jit_comparator.jit_cache_.size(), 1);
  EXPECT_EQ(jit_comparator.jit_cache_.begin()->first, "__test__trivial");
}

TEST(RunRoutinesTest, FallibleFunctionQuickChecks) {
  constexpr const char* kProgram = R"(
fn do_fail(x: bool) -> bool { fail!("oh_no", x) }

#[quickcheck]
fn qc(x: bool) -> bool { do_fail(x) }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST(RunRoutinesTest, FailingQuickCheck) {
  constexpr const char* kProgram = R"(
#[quickcheck(test_count=2)]
fn trivial(x: u5) -> bool { false }
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr const char* kModuleName = "test";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  options.seed = int64_t{42};
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, std::string(temp_file.path()),
                   options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST(RunRoutinesTest, FailingProc) {
  constexpr std::string_view kProgram = R"(
#[test_proc]
proc doomed {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        (terminator,)
    }

    next(tok: token, state: ()) {
      let tok = send(tok, terminator, false);
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr const char* kModuleName = "test";
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, std::string(temp_file.path()),
                   options));
  EXPECT_THAT(result,
              IsTestResult(TestResult::kParseOrTypecheckError, 0, 0, 0));
}

// Verifies that the QuickCheck mechanism can find counter-examples for a simple
// erroneous function.
TEST(QuickcheckTest, QuickCheckBits) {
  Package package("bad_bits_property");
  std::string ir_text = R"(
  fn adjacent_bits(x: bits[2]) -> bits[1] {
    first_bit: bits[1] = bit_slice(x, start=0, width=1)
    second_bit: bits[1] = bit_slice(x, start=1, width=1)
    ret eq_value: bits[1] = eq(first_bit, second_bit)
  }
  )";
  int64_t seed = 0;
  int64_t num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));
  std::vector<Value> results = quickcheck_info.results;
  // If a counter-example was found, the last result will be 0.
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

TEST(QuickcheckTest, QuickCheckArray) {
  Package package("bad_array_property");
  std::string ir_text = R"(
  fn adjacent_elements(x: bits[8][5]) -> bits[1] {
    zero: bits[32] = literal(value=0)
    one: bits[32] = literal(value=1)
    first_element: bits[8] = array_index(x, indices=[zero])
    second_element: bits[8] = array_index(x, indices=[one])
    ret eq_value: bits[1] = eq(first_element, second_element)
  }
  )";
  int64_t seed = 0;
  int64_t num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));
  std::vector<Value> results = quickcheck_info.results;
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

TEST(QuickcheckTest, QuickCheckTuple) {
  Package package("bad_tuple_property");
  std::string ir_text = R"(
  fn adjacent_elements(x: (bits[8], bits[8])) -> bits[1] {
    first_member: bits[8] = tuple_index(x, index=0)
    second_member: bits[8] = tuple_index(x, index=1)
    ret eq_value: bits[1] = eq(first_member, second_member)
  }
  )";
  int64_t seed = 0;
  int64_t num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));
  std::vector<Value> results = quickcheck_info.results;
  EXPECT_EQ(results.back(), Value(UBits(0, 1)));
}

// If the QuickCheck mechanism can't find a falsifying example, we expect
// the argsets and results vectors to have lengths of 'num_tests'.
TEST(QuickcheckTest, NumTests) {
  Package package("always_true");
  std::string ir_text = R"(
  fn ret_true(x: bits[32]) -> bits[1] {
    ret eq_value: bits[1] = eq(x, x)
  }
  )";
  int64_t seed = 0;
  int64_t num_tests = 5050;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));

  std::vector<std::vector<Value>> argsets = quickcheck_info.arg_sets;
  std::vector<Value> results = quickcheck_info.results;
  EXPECT_EQ(argsets.size(), 5050);
  EXPECT_EQ(results.size(), 5050);
}

// Given a constant seed, we expect the same argsets and results vectors from
// two runs through the QuickCheck mechanism.
TEST(QuickcheckTest, Seeding) {
  Package package("sometimes_false");
  std::string ir_text = R"(
  fn gt_one(x: bits[8]) -> bits[1] {
    literal.2: bits[8] = literal(value=1)
    ret ugt.3: bits[1] = ugt(x, literal.2)
  }
  )";
  int64_t seed = 12345;
  int64_t num_tests = 1000;
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info1,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info2,
      DoQuickCheck(function, kFakeIrName, &jit_comparator, seed, num_tests));

  const auto& [argsets1, results1] = quickcheck_info1;
  const auto& [argsets2, results2] = quickcheck_info2;

  EXPECT_EQ(argsets1, argsets2);
  EXPECT_EQ(results1, results2);
}

TEST(ParseAndTestTest, DeadlockedProc) {
  // Test proc never sends to the subproc, so network is deadlocked.
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i) = recv(tok, in_ch);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_out, input_in) = chan<u32>;
    let (output_out, output_in) = chan<u32>;
    spawn incrementer(input_in, output_out);
    (input_out, output_in, terminator)
  }

  next(tok: token, state: ()) {
    let tok = send_if(tok, data_out, false, u32:42);
    let (tok, _result) = recv(tok, data_in);
    let tok = send(tok, terminator, true);
 }
})";
  ParseAndTestOptions options;
  options.max_ticks = 100;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test_module", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST(ParseAndTestTest, TooManyTicks) {
  // Test proc never receives and spins forever.
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(tok: token, _: ()) {
    let (tok, i) = recv_if(tok, in_ch, false, u32:0);
    let tok = send_if(tok, out_ch, false, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_out, input_in) = chan<u32>;
    let (output_out, output_in) = chan<u32>;
    spawn incrementer(input_in, output_out);
    (input_out, output_in, terminator)
  }

  next(tok: token, state: ()) {
    let tok = send(tok, data_out, u32:42);
    let (tok, _result) = recv(tok, data_in);
    let tok = send(tok, terminator, true);
 }
})";
  ParseAndTestOptions options;
  options.max_ticks = 100;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test_module", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

inline constexpr std::string_view kTwoTests = R"(
#[test] fn test_one() {}
#[test] fn test_two() {}
)";

TEST(ParseAndTestTest, TestFilterEmpty) {
  const ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 0, 0));
}

TEST(ParseAndTestTest, TestFilterSelectNone) {
  const RE2 test_filter("doesnotexist");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 2, 0));
}

TEST(ParseAndTestTest, TestFilterSelectOne) {
  const RE2 test_filter(".*_one");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 1, 0));
}

TEST(ParseAndTestTest, TestFilterSelectBoth) {
  const RE2 test_filter("test_.*");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 0, 0));
}

}  // namespace xls::dslx
