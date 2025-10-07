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

#include "xls/dslx/run_routines/run_routines.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/run_routines/ir_test_runner.h"
#include "xls/dslx/run_routines/run_comparator.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

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

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

enum class RunnerType : int8_t {
  kDslxInterpreter,
  kIrJit,
  kIrInterpreter,
  kIrJitProcScoped,
  kIrInterpreterProcScoped,
};

template <typename Sink>
void AbslStringify(Sink& sink, const RunnerType& v) {
  switch (v) {
    case RunnerType::kDslxInterpreter:
      absl::Format(&sink, "DslxInterpreterTestRunner");
      break;
    case RunnerType::kIrJit:
      absl::Format(&sink, "IrJitTestRunner");
      break;
    case RunnerType::kIrInterpreter:
      absl::Format(&sink, "IrInterpreterTestRunner");
      break;
    case RunnerType::kIrJitProcScoped:
      absl::Format(&sink, "IrJitTestRunnerProcScoped");
      break;
    case RunnerType::kIrInterpreterProcScoped:
      absl::Format(&sink, "IrInterpreterTestRunnerProcScoped");
      break;
  }
}
}  // namespace

class RunRoutinesTest : public testing::TestWithParam<RunnerType> {
 public:
  absl::StatusOr<TestResultData> ParseAndTest(
      std::string_view program, std::string_view module_name,
      std::string_view filename, const ParseAndTestOptions& original_options) {
    DslxInterpreterTestRunner dslx;
    IrInterpreterTestRunner ir;
    IrJitTestRunner jit;
    AbstractTestRunner* runner;
    ParseAndTestOptions options(original_options);
    switch (GetParam()) {
      case RunnerType::kDslxInterpreter:
        runner = &dslx;
        break;
      case RunnerType::kIrJit:
        runner = &jit;
        break;
      case RunnerType::kIrInterpreter:
        runner = &ir;
        break;
      case RunnerType::kIrJitProcScoped:
        runner = &jit;
        options.convert_options.proc_scoped_channels = true;
        break;
      case RunnerType::kIrInterpreterProcScoped:
        runner = &ir;
        options.convert_options.proc_scoped_channels = true;
        break;
    }
    return runner->ParseAndTest(program, module_name, filename, options);
  }
};

using ParseAndTestTest = RunRoutinesTest;

TEST_P(RunRoutinesTest, TestInvokedFunctionDoesJit) {
  constexpr const char* kProgram = R"(
fn unit() -> () { () }

#[test]
fn test_simple() { unit() }
)";
  if (GetParam() != RunnerType::kDslxInterpreter) {
    GTEST_SKIP()
        << "comparator only supported on dslx interpreter for non-quickchecks";
  }
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
  EXPECT_EQ(jit_comparator.jit_cache_.begin()->first, "__test__unit");
}

TEST_P(RunRoutinesTest, QuickcheckInvokedFunctionDoesJit) {
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

// A simple exhaustive quickcheck that passes for all values.
TEST_P(RunRoutinesTest, QuickcheckExhaustive) {
  constexpr const char* kProgram = R"(
fn id(x: bool) -> bool { x }

#[quickcheck(exhaustive)]
fn trivial(x: u2) -> bool { id(true) }
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

TEST_P(RunRoutinesTest, QuickcheckExhaustiveEnumWithFail) {
  constexpr const char* kProgram = R"(
enum MyEnum: u2 {
  A = 0,
  B = 1,
  C = 2,
}

#[quickcheck(exhaustive)]
fn qc(x: MyEnum) -> bool {
    match x {
        MyEnum::A | MyEnum::B | MyEnum::C => true,
        _ => fail!("impossible_value", false),
    }
}
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.parse_and_typecheck_options.warnings =
      DisableWarning(kAllWarningsSet, WarningKind::kAlreadyExhaustiveMatch);
  options.run_comparator = &jit_comparator;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));

  ASSERT_EQ(jit_comparator.jit_cache_.size(), 1);
  EXPECT_EQ(jit_comparator.jit_cache_.begin()->first, "__test__qc");
}

TEST_P(RunRoutinesTest, EmptyEnum) {
  constexpr const char* kProgram = R"(
enum EmptyEnum: u2 {
}

#[quickcheck(exhaustive)]
fn qc(x: EmptyEnum) -> bool {
    true
}
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.vfs_factory = [kProgram] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  options.run_comparator = &jit_comparator;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
  ASSERT_EQ(result.GetFailureMessages().size(), 1);
  std::string failure_message = result.GetFailureMessages()[0];
  EXPECT_THAT(failure_message,
              HasSubstr("quickcheck of `qc` rejected all input samples"));
}

// Quickcheck function that takes an `xN` based value and returns an `xN` based
// value.
TEST_P(RunRoutinesTest, QuickcheckXn) {
  constexpr const char* kProgram = R"(
const S: bool = false;
type MyBool = xN[false][1];
type MyBool2 = xN[MyBool:0][1];
#[quickcheck(exhaustive)]
fn qc(x: xN[S][4]) -> MyBool2 {
    let lsb = x[0 +: MyBool2];
    if lsb { lsb } else { MyBool2:1 }
}
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
}

TEST_P(RunRoutinesTest, QuickcheckImplicitTokenRoutineThatFails) {
  constexpr const char* kProgram = R"(
#[quickcheck]
fn qc_with_implicit_token(x: u2) -> bool {
    trace_fmt!("{}", x);  // make an implicit token calling convention
    false  // fail immediately
}
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  options.vfs_factory = [kProgram] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST_P(RunRoutinesTest, GithubIssue1586) {
  constexpr const char* kProgram = R"(
import apfloat;
import bfloat16;
import float32;

const BF16_TOTAL_SZ: u32 = u32:16;

#[quickcheck]
fn bfloat16_bits_to_float32_bits_upcast_is_zero_pad(x: bits[BF16_TOTAL_SZ]) -> bool {
    (x ++ bits[u32:16]:0 ==
    float32::flatten(
        apfloat::upcast_daz<float32::F32_EXP_SZ, float32::F32_FRACTION_SZ>(bfloat16::unflatten(x))))
}
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;

  const std::filesystem::path root("/");
  auto get_stdlib_contents = [](std::string_view filename) -> std::string {
    std::filesystem::path path =
        std::filesystem::path(kDefaultDslxStdlibPath) / filename;
    return GetFileContents(path).value();
  };
  options.parse_and_typecheck_options.dslx_stdlib_path = root / "stdlib";
  options.seed = int64_t{431969656495450};
  options.vfs_factory = [&]() -> std::unique_ptr<VirtualizableFilesystem> {
    return std::make_unique<FakeFilesystem>(
        absl::flat_hash_map<std::filesystem::path, std::string>{
            {"/test.x", kProgram},
            {"/stdlib/std.x", get_stdlib_contents("std.x")},
            {"/stdlib/abs_diff.x", get_stdlib_contents("abs_diff.x")},
            {"/stdlib/apfloat.x", get_stdlib_contents("apfloat.x")},
            {"/stdlib/bfloat16.x", get_stdlib_contents("bfloat16.x")},
            {"/stdlib/float32.x", get_stdlib_contents("float32.x")},
        },
        /*cwd=*/root);
  };
  // Run the quickcheck and inspect that there's a u16 reported in the output.
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
  // Look at the failure message to make sure the u16 is reported.
  std::vector<std::string> failures = result.GetFailureMessages();
  ASSERT_EQ(failures.size(), 1);
  EXPECT_THAT(failures[0], HasSubstr("tests: [u16:"));
}

// An exhaustive quickcheck that fails just for one value in a decently large
// space.
TEST_P(RunRoutinesTest, QuickcheckExhaustiveFail) {
  constexpr std::string_view kProgram = R"(
#[quickcheck(exhaustive)]
fn trivial(x: u11) -> bool { x != u11::MAX }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  options.vfs_factory = [kProgram] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, /*test_cases=*/1,
                                   /*ran_count=*/0, /*failed_count=*/1));
}

// An exhaustive quickcheck that fails just for one value in a decently large
// space using a two tuple of params.
TEST_P(RunRoutinesTest, QuickcheckExhaustive2ParamFail) {
  constexpr std::string_view kProgram = R"(
#[quickcheck(exhaustive)]
fn trivial(x: u5, y: u6) -> bool { !(x == u5::MAX && y == u6::MAX) }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.run_comparator = &jit_comparator;
  options.vfs_factory = [kProgram] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, /*test_cases=*/1,
                                   /*ran_count=*/0, /*failed_count=*/1));
}

TEST_P(RunRoutinesTest, NoSeedStillQuickChecks) {
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

TEST_P(RunRoutinesTest, FallibleFunctionQuickChecks) {
  constexpr const char* kProgram = R"(
fn do_fail(x: bool) -> bool { fail!("oh_no", x) }

#[quickcheck]
fn qc(x: bool) -> bool { do_fail(x) }
)";
  constexpr const char* kModuleName = "test";
  constexpr const char* kFilename = "test.x";
  RunComparator jit_comparator(CompareMode::kJit);
  ParseAndTestOptions options;
  options.seed = int64_t{2316476071057580};
  options.run_comparator = &jit_comparator;
  options.vfs_factory = [kProgram] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, kFilename, options));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST_P(RunRoutinesTest, FailingQuickCheck) {
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

TEST_P(RunRoutinesTest, TwoNonParametricProcs) {
  constexpr std::string_view kProgram = R"(
proc FirstProc {
    data_r: chan<u32> in;
    data_s: chan<u32> out;

    init { () }

    config(data_r: chan<u32> in, data_s: chan<u32> out) { (data_r, data_s) }

    next( state: ()) {
        let (tok, data) = recv(join(), data_r);
        let tok = send(tok, data_s, data);
    }
}

proc MyOtherProc {
    data_r: chan<u32> in;
    data_s: chan<u32> out;

    init {()}

    config(data_r: chan<u32> in, data_s: chan<u32> out) {
        (data_r, data_s)
    }

    next( state: ()) {
        let (tok, data) = recv(join(), data_r);
        let tok = send(tok, data_s, data);
    }
})";

  constexpr const char* kModuleName = "test";
  ParseAndTestOptions options;
  RunComparator jit_comparator(CompareMode::kJit);
  options.run_comparator = &jit_comparator;

  EXPECT_THAT(ParseAndTest(kProgram, kModuleName, "test_module.x", options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Consider turning off comparison")));
}

TEST_P(RunRoutinesTest, FailingProc) {
  constexpr std::string_view kProgram = R"(
#[test_proc]
proc doomed {
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
        (terminator,)
    }

    next(state: ()) {
      let tok = send(join(), terminator, false);
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

TEST_P(RunRoutinesTest, TestProcExpectedToFailOnAssert) {
  if (GetParam() != RunnerType::kDslxInterpreter) {
    GTEST_SKIP() << "expected_fail_label attribute is only supported on dslx "
                    "interpreter for test_procs";
  }

  constexpr std::string_view kProgram = R"(
#[test_proc(expected_fail_label="my_fail")]
proc tester {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (terminator,)
    }
    init {}
    next(_: ()) {
        assert!(false, "my_fail");
        send(join(), terminator, true);
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
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));
}

TEST_P(RunRoutinesTest, TestProcFailingOnDifferentAssertLabelThanExpected) {
  if (GetParam() != RunnerType::kDslxInterpreter) {
    GTEST_SKIP() << "expected_fail_label attribute is only supported on dslx "
                    "interpreter for test_procs";
  }

  constexpr std::string_view kProgram = R"(
#[test_proc(expected_fail_label="my_fail")]
proc tester {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (terminator,)
    }
    init {}
    next(_: ()) {
        assert!(false, "unexpected_fail");
        send(join(), terminator, true);
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

  std::vector<std::string> failures = result.GetFailureMessages();
  EXPECT_THAT(
      failures[0],
      HasSubstr("The program being interpreted failed! Proc failed on "
                "'unexpected_fail', but expected to fail on 'my_fail'"));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST_P(RunRoutinesTest, TestProcExpectedToFailOnFail) {
  if (GetParam() != RunnerType::kDslxInterpreter) {
    GTEST_SKIP() << "expected_fail_label attribute is only supported on dslx "
                    "interpreter for test_procs";
  }

  constexpr std::string_view kProgram = R"(
#[test_proc(expected_fail_label="my_fail")]
proc tester {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (terminator,)
    }
    init {}
    next(_: ()) {
        fail!("my_fail", ());
        send(join(), terminator, true);
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
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));
}

TEST_P(RunRoutinesTest, TestProcFailingOnDifferentFailLabelThanExpected) {
  if (GetParam() != RunnerType::kDslxInterpreter) {
    GTEST_SKIP() << "expected_fail_label attribute is only supported on dslx "
                    "interpreter for test_procs";
  }

  constexpr std::string_view kProgram = R"(
#[test_proc(expected_fail_label="my_fail")]
proc tester {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (terminator,)
    }
    init {}
    next(_: ()) {
        fail!("unexpected_fail", ());
        send(join(), terminator, true);
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

  std::vector<std::string> failures = result.GetFailureMessages();
  EXPECT_THAT(
      failures[0],
      HasSubstr("The program being interpreted failed! Proc failed on "
                "'unexpected_fail', but expected to fail on 'my_fail'"));
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
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
  QuickCheckTestCases test_cases = QuickCheckTestCases::Counted(1000);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);

  std::vector<std::unique_ptr<dslx::Type>> params;
  params.push_back(std::make_unique<dslx::BitsType>(false, 2));
  auto return_type = std::make_unique<dslx::BitsType>(false, 1);
  dslx::FunctionType fn_type(std::move(params), std::move(return_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));
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
  QuickCheckTestCases test_cases = QuickCheckTestCases::Counted(1000);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);

  std::vector<std::unique_ptr<dslx::Type>> params;
  params.push_back(std::make_unique<dslx::ArrayType>(
      std::make_unique<dslx::BitsType>(false, 8), TypeDim::CreateU32(5)));
  auto return_type = std::make_unique<dslx::BitsType>(false, 1);
  dslx::FunctionType fn_type(std::move(params), std::move(return_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));
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
  QuickCheckTestCases test_cases = QuickCheckTestCases::Counted(1000);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);

  std::vector<std::unique_ptr<dslx::Type>> tuple_elems;
  tuple_elems.push_back(std::make_unique<dslx::BitsType>(false, 8));
  tuple_elems.push_back(std::make_unique<dslx::BitsType>(false, 8));

  std::vector<std::unique_ptr<dslx::Type>> params;
  params.push_back(std::make_unique<dslx::TupleType>(std::move(tuple_elems)));
  auto return_type = std::make_unique<dslx::BitsType>(false, 1);
  dslx::FunctionType fn_type(std::move(params), std::move(return_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));
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
  QuickCheckTestCases test_cases = QuickCheckTestCases::Counted(5050);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);

  std::vector<std::unique_ptr<dslx::Type>> params;
  params.push_back(std::make_unique<dslx::BitsType>(false, 32));
  auto return_type = std::make_unique<dslx::BitsType>(false, 1);
  dslx::FunctionType fn_type(std::move(params), std::move(return_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));

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
  QuickCheckTestCases test_cases = QuickCheckTestCases::Counted(1000);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * function,
                           Parser::ParseFunction(ir_text, &package));
  RunComparator jit_comparator(CompareMode::kJit);
  std::vector<std::unique_ptr<dslx::Type>> params;

  params.push_back(std::make_unique<dslx::BitsType>(false, 8));
  auto return_type = std::make_unique<dslx::BitsType>(false, 1);
  dslx::FunctionType fn_type(std::move(params), std::move(return_type));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info1,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto quickcheck_info2,
      DoQuickCheck(/*requires_implicit_token=*/false, &fn_type, function,
                   kFakeIrName, &jit_comparator, seed, test_cases));

  const auto& [argsets1, results1] = quickcheck_info1;
  const auto& [argsets2, results2] = quickcheck_info2;

  EXPECT_EQ(argsets1, argsets2);
  EXPECT_EQ(results1, results2);
}

TEST(QuickcheckTest, ProofFailure) {
  constexpr std::string_view kProgram = R"(
#[quickcheck(exhaustive)]
fn quickcheck_that_fails(x: u1) -> bool {
  x != x
}
)";
  ParseAndProveOptions options;
  options.vfs_factory = [&] {
    return std::make_unique<UniformContentFilesystem>(kProgram, "test.x");
  };
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           ParseAndProve(kProgram, "test", "test.x", options));
  EXPECT_THAT(result.test_result_data,
              IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
  EXPECT_THAT(result.counterexamples.size(), 1);
  EXPECT_THAT(result.counterexamples.begin()->second,
              testing::ElementsAre(Value(UBits(0, 1))));
}

TEST_P(ParseAndTestTest, DeadlockedProc) {
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

  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
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
    let (input_out, input_in) = chan<u32>("input");
    let (output_out, output_in) = chan<u32>("output");
    spawn incrementer(input_in, output_out);
    (input_out, output_in, terminator)
  }

  next(state: ()) {
    let tok = send_if(join(), data_out, false, u32:42);
    let (tok, _result) = recv(tok, data_in);
    let tok = send(tok, terminator, u1:1);
 }
})";
  ParseAndTestOptions options;
  options.max_ticks = 100;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test_module", "test.x", options));

  std::vector<std::string> failures = result.GetFailureMessages();
  EXPECT_EQ(failures.size(), 1);
  if (GetParam() == RunnerType::kDslxInterpreter) {
    EXPECT_THAT(failures[0],
                AllOf(HasSubstr("proc `incrementer` is blocked on receive on "
                                "channel `tester_proc->incrementer#0::in_ch`"),
                      HasSubstr("proc `tester_proc` is blocked on receive on "
                                "channel `tester_proc::data_in`")));
  } else {
    EXPECT_THAT(failures[0], HasSubstr("deadlocked"));
  }
  EXPECT_THAT(result, IsTestResult(TestResult::kSomeFailed, 1, 0, 1));
}

TEST_P(ParseAndTestTest, TooManyTicks) {
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
  next(_: ()) {
    let (tok, i) = recv_if(join(), in_ch, false, u32:0);
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
    let (input_out, input_in) = chan<u32>("input");
    let (output_out, output_in) = chan<u32>("output");
    spawn incrementer(input_in, output_out);
    (input_out, output_in, terminator)
  }

  next(state: ()) {
    let tok = send(join(), data_out, u32:42);
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

TEST_P(ParseAndTestTest, TestFilterEmpty) {
  const ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 0, 0));
}

TEST_P(ParseAndTestTest, TestFilterSelectNone) {
  const RE2 test_filter("doesnotexist");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 2, 0));
}

TEST_P(ParseAndTestTest, TestFilterSelectOne) {
  const RE2 test_filter(".*_one");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 1, 0));
}

TEST_P(ParseAndTestTest, TestFilterSelectBoth) {
  const RE2 test_filter("test_.*");
  ParseAndTestOptions options;
  options.test_filter = &test_filter;
  XLS_ASSERT_OK_AND_ASSIGN(TestResultData result,
                           ParseAndTest(kTwoTests, "test", "test.x", options));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 2, 0, 0));
}

// Exercises https://github.com/google/xls/issues/1368
TEST_P(ParseAndTestTest, StructParametricFromProcParametric) {
  constexpr std::string_view kProgram = R"(
struct Data<WIDTH: u32> {struct_field_value: uN[WIDTH]}

proc MyProc<DATA_WIDTH: u32> {
  type MyProcData = Data<DATA_WIDTH>;
  out_s: chan<MyProcData> out;

  config(out_s: chan<MyProcData> out) {(out_s, )}
  init {}

 next(_: ()) {
    // This line was failing in typecheck previously
    send(join(), out_s, MyProcData{struct_field_value: uN[DATA_WIDTH]:42});
  }
}

const TEST_WIDTH = u32:10;

#[test_proc]
proc MyProcTest {
  type MyProcTestData = Data<TEST_WIDTH>;
  terminator: chan<bool> out;
  out_r: chan<MyProcTestData> in;

  config(terminator: chan<bool> out) {
    let(out_s, out_r) = chan<MyProcTestData>("out");
    spawn MyProc<TEST_WIDTH>(out_s);
    (terminator, out_r)
  }

  init {}

  next(_: ()) {
    let(tok, returned_struct) = recv(join(), out_r);
    assert_eq(returned_struct.struct_field_value, uN[TEST_WIDTH]:42);
    send(tok, terminator, true);
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", "test.x", ParseAndTestOptions{}));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));
}

TEST_P(ParseAndTestTest, StructParametricFromFnParametric) {
  constexpr std::string_view kProgram = R"(
struct Data<WIDTH: u32> {value: uN[WIDTH]}

fn myFn<DATA_WIDTH: u32>() -> Data<DATA_WIDTH> {
  type MyFnData = Data<DATA_WIDTH>;
  let data = MyFnData{value: uN[DATA_WIDTH]:42};
  data
}

const TEST_WIDTH = u32:32;

#[test]
fn test_simple() {
  let output = myFn<TEST_WIDTH>();
  assert_eq(output.value, uN[TEST_WIDTH]:42);
  }
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", "test.x", ParseAndTestOptions{}));
  EXPECT_THAT(result, IsTestResult(TestResult::kAllPassed, 1, 0, 0));
}

INSTANTIATE_TEST_SUITE_P(RunRoutinesTest, RunRoutinesTest,
                         testing::Values(RunnerType::kDslxInterpreter,
                                         RunnerType::kIrInterpreter,
                                         RunnerType::kIrJit,
                                         RunnerType::kIrInterpreterProcScoped,
                                         RunnerType::kIrJitProcScoped),
                         testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(ParseAndTestTest, ParseAndTestTest,
                         testing::Values(RunnerType::kDslxInterpreter,
                                         RunnerType::kIrInterpreter,
                                         RunnerType::kIrJit,
                                         RunnerType::kIrInterpreterProcScoped,
                                         RunnerType::kIrJitProcScoped),
                         testing::PrintToStringParamName());

}  // namespace xls::dslx
