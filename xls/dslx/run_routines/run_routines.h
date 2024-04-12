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

// Routines for "running" DSLX files; i.e. parsing and testing all of the tests
// contained inside.

#ifndef XLS_DSLX_RUN_ROUTINES_H_
#define XLS_DSLX_RUN_ROUTINES_H_

#include <algorithm>
#include <any>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/run_routines/test_xml.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/events.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "re2/re2.h"

namespace xls::dslx {

// Abstract API used for comparing DSLX-interpreter results to executed IR
// results. This is a virtual API to help decouple from implementation details
// like whether the JIT is available or only interpretation, or whether we
// should perhaps compare against both.
class AbstractRunComparator {
 public:
  virtual ~AbstractRunComparator() = default;

  // Runs a comparison of the DSLX_interpreter-determined value against the
  // otherwise-determined value (e.g. IR interpreter or IR JIT).
  virtual absl::Status RunComparison(Package* ir_package,
                                     bool requires_implicit_token,
                                     const Function* f,
                                     absl::Span<InterpValue const> args,
                                     const ParametricEnv* parametric_env,
                                     const InterpValue& got) = 0;

  // Helper for abstracting over the running of IR functions. i.e. we implement
  // this in subclasses to either execute JIT'd computations or interpreted
  // ones.
  //
  // Args:
  //  ir_name: Already-mangled DSLX name.
  //  ir_function: Corresponding IR function.
  //  ir_args: Arguments to invoke the IR function with.
  virtual absl::StatusOr<InterpreterResult<xls::Value>> RunIrFunction(
      std::string_view ir_name, xls::Function* ir_function,
      absl::Span<const xls::Value> ir_args) = 0;
};

// Optional arguments to ParseAndTest (that have sensible defaults).
//
//   test_filter: Test filter specification (e.g. as passed from bazel test
//     environment).
//   run_comparator: Optional object that can compare DSLX interpreter
//    executions with a reference (e.g. IR execution).
//   execute: Whether or not to execute the quickchecks and tests.
//   seed: Seed for QuickCheck random input stimulus.
//   convert_options: Options used in IR conversion, see `ConvertOptions` for
//    details.
//   warnings_as_errors: Whether warnings should be reported as errors (i.e.
//    cause the run routine to report failure when a warning is encountered).
//   warnings: Set of warnings to enable for reporting.
struct ParseAndTestOptions {
  std::string stdlib_path = xls::kDefaultDslxStdlibPath;
  absl::Span<const std::filesystem::path> dslx_paths;
  const RE2* test_filter = nullptr;
  FormatPreference format_preference = FormatPreference::kDefault;
  AbstractRunComparator* run_comparator = nullptr;
  bool execute = true;
  std::optional<int64_t> seed = std::nullopt;
  ConvertOptions convert_options;
  bool warnings_as_errors = true;
  WarningKindSet warnings = kDefaultWarningsSet;
  bool trace_channels = false;
  std::optional<int64_t> max_ticks;
};

// As above, but a subset of the options required for the ParseAndProve()
// routine.
struct ParseAndProveOptions {
  std::string stdlib_path = xls::kDefaultDslxStdlibPath;
  absl::Span<const std::filesystem::path> dslx_paths;
  bool warnings_as_errors = true;
  WarningKindSet warnings = kDefaultWarningsSet;
};

enum class TestResult : uint8_t {
  kFailedWarnings,
  kSomeFailed,
  kAllPassed,
  kParseOrTypecheckError,
};

std::string_view TestResultToString(TestResult tr);

template <typename Sink>
inline void AbslStringify(Sink& sink, TestResult tr) {
  absl::Format(&sink, "%s", TestResultToString(tr));
}

inline std::ostream& operator<<(std::ostream& os, TestResult tr) {
  os << TestResultToString(tr);
  return os;
}

// Holds the results (and summary result) from test execution from e.g.
// `ParseAndTest()`.
//
// Implementation note: this datatype also serves as a test data collector --
// ideally those concerns might be separated out but there's a minimal amount of
// functionality really required so doesn't seem worthwhile.
class TestResultData {
 public:
  TestResultData(absl::Time start_time,
                 std::vector<test_xml::TestCase> test_cases);

  bool operator==(const TestResultData& other) const = default;

  int64_t GetRanCount() const { return test_cases_.size(); }
  int64_t GetFailedCount() const;
  int64_t GetSkippedCount() const;
  bool DidAnyFail() const;

  // Wraps up the test cases we observed into a suites object which is the root
  // of XML-based reporting.
  test_xml::TestSuites ToXmlSuites(std::string_view module_name) const;

  void AddTestCase(test_xml::TestCase test_case) {
    test_cases_.push_back(std::move(test_case));
  }
  // When the test data is completed we can note the final test result (status)
  // and the duration it took.
  void Finish(TestResult result, absl::Duration duration) {
    result_ = result;
    duration_ = duration;
  }

  TestResult result() const { return result_; }
  const absl::Time& start_time() const { return start_time_; }
  const absl::Duration& duration() const { return duration_; }

 private:
  TestResult result_ = TestResult::kAllPassed;
  absl::Time start_time_;
  absl::Duration duration_ = absl::InfiniteDuration();
  std::vector<test_xml::TestCase> test_cases_;
};

// Parses program and run all tests contained inside.
//
// Args:
//   program: The program text to parse.
//   module_name: Name for the module.
//   filename: The filename from which "program" text originates.
//   dslx_paths: Additional paths at which we search for imported module files.
//   options: Bundles together optional arguments -- see ParseAndTestOptions.
//
// Returns:
//  Status of passing tests and associated pass/fail/skip counts.
absl::StatusOr<TestResultData> ParseAndTest(std::string_view program,
                                            std::string_view module_name,
                                            std::string_view filename,
                                            const ParseAndTestOptions& options);

struct ParseAndProveResult {
  TestResultData test_result_data;
  std::optional<std::vector<Value>> counterexample;
};

// Parses program and attempts to prove the given quickcheck property.
//
// Note that this call can only prove a single quickcheck function.
absl::StatusOr<ParseAndProveResult> ParseAndProve(
    std::string_view program, std::string_view module_name,
    std::string_view filename, std::string_view quickcheck_name,
    const ParseAndProveOptions& options);

struct QuickCheckResults {
  std::vector<std::vector<Value>> arg_sets;
  std::vector<Value> results;
};

// JIT-compiles the given xls_function and invokes it with num_tests randomly
// generated arguments -- returns `([argset, ...], [results, ...])` (i.e. in
// structure-of-array style).
//
// xls_function is a predicate we're trying to find evidence to falsify, so if
// this finds an example that falsifies the predicate, we early-return (i.e. the
// length of the returned vectors may be < 1000).
absl::StatusOr<QuickCheckResults> DoQuickCheck(
    xls::Function* xls_function, std::string_view ir_name,
    AbstractRunComparator* run_comparator, int64_t seed, int64_t num_tests);

}  // namespace xls::dslx

#endif  // XLS_DSLX_RUN_ROUTINES_H_
