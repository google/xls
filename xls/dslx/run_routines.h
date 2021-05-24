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

#include "xls/dslx/interp_value.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/jit/ir_jit.h"

namespace xls::dslx {

// Indicates whether the RunComparator should be comparing to the JIT's IR
// execution or the IR interpreter's.
enum class CompareMode {
  kJit,
  kInterpreter,
};

// Helper object that is used as a post-execution hook in the interpreter,
// comparing interpreter results to results computed by the JIT to check that
// they're equivalent.
//
// Implementation note: slightly simpler to keep in object form so we can
// inspect cache state more easily than closing over it, e.g. for testing.
class RunComparator {
 public:
  explicit RunComparator(CompareMode mode) : mode_(mode) {}

  // Runs a comparison of the interpreter-determined value against the
  // JIT-determined value.
  absl::Status RunComparison(Package* ir_package, bool requires_implicit_token,
                             Function* f, absl::Span<InterpValue const> args,
                             const SymbolicBindings* symbolic_bindings,
                             const InterpValue& got);

  // Returns the cached or newly-compiled jit function for ir_name.  ir_name has
  // already been mangled (see MangleDslxName) so it should be unique in the
  // program and is used as the cache key.
  //
  // Note: There is no locking in jit compilation or on the jit function cache
  // so this function is *not* thread-safe.
  absl::StatusOr<IrJit*> GetOrCompileJitFunction(std::string ir_name,
                                                 xls::Function* ir_function);

 private:
  friend class RunRoutinesTest_TestInvokedFunctionDoesJit_Test;
  friend class RunRoutinesTest_QuickcheckInvokedFunctionDoesJit_Test;
  friend class RunRoutinesTest_NoSeedStillQuickChecks_Test;

  absl::flat_hash_map<std::string, std::unique_ptr<IrJit>> jit_cache_;
  CompareMode mode_;
};

// Optional arguments to ParseAndTest (that have sensible defaults).
//
//   test_filter: Test filter specification (e.g. as passed from bazel test
//     environment).
//   trace_all: Whether or not to trace all expressions.
//   run_comparator: Optional object that can compare DSLX interpreter
//    executions with a reference (e.g. IR execution).
//   execute: Whether or not to execute the quickchecks and tests.
//   seed: Seed for QuickCheck random input stimulus.
//   convert_options: Options used in IR conversion, see `ConvertOptions` for
//    details.
struct ParseAndTestOptions {
  absl::Span<const std::filesystem::path> dslx_paths = {};
  absl::optional<absl::string_view> test_filter = absl::nullopt;
  bool trace_all = false;
  FormatPreference trace_format_preference = FormatPreference::kDefault;
  RunComparator* run_comparator = nullptr;
  bool execute = true;
  absl::optional<int64_t> seed = absl::nullopt;
  ConvertOptions convert_options;
};

enum class TestResult {
  kSomeFailed,
  kAllPassed,
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
//   Whether any test failed (as a boolean).
absl::StatusOr<TestResult> ParseAndTest(absl::string_view program,
                                        absl::string_view module_name,
                                        absl::string_view filename,
                                        const ParseAndTestOptions& options);

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
absl::StatusOr<QuickCheckResults> DoQuickCheck(xls::Function* xls_function,
                                               std::string ir_name,
                                               RunComparator* run_comparator,
                                               int64_t seed, int64_t num_tests);

}  // namespace xls::dslx

#endif  // XLS_DSLX_RUN_ROUTINES_H_
