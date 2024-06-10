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

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_cache.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/test_xml.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/solvers/z3_ir_translator.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {
// A few constants relating to the number of spaces to use in text formatting
// our test-runner output.
constexpr int kUnitSpaces = 7;
constexpr int kQuickcheckSpaces = 15;

void HandleError(TestResultData& result, const absl::Status& status,
                 std::string_view test_name, const Pos& start_pos,
                 const absl::Time& start, const absl::Duration& duration,
                 bool is_quickcheck) {
  VLOG(1) << "Handling error; status: " << status
          << " test_name: " << test_name;
  absl::StatusOr<PositionalErrorData> data_or = GetPositionalErrorData(status);

  std::string one_liner;
  std::string suffix;
  if (data_or.ok()) {
    const auto& data = data_or.value();
    CHECK_OK(PrintPositionalError(
        data.span, data.GetMessageWithType(), std::cerr,
        /*get_file_contents=*/nullptr, PositionalErrorColor::kErrorColor));
    one_liner = data.GetMessageWithType();
  } else {
    // If we can't extract positional data we log the error and put the error
    // status into the "failed" prompted.
    LOG(ERROR) << "Internal error: " << status;
    suffix = absl::StrCat(": internal error: ", status.ToString());
    one_liner = suffix;
  }

  // Add to test tracking data.
  result.AddTestCase(
      test_xml::TestCase{.name = std::string(test_name),
                         .file = start_pos.filename(),
                         .line = start_pos.GetHumanLineno(),
                         .status = test_xml::RunStatus::kRun,
                         .result = test_xml::RunResult::kCompleted,
                         .time = duration,
                         .timestamp = start,
                         .failure = test_xml::Failure{.message = one_liner}});

  std::string spaces((is_quickcheck ? kQuickcheckSpaces : kUnitSpaces), ' ');
  std::cerr << absl::StreamFormat("[ %sFAILED ] %s%s", spaces, test_name,
                                  suffix)
            << "\n";
};

absl::Status RunTestFunction(ImportData* import_data, TypeInfo* type_info,
                             Module* module, TestFunction* tf,
                             const BytecodeInterpreterOptions& options) {
  auto cache = std::make_unique<BytecodeCache>(import_data);
  import_data->SetBytecodeCache(std::move(cache));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(
          import_data, type_info, tf->fn(), std::nullopt,
          BytecodeEmitterOptions{.format_preference =
                                     options.format_preference()}));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{},
                                        options)
      .status();
}

absl::Status RunTestProc(ImportData* import_data, TypeInfo* type_info,
                         Module* module, TestProc* tp,
                         const BytecodeInterpreterOptions& options) {
  auto cache = std::make_unique<BytecodeCache>(import_data);
  import_data->SetBytecodeCache(std::move(cache));

  XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                       type_info->GetTopLevelProcTypeInfo(tp->proc()));

  std::vector<ProcInstance> proc_instances;
  XLS_ASSIGN_OR_RETURN(InterpValue terminator,
                       ti->GetConstExpr(tp->proc()->config().params()[0]));
  XLS_RETURN_IF_ERROR(ProcConfigBytecodeInterpreter::InitializeProcNetwork(
      import_data, ti, tp->proc(), terminator, &proc_instances, options));

  std::shared_ptr<InterpValue::Channel> term_chan =
      terminator.GetChannelOrDie();
  int64_t tick_count = 0;
  while (term_chan->empty()) {
    bool progress_made = false;
    if (options.max_ticks().has_value() &&
        tick_count > options.max_ticks().value()) {
      return absl::DeadlineExceededError(
          absl::StrFormat("Exceeded limit of %d proc ticks before terminating",
                          options.max_ticks().value()));
    }

    std::vector<std::string> blocked_channels;
    for (auto& p : proc_instances) {
      XLS_ASSIGN_OR_RETURN(ProcRunResult run_result, p.Run());
      if (run_result.execution_state == ProcExecutionState::kBlockedOnReceive) {
        XLS_RET_CHECK(run_result.blocked_channel_name.has_value());
        blocked_channels.push_back(run_result.blocked_channel_name.value());
      }
      progress_made |= run_result.progress_made;
    }

    if (!progress_made) {
      return absl::DeadlineExceededError(
          absl::StrFormat("Procs are deadlocked. Blocked channels: %s",
                          absl::StrJoin(blocked_channels, ", ")));
    }
    ++tick_count;
  }

  InterpValue ret_val = term_chan->front();
  XLS_RET_CHECK(ret_val.IsBool());
  if (!ret_val.IsTrue()) {
    return FailureErrorStatus(tp->proc()->span(),
                              "Proc reported failure upon exit.");
  }
  return absl::OkStatus();
}

}  // namespace

TestResultData::TestResultData(absl::Time start_time,
                               std::vector<test_xml::TestCase> test_cases)
    : start_time_(start_time), test_cases_(std::move(test_cases)) {}

int64_t TestResultData::GetFailedCount() const {
  return std::count_if(
      test_cases_.begin(), test_cases_.end(),
      [](const auto& test_case) { return test_case.failure.has_value(); });
}
int64_t TestResultData::GetSkippedCount() const {
  return std::count_if(
      test_cases_.begin(), test_cases_.end(), [](const auto& test_case) {
        return test_case.result == test_xml::RunResult::kFiltered;
      });
}

bool TestResultData::DidAnyFail() const {
  return std::any_of(
      test_cases_.begin(), test_cases_.end(),
      [](const auto& test_case) { return test_case.failure.has_value(); });
}

test_xml::TestSuites TestResultData::ToXmlSuites(
    std::string_view module_name) const {
  test_xml::TestCounts counts = {
      .tests = static_cast<int64_t>(test_cases_.size()),
      .failures = GetFailedCount(),
      .disabled = 0,
      .skipped = GetSkippedCount(),
      .errors = 0,
  };
  test_xml::TestSuites suites = {
      .counts = counts,
      .time = duration_,
      .timestamp = start_time_,
      .test_suites =
          {
              // We currently consider all the test cases inside of a single
              // file to be part of one suite.
              //
              // TODO(leary): 2024-02-08 We may want to break out quickcheck
              // tests vs
              // unit tests in the future.
              test_xml::TestSuite{
                  .name = absl::StrCat(module_name, " tests"),
                  .counts = counts,
                  .time = duration_,
                  .timestamp = start_time_,
                  .test_cases = test_cases_,
              },
          },
  };
  return suites;
}

static bool TestMatchesFilter(std::string_view test_name,
                              const RE2* test_filter) {
  if (test_filter == nullptr) {
    // All tests vacuously match the filter if there is no filter (i.e. we run
    // them all).
    return true;
  }
  return RE2::FullMatch(test_name, *test_filter);
}

absl::StatusOr<QuickCheckResults> DoQuickCheck(
    xls::Function* xls_function, std::string_view ir_name,
    AbstractRunComparator* run_comparator, int64_t seed, int64_t num_tests) {
  QuickCheckResults results;
  std::minstd_rand rng_engine(seed);

  for (int i = 0; i < num_tests; i++) {
    results.arg_sets.push_back(
        RandomFunctionArguments(xls_function, rng_engine));
    // TODO(https://github.com/google/xls/issues/506): 2021-10-15
    // Assertion failures should work out, but we should consciously decide
    // if/how we want to dump traces when running QuickChecks (always, for
    // failures, flag-controlled, ...).
    XLS_ASSIGN_OR_RETURN(xls::Value result,
                         DropInterpreterEvents(run_comparator->RunIrFunction(
                             ir_name, xls_function, results.arg_sets.back())));

    // In the case of an implicit token signature we get (token, bool) as the
    // result of the quickcheck'd function, so we unbox the boolean here.
    if (result.IsTuple()) {
      result = result.elements()[1];
      XLS_RET_CHECK(result.IsBits());
    }

    results.results.push_back(result);

    if (result.IsAllZeros()) {
      // We were able to falsify the xls_function (predicate), bail out early
      // and present this evidence.
      break;
    }
  }

  return results;
}

struct QuickcheckIrFn {
  std::string ir_name;
  xls::Function* ir_function;
};

static absl::StatusOr<QuickcheckIrFn> FindQuickcheckIrFn(Function* fn,
                                                         Package* ir_package) {
  // First we try to get the version of the function that doesn't need a token.
  XLS_ASSIGN_OR_RETURN(std::string ir_name,
                       MangleDslxName(fn->owner()->name(), fn->identifier(),
                                      CallingConvention::kTypical,
                                      fn->GetFreeParametricKeySet()));
  std::optional<xls::Function*> maybe_ir_function =
      ir_package->TryGetFunction(ir_name);
  if (maybe_ir_function.has_value()) {
    return QuickcheckIrFn{ir_name, maybe_ir_function.value()};
  }

  XLS_ASSIGN_OR_RETURN(ir_name,
                       MangleDslxName(fn->owner()->name(), fn->identifier(),
                                      CallingConvention::kImplicitToken,
                                      fn->GetFreeParametricKeySet()));
  maybe_ir_function = ir_package->TryGetFunction(ir_name);
  if (maybe_ir_function.has_value()) {
    return QuickcheckIrFn{ir_name, maybe_ir_function.value()};
  }
  return absl::InternalError(
      absl::StrFormat("Could not find DSLX quickcheck function `%s` in IR "
                      "package `%s`; available IR functions: [%s]",
                      fn->identifier(), ir_package->name(),
                      absl::StrJoin(ir_package->GetFunctionNames(), ", ")));
}

static absl::Status RunQuickCheck(AbstractRunComparator* run_comparator,
                                  Package* ir_package, QuickCheck* quickcheck,
                                  TypeInfo* type_info, int64_t seed) {
  // Note: DSLX function.
  Function* fn = quickcheck->f();

  XLS_ASSIGN_OR_RETURN(QuickcheckIrFn qc_fn,
                       FindQuickcheckIrFn(fn, ir_package));

  XLS_ASSIGN_OR_RETURN(
      QuickCheckResults qc_results,
      DoQuickCheck(qc_fn.ir_function, qc_fn.ir_name, run_comparator, seed,
                   quickcheck->GetTestCountOrDefault()));
  const auto& [arg_sets, results] = qc_results;
  XLS_ASSIGN_OR_RETURN(Bits last_result, results.back().GetBitsWithStatus());
  if (!last_result.IsZero()) {
    // Did not find a falsifying example.
    return absl::OkStatus();
  }

  const std::vector<Value>& last_argset = arg_sets.back();
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       type_info->GetItemAs<FunctionType>(fn));
  const std::vector<std::unique_ptr<Type>>& params = fn_type->params();

  std::vector<InterpValue> dslx_argset;
  for (int64_t i = 0; i < params.size(); ++i) {
    const Type& arg_type = *params[i];
    const Value& value = last_argset[i];
    XLS_ASSIGN_OR_RETURN(InterpValue interp_value,
                         ValueToInterpValue(value, &arg_type));
    dslx_argset.push_back(interp_value);
  }
  std::string dslx_argset_str = absl::StrJoin(
      dslx_argset, ", ", [](std::string* out, const InterpValue& v) {
        absl::StrAppend(out, v.ToString());
      });
  return FailureErrorStatus(
      fn->span(),
      absl::StrFormat("Found falsifying example after %d tests: [%s]",
                      results.size(), dslx_argset_str));
}

static absl::Status RunQuickChecksIfJitEnabled(
    const RE2* test_filter, Module* entry_module, TypeInfo* type_info,
    AbstractRunComparator* run_comparator, Package* ir_package,
    std::optional<int64_t> seed, TestResultData& result) {
  if (run_comparator == nullptr) {
    // TODO(leary): 2024-02-08 Note that this skips /all/ the quickchecks so we
    // don't make an entry for it right now in the test XML.
    std::cerr << "[ SKIPPING QUICKCHECKS  ] (JIT is disabled)" << "\n";
    return absl::OkStatus();
  }
  if (!seed.has_value()) {
    // Note: we *want* to *provide* non-determinism by default. See
    // https://abseil.io/docs/cpp/guides/random#stability-of-generated-sequences
    // for rationale.
    seed = static_cast<int64_t>(getpid()) * static_cast<int64_t>(time(nullptr));
  }
  bool any_quicktest_run = false;
  for (QuickCheck* quickcheck : entry_module->GetQuickChecks()) {
    const std::string& quickcheck_name = quickcheck->identifier();
    const Pos& start_pos = quickcheck->span().start();
    const absl::Time test_case_start = absl::Now();
    if (!TestMatchesFilter(quickcheck_name, test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(
          test_xml::TestCase{.name = quickcheck_name,
                             .file = start_pos.filename(),
                             .line = start_pos.GetHumanLineno(),
                             .status = test_xml::RunStatus::kRun,
                             .result = test_xml::RunResult::kFiltered,
                             .time = test_case_end - test_case_start,
                             .timestamp = test_case_start});
      continue;
    }

    if (!any_quicktest_run) {
      // Only print the SEED if there is actually a test that is executed.
      std::cerr << absl::StreamFormat("[ SEED %*d ]\n", kQuickcheckSpaces + 1,
                                      *seed);
      any_quicktest_run = true;
    }
    std::cerr << "[ RUN QUICKCHECK        ] " << quickcheck_name
              << " count: " << quickcheck->GetTestCountOrDefault() << "\n";
    const absl::Status status =
        RunQuickCheck(run_comparator, ir_package, quickcheck, type_info, *seed);
    const absl::Duration duration = absl::Now() - test_case_start;
    if (!status.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  duration,
                  /*is_quickcheck=*/true);
    } else {
      result.AddTestCase(
          test_xml::TestCase{.name = quickcheck_name,
                             .file = start_pos.filename(),
                             .line = start_pos.GetHumanLineno(),
                             .status = test_xml::RunStatus::kRun,
                             .result = test_xml::RunResult::kCompleted,
                             .time = duration,
                             .timestamp = test_case_start});
      std::cerr << "[                    OK ] " << quickcheck_name << "\n";
    }
  }
  std::cerr << absl::StreamFormat(
                   "[=======================] %d quickcheck(s) ran.",
                   entry_module->GetQuickChecks().size())
            << "\n";
  return absl::OkStatus();
}

absl::StatusOr<ParseAndProveResult> ParseAndProve(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndProveOptions& options) {
  const absl::Time start = absl::Now();
  TestResultData result(start, /*test_cases=*/{});

  auto import_data = CreateImportData(options.stdlib_path, options.dslx_paths,
                                      options.warnings);
  absl::StatusOr<TypecheckedModule> tm_or =
      ParseAndTypecheck(program, filename, module_name, &import_data);
  if (!tm_or.ok()) {
    if (TryPrintError(tm_or.status())) {
      result.Finish(TestResult::kParseOrTypecheckError, absl::Now() - start);
      return ParseAndProveResult{.test_result_data = result};
    }
    return tm_or.status();
  }

  // If we're not executing, then we're just scanning for errors -- if warnings
  // are *not* errors, just elide printing them (or e.g. we'd show warnings for
  // files that had warnings suppressed at build time, which would gunk up build
  // logs unnecessarily.).
  if (options.warnings_as_errors) {
    PrintWarnings(tm_or->warnings);
  }

  if (options.warnings_as_errors && !tm_or->warnings.warnings().empty()) {
    result.Finish(TestResult::kFailedWarnings, absl::Now() - start);
    return ParseAndProveResult{.test_result_data = result};
  }

  Module* entry_module = tm_or.value().module;

  // We need to IR-convert the quickcheck property and then try to prove that
  // the return value is always true.
  absl::flat_hash_map<std::string, QuickCheck*> qcs =
      entry_module->GetQuickCheckByName();

  // Counter-examples map from failing test name -> counterexample values.
  absl::flat_hash_map<std::string, std::vector<Value>> counterexamples;

  for (const std::string& quickcheck_name :
       entry_module->GetQuickCheckNames()) {
    QuickCheck* quickcheck = qcs.at(quickcheck_name);
    const Pos& start_pos = quickcheck->span().start();
    Function* f = quickcheck->f();
    VLOG(1) << "Found quickcheck function: " << f->identifier();
    std::cerr << "[ RUN QUICKCHECK        ] " << quickcheck_name << '\n';
    absl::Status status;

    auto test_case_start = absl::Now();

    if (!TestMatchesFilter(quickcheck_name, options.test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(
          test_xml::TestCase{.name = quickcheck_name,
                             .file = start_pos.filename(),
                             .line = start_pos.GetHumanLineno(),
                             .status = test_xml::RunStatus::kRun,
                             .result = test_xml::RunResult::kFiltered,
                             .time = test_case_end - test_case_start,
                             .timestamp = test_case_start});
      continue;
    }
    dslx::PackageConversionData conv{
        .package = std::make_unique<Package>(entry_module->name())};
    Package& package = *conv.package;

    status = ConvertOneFunctionIntoPackage(entry_module, f, &import_data,
                                           /*parametric_env=*/nullptr,
                                           ConvertOptions{}, &conv);
    if (!status.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - start, /*is_quickcheck=*/true);
      continue;
    }

    // Note: we need this to eliminate unoptimized IR constructs that are not
    // currently handled for translation; e.g. bounded-for-loops and non-inlined
    // function calls.
    status = RunOptimizationPassPipeline(&package).status();
    if (!status.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - start, /*is_quickcheck=*/true);
      continue;
    }

    absl::StatusOr<std::string> ir_function_name_or = MangleDslxName(
        entry_module->name(), f->identifier(), CallingConvention::kTypical);
    if (!ir_function_name_or.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - start, /*is_quickcheck=*/true);
      continue;
    }

    absl::StatusOr<xls::Function*> ir_function_or =
        package.GetFunction(ir_function_name_or.value());
    if (!ir_function_or.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - start, /*is_quickcheck=*/true);
      continue;
    }

    VLOG(1) << "Found IR function: " << ir_function_or.value()->name();

    absl::StatusOr<solvers::z3::ProverResult> proven_or = solvers::z3::TryProve(
        ir_function_or.value(), ir_function_or.value()->return_value(),
        solvers::z3::Predicate::NotEqualToZero(), absl::InfiniteDuration());

    if (!proven_or.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - start, /*is_quickcheck=*/true);
      continue;
    }

    VLOG(1) << "Proven? "
            << (std::holds_alternative<solvers::z3::ProvenTrue>(
                    proven_or.value())
                    ? "true"
                    : "false");

    if (std::holds_alternative<solvers::z3::ProvenTrue>(proven_or.value())) {
      absl::Time test_case_end = absl::Now();
      absl::Duration duration = test_case_end - test_case_start;
      result.AddTestCase(test_xml::TestCase{
          .name = std::string(quickcheck_name),
          .file = start_pos.filename(),
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kCompleted,
          .time = duration,
          .timestamp = test_case_start,
      });
      std::cerr << "[                    OK ] " << quickcheck_name << "\n";
      continue;
    }

    const auto& proven_false =
        std::get<solvers::z3::ProvenFalse>(proven_or.value());

    // Extract the counterexample, and collapse it back into sequential order.
    std::vector<Value> counterexample;
    using ParamValues = absl::flat_hash_map<const xls::Param*, Value>;
    XLS_ASSIGN_OR_RETURN(ParamValues counterexample_map,
                         proven_false.counterexample);
    for (const xls::Param* param : ir_function_or.value()->params()) {
      counterexample.push_back(counterexample_map[param]);
    }
    std::string one_liner =
        absl::StrCat("counterexample: ", absl::StrJoin(counterexample, ", "));
    status = ProofErrorStatus(quickcheck->span(), one_liner);
    counterexamples[quickcheck_name] = std::move(counterexample);
    absl::Time test_case_end = absl::Now();
    absl::Duration duration = test_case_end - test_case_start;
    HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                duration, /*is_quickcheck=*/true);
  }

  result.Finish(TestResult::kSomeFailed, absl::Now() - start);
  std::cerr
      << absl::StreamFormat(
             "[=======================] %d test(s) ran; %d failed; %d skipped.",
             result.GetRanCount(), result.GetFailedCount(),
             result.GetSkippedCount())
      << '\n';

  result.Finish(
      result.DidAnyFail() ? TestResult::kSomeFailed : TestResult::kAllPassed,
      absl::Now() - start);
  return ParseAndProveResult{.test_result_data = std::move(result),
                             .counterexamples = std::move(counterexamples)};
}

absl::StatusOr<TestResultData> ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) {
  const absl::Time start = absl::Now();
  TestResultData result(start, /*test_cases=*/{});

  auto import_data = CreateImportData(options.stdlib_path, options.dslx_paths,
                                      options.warnings);

  absl::StatusOr<TypecheckedModule> tm_or =
      ParseAndTypecheck(program, filename, module_name, &import_data);
  if (!tm_or.ok()) {
    if (TryPrintError(tm_or.status())) {
      result.Finish(TestResult::kParseOrTypecheckError, absl::Now() - start);
      return result;
    }
    return tm_or.status();
  }

  // If we're not executing, then we're just scanning for errors -- if warnings
  // are *not* errors, just elide printing them (or e.g. we'd show warnings for
  // files that had warnings suppressed at build time, which would gunk up build
  // logs unnecessarily.).
  if (options.execute || options.warnings_as_errors) {
    PrintWarnings(tm_or->warnings);
  }

  if (options.warnings_as_errors && !tm_or->warnings.warnings().empty()) {
    result.Finish(TestResult::kFailedWarnings, absl::Now() - start);
    return result;
  }

  // If not executing tests and quickchecks, then return vacuous success.
  if (!options.execute) {
    result.Finish(TestResult::kAllPassed, absl::Now() - start);
    return result;
  }

  Module* entry_module = tm_or.value().module;

  // If JIT comparisons are "on", we register a post-evaluation hook to compare
  // with the interpreter.
  std::unique_ptr<Package> ir_package;
  PostFnEvalHook post_fn_eval_hook;
  if (options.run_comparator != nullptr) {
    absl::StatusOr<dslx::PackageConversionData> ir_package_or =
        ConvertModuleToPackage(entry_module, &import_data,
                               options.convert_options);
    if (!ir_package_or.ok()) {
      if (TryPrintError(ir_package_or.status())) {
        result.Finish(TestResult::kSomeFailed, absl::Now() - start);
        return result;
      }
      return ir_package_or.status();
    }
    ir_package = std::move(ir_package_or).value().package;
    post_fn_eval_hook = [&ir_package, &import_data, &options](
                            const Function* f,
                            absl::Span<const InterpValue> args,
                            const ParametricEnv* parametric_env,
                            const InterpValue& got) -> absl::Status {
      XLS_RET_CHECK(f != nullptr);
      std::optional<bool> requires_implicit_token =
          import_data.GetRootTypeInfoForNode(f)
              .value()
              ->GetRequiresImplicitToken(*f);
      XLS_RET_CHECK(requires_implicit_token.has_value());
      return options.run_comparator->RunComparison(ir_package.get(),
                                                   *requires_implicit_token, f,
                                                   args, parametric_env, got);
    };
  }

  // Run unit tests.
  for (const std::string& test_name : entry_module->GetTestNames()) {
    auto test_case_start = absl::Now();
    ModuleMember* member = entry_module->FindMemberWithName(test_name).value();
    const Pos start_pos = GetPos(*member);

    if (!TestMatchesFilter(test_name, options.test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(
          test_xml::TestCase{.name = test_name,
                             .file = start_pos.filename(),
                             .line = start_pos.GetHumanLineno(),
                             .status = test_xml::RunStatus::kRun,
                             .result = test_xml::RunResult::kFiltered,
                             .time = test_case_end - test_case_start,
                             .timestamp = test_case_start});
      continue;
    }

    std::cerr << "[ RUN UNITTEST  ] " << test_name << '\n';
    absl::Status status;
    BytecodeInterpreterOptions interpreter_options;
    interpreter_options.post_fn_eval_hook(post_fn_eval_hook)
        .trace_hook(InfoLoggingTraceHook)
        .trace_channels(options.trace_channels)
        .max_ticks(options.max_ticks)
        .format_preference(options.format_preference);
    if (std::holds_alternative<TestFunction*>(*member)) {
      XLS_ASSIGN_OR_RETURN(TestFunction * tf, entry_module->GetTest(test_name));
      status = RunTestFunction(&import_data, tm_or.value().type_info,
                               entry_module, tf, interpreter_options);
    } else {
      XLS_ASSIGN_OR_RETURN(TestProc * tp, entry_module->GetTestProc(test_name));
      status = RunTestProc(&import_data, tm_or.value().type_info, entry_module,
                           tp, interpreter_options);
    }
    auto test_case_end = absl::Now();

    if (status.ok()) {
      // Add to the tracking data.
      result.AddTestCase(
          test_xml::TestCase{.name = test_name,
                             .file = start_pos.filename(),
                             .line = start_pos.GetHumanLineno(),
                             .status = test_xml::RunStatus::kRun,
                             .result = test_xml::RunResult::kCompleted,
                             .time = test_case_end - test_case_start,
                             .timestamp = test_case_start});
      std::cerr << "[            OK ]" << '\n';
    } else {
      HandleError(result, status, test_name, start_pos, test_case_start,
                  test_case_end - test_case_start,
                  /*is_quickcheck=*/false);
    }
  }

  std::cerr << absl::StreamFormat(
                   "[===============] %d test(s) ran; %d failed; %d skipped.",
                   result.GetRanCount(), result.GetFailedCount(),
                   result.GetSkippedCount())
            << '\n';

  // Run quickchecks, but only if the JIT is enabled.
  if (!entry_module->GetQuickChecks().empty()) {
    XLS_RETURN_IF_ERROR(RunQuickChecksIfJitEnabled(
        options.test_filter, entry_module, tm_or.value().type_info,
        options.run_comparator, ir_package.get(), options.seed, result));
  }

  result.Finish(
      result.DidAnyFail() ? TestResult::kSomeFailed : TestResult::kAllPassed,
      absl::Now() - start);
  return result;
}

std::string_view TestResultToString(TestResult tr) {
  switch (tr) {
    case TestResult::kFailedWarnings:
      return "failed-warnings";
    case TestResult::kSomeFailed:
      return "some-failed";
    case TestResult::kAllPassed:
      return "all-passed";
    case TestResult::kParseOrTypecheckError:
      return "parse-or-typecheck-error";
  }
  LOG(FATAL) << "Invalid test result value: " << static_cast<int>(tr);
}

}  // namespace xls::dslx
