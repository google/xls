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

#include <random>

#include "xls/dslx/bindings.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/typecheck.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/random_value.h"

namespace xls::dslx {
namespace {
// A few constants relating to the number of spaces to use in text formatting
// our test-runner output.
constexpr int kUnitSpaces = 7;
constexpr int kQuickcheckSpaces = 15;
}  // namespace

absl::StatusOr<IrJit*> RunComparator::GetOrCompileJitFunction(
    std::string ir_name, xls::Function* ir_function) {
  auto it = jit_cache_.find(ir_name);
  if (it != jit_cache_.end()) {
    return it->second.get();
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<IrJit> jit, IrJit::Create(ir_function));
  IrJit* result = jit.get();
  jit_cache_[ir_name] = std::move(jit);
  return result;
}

absl::Status RunComparator::RunComparison(
    Package* ir_package, bool requires_implicit_token, dslx::Function* f,
    absl::Span<InterpValue const> args,
    const SymbolicBindings* symbolic_bindings, const InterpValue& got) {
  XLS_RET_CHECK(ir_package != nullptr);

  XLS_ASSIGN_OR_RETURN(
      std::string ir_name,
      MangleDslxName(f->owner()->name(), f->identifier(),
                     requires_implicit_token ? CallingConvention::kImplicitToken
                                             : CallingConvention::kTypical,
                     f->GetFreeParametricKeySet(), symbolic_bindings));

  auto get_result = ir_package->GetFunction(ir_name);

  // The (converted) IR package does not include specializations of parametric
  // functions that are only called from test code, so not finding the function
  // may be benign.
  //
  // TODO(amfv): 2021-03-18 Extend IR conversion to include those functions.
  if (!get_result.ok()) {
    XLS_LOG(WARNING) << "Could not find " << ir_name
                     << " function for JIT comparison";
    return absl::OkStatus();
  }

  xls::Function* ir_function = get_result.value();

  XLS_ASSIGN_OR_RETURN(std::vector<Value> ir_args,
                       InterpValue::ConvertValuesToIr(args));

  // We need to know if the function-that-we're-doing-a-comparison-for needs an
  // implicit token.
  if (requires_implicit_token) {
    ir_args.insert(ir_args.begin(), Value::Bool(true));
    ir_args.insert(ir_args.begin(), Value::Token());
  }

  const char* mode_str = nullptr;
  Value ir_result;
  switch (mode_) {
    case CompareMode::kJit: {
      XLS_ASSIGN_OR_RETURN(IrJit * jit,
                           GetOrCompileJitFunction(ir_name, ir_function));
      XLS_ASSIGN_OR_RETURN(ir_result, jit->Run(ir_args));
      mode_str = "JIT";
      break;
    }
    case CompareMode::kInterpreter: {
      XLS_ASSIGN_OR_RETURN(ir_result, InterpretFunction(ir_function, ir_args));
      mode_str = "interpreter";
      break;
    }
  }

  if (requires_implicit_token) {
    // Slice off the first value.
    XLS_RET_CHECK(ir_result.element(0).IsToken());
    XLS_RET_CHECK_EQ(ir_result.size(), 2);
    Value real_ir_result = std::move(ir_result.element(1));
    ir_result = std::move(real_ir_result);
  }

  // Convert the interpreter value to an IR value so we can compare it.
  //
  // Note this conversion is lossy, but that's ok because we're just looking for
  // mismatches.
  XLS_ASSIGN_OR_RETURN(Value interp_ir_value, got.ConvertToIr());

  if (interp_ir_value != ir_result) {
    return absl::InternalError(
        absl::StrFormat("IR %s produced a different value from the DSL "
                        "interpreter for %s; IR %s: %s "
                        "DSL interpreter: %s",
                        mode_str, ir_function->name(), mode_str,
                        ir_result.ToString(), interp_ir_value.ToString()));
  }
  return absl::OkStatus();
}

static bool TestMatchesFilter(absl::string_view test_name,
                              absl::optional<absl::string_view> test_filter) {
  if (!test_filter.has_value()) {
    return true;
  }
  // TODO(leary): 2019-08-28 Implement wildcards.
  return test_name == *test_filter;
}

absl::StatusOr<QuickCheckResults> DoQuickCheck(xls::Function* xls_function,
                                               std::string ir_name,
                                               RunComparator* run_comparator,
                                               int64_t seed,
                                               int64_t num_tests) {
  XLS_ASSIGN_OR_RETURN(IrJit * jit, run_comparator->GetOrCompileJitFunction(
                                        std::move(ir_name), xls_function));

  QuickCheckResults results;
  std::minstd_rand rng_engine(seed);

  for (int i = 0; i < num_tests; i++) {
    results.arg_sets.push_back(
        RandomFunctionArguments(xls_function, &rng_engine));
    XLS_ASSIGN_OR_RETURN(xls::Value result, jit->Run(results.arg_sets.back()));
    results.results.push_back(result);
    if (result.IsAllZeros()) {
      // We were able to falsify the xls_function (predicate), bail out early
      // and present this evidence.
      break;
    }
  }

  return results;
}

static absl::Status RunQuickCheck(RunComparator* run_comparator,
                                  Package* ir_package, QuickCheck* quickcheck,
                                  TypeInfo* type_info, int64_t seed) {
  Function* fn = quickcheck->f();
  XLS_ASSIGN_OR_RETURN(std::string ir_name,
                       MangleDslxName(fn->owner()->name(), fn->identifier(),
                                      CallingConvention::kTypical,
                                      fn->GetFreeParametricKeySet()));
  XLS_ASSIGN_OR_RETURN(xls::Function * ir_function,
                       ir_package->GetFunction(ir_name));

  XLS_ASSIGN_OR_RETURN(
      QuickCheckResults qc_results,
      DoQuickCheck(ir_function, std::move(ir_name), run_comparator, seed,
                   quickcheck->test_count()));
  const auto& [arg_sets, results] = qc_results;
  XLS_ASSIGN_OR_RETURN(Bits last_result, results.back().GetBitsWithStatus());
  if (!last_result.IsZero()) {
    // Did not find a falsifying example.
    return absl::OkStatus();
  }

  const std::vector<Value>& last_argset = arg_sets.back();
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       type_info->GetItemAs<FunctionType>(fn));
  const std::vector<std::unique_ptr<ConcreteType>>& params = fn_type->params();

  std::vector<InterpValue> dslx_argset;
  for (int64_t i = 0; i < params.size(); ++i) {
    const ConcreteType& arg_type = *params[i];
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

using HandleError = const std::function<void(
    const absl::Status&, absl::string_view test_name, bool is_quickcheck)>;

static absl::Status RunQuickChecksIfJitEnabled(
    Module* entry_module, TypeInfo* type_info, RunComparator* run_comparator,
    Package* ir_package, absl::optional<int64_t> seed,
    const HandleError& handle_error) {
  if (run_comparator == nullptr) {
    std::cerr << "[ SKIPPING QUICKCHECKS  ] (JIT is disabled)" << std::endl;
    return absl::OkStatus();
  }
  if (!seed.has_value()) {
    // Note: we *want* to *provide* non-determinism by default. See
    // https://abseil.io/docs/cpp/guides/random#stability-of-generated-sequences
    // for rationale.
    seed = static_cast<int64_t>(getpid()) * static_cast<int64_t>(time(nullptr));
  }
  std::cerr << absl::StreamFormat("[ SEED %*d ]", kQuickcheckSpaces + 1, *seed)
            << std::endl;
  for (QuickCheck* quickcheck : entry_module->GetQuickChecks()) {
    const std::string& test_name = quickcheck->identifier();
    std::cerr << "[ RUN QUICKCHECK        ] " << test_name
              << " count: " << quickcheck->test_count() << std::endl;
    absl::Status status =
        RunQuickCheck(run_comparator, ir_package, quickcheck, type_info, *seed);
    if (!status.ok()) {
      handle_error(status, test_name, /*is_quickcheck=*/true);
    } else {
      std::cerr << "[                    OK ] " << test_name << std::endl;
    }
  }
  std::cerr << absl::StreamFormat(
                   "[=======================] %d quickcheck(s) ran.",
                   entry_module->GetQuickChecks().size())
            << std::endl;
  return absl::OkStatus();
}

absl::StatusOr<TestResult> ParseAndTest(absl::string_view program,
                                        absl::string_view module_name,
                                        absl::string_view filename,
                                        const ParseAndTestOptions& options) {
  int64_t ran = 0;
  int64_t failed = 0;
  int64_t skipped = 0;

  auto handle_error = [&](const absl::Status& status,
                          absl::string_view test_name, bool is_quickcheck) {
    XLS_VLOG(1) << "Handling error; status: " << status
                << " test_name: " << test_name;
    absl::StatusOr<PositionalErrorData> data_or =
        GetPositionalErrorData(status);
    std::string suffix;
    if (data_or.ok()) {
      const auto& data = data_or.value();
      XLS_CHECK_OK(PrintPositionalError(data.span, data.GetMessageWithType(),
                                        std::cerr));
    } else {
      // If we can't extract positional data we log the error and put the error
      // status into the "failed" prompted.
      XLS_LOG(ERROR) << "Internal error: " << status;
      suffix = absl::StrCat(": internal error: ", status.ToString());
    }
    std::string spaces((is_quickcheck ? kQuickcheckSpaces : kUnitSpaces), ' ');
    std::cerr << absl::StreamFormat("[ %sFAILED ] %s%s", spaces, test_name,
                                    suffix)
              << std::endl;
    failed += 1;
  };

  ImportData import_data;
  absl::StatusOr<TypecheckedModule> tm_or = ParseAndTypecheck(
      program, filename, module_name, &import_data, options.dslx_paths);
  if (!tm_or.ok()) {
    if (TryPrintError(tm_or.status())) {
      return TestResult::kSomeFailed;
    }
    return tm_or.status();
  }

  // If not executing tests and quickchecks, then return vacuous success.
  if (!options.execute) {
    return TestResult::kAllPassed;
  }

  Module* entry_module = tm_or.value().module;

  // If JIT comparisons are "on", we register a post-evaluation hook to compare
  // with the interpreter.
  std::unique_ptr<Package> ir_package;
  Interpreter::PostFnEvalHook post_fn_eval_hook;
  if (options.run_comparator != nullptr) {
    absl::StatusOr<std::unique_ptr<Package>> ir_package_or =
        ConvertModuleToPackage(entry_module, &import_data,
                               options.convert_options,
                               /*traverse_tests=*/true);
    if (!ir_package_or.ok()) {
      if (TryPrintError(ir_package_or.status())) {
        return TestResult::kSomeFailed;
      }
      return ir_package_or.status();
    }
    ir_package = std::move(ir_package_or).value();
    post_fn_eval_hook = [&ir_package, &import_data, &options](
                            Function* f, absl::Span<const InterpValue> args,
                            const SymbolicBindings* symbolic_bindings,
                            const InterpValue& got) -> absl::Status {
      absl::optional<bool> requires_implicit_token =
          import_data.GetRootTypeInfoForNode(f)
              .value()
              ->GetRequiresImplicitToken(f);
      XLS_RET_CHECK(requires_implicit_token.has_value());
      bool use_implicit_token = options.convert_options.emit_fail_as_assert &&
                                *requires_implicit_token;
      return options.run_comparator->RunComparison(ir_package.get(),
                                                   use_implicit_token, f, args,
                                                   symbolic_bindings, got);
    };
  }

  auto typecheck_callback = [&import_data, &options](Module* module) {
    return CheckModule(module, &import_data, options.dslx_paths);
  };

  Interpreter interpreter(entry_module, typecheck_callback, options.dslx_paths,
                          &import_data, options.trace_all,
                          options.trace_format_preference, post_fn_eval_hook);

  // Run unit tests.
  for (const std::string& test_name : entry_module->GetTestNames()) {
    if (!TestMatchesFilter(test_name, options.test_filter)) {
      skipped += 1;
      continue;
    }

    ran += 1;
    std::cerr << "[ RUN UNITTEST  ] " << test_name << std::endl;
    absl::Status status = interpreter.RunTest(test_name);
    if (status.ok()) {
      std::cerr << "[            OK ]" << std::endl;
    } else {
      handle_error(status, test_name, /*is_quickcheck=*/false);
    }
  }

  std::cerr << absl::StreamFormat(
                   "[===============] %d test(s) ran; %d failed; %d skipped.",
                   ran, failed, skipped)
            << std::endl;

  // Run quickchecks, but only if the JIT is enabled.
  if (!entry_module->GetQuickChecks().empty()) {
    XLS_RETURN_IF_ERROR(RunQuickChecksIfJitEnabled(
        entry_module, interpreter.current_type_info(), options.run_comparator,
        ir_package.get(), options.seed, handle_error));
  }

  return failed == 0 ? TestResult::kAllPassed : TestResult::kSomeFailed;
}

}  // namespace xls::dslx
