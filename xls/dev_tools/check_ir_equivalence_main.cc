// Copyright 2020 The XLS Authors
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
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/optional_ref.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/dev_tools/tool_timeout.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/ir_equivalence.h"
#include "xls/solvers/solver.h"

static constexpr std::string_view kUsage = R"(
Verifies that the two provided XLS IR files are logically equivalent; that is,
they produce the same outputs across all inputs. The most common usage is to
prove that optimizations are safe - that they don't change program outputs.

Example invocation:
  check_ir_equivalence_main <IR file> <IR file>

If there are multiple functions in the specified files, then it's _strongly_
recommended that you specify --top to ensure that the right functions are
compared. If the tool picks the wrong one, a crash may result.

If the top is a proc then the --activation_count flag must be passed.

Exits with code --mismatch_exit_code if equivalence is not found.
)";

// LINT.IfChange
ABSL_FLAG(std::string, top, "",
          "The top entity to check. If unspecified, an attempt will be made"
          "to find and check a top entity for the package. Currently, only"
          "Functions are supported.");
ABSL_FLAG(std::optional<int64_t>, activation_count, std::nullopt,
          "How many activations to check proc equivalence for. This must be "
          "passed if top is a proc.");
ABSL_FLAG(int, mismatch_exit_code, 255,
          "Value to exit with if equivalence is not proven.");
ABSL_FLAG(int, match_exit_code, 0,
          "Value to exit with if equivalence is not proven.");
ABSL_FLAG(xls::solvers::SolverKind, solver, xls::solvers::SolverKind::kZ3,
          "Solver backend to use (z3 or bitwuzla).");
ABSL_FLAG(bool, optimize_merged_ir, false,
          "If true, run the standard optimization pass pipeline on the merged "
          "equivalence-checking IR before invoking the solver.");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

ABSL_FLAG(bool, print_timing, false,
          "If true, print phase wall-clock timings to stderr.");

namespace xls {
namespace {

absl::StatusOr<solvers::ProverResult> CheckFunctionEquivalence(
    Function* f1, Function* f2, solvers::SolverKind solver,
    bool optimize_merged_ir,
    absl::optional_ref<solvers::EquivalenceTiming> timing) {
  std::function<absl::Status(Package*)> transform_merged;
  if (optimize_merged_ir) {
    transform_merged = [](Package* pkg) -> absl::Status {
      return RunOptimizationPassPipeline(pkg).status();
    };
  }
  return solvers::TryProveEquivalence(f1, f2, /*ignore_asserts=*/false, solver,
                                      /*limit=*/{}, transform_merged, timing);
}
absl::StatusOr<solvers::ProverResult> CheckProcEquivalence(
    Proc* p1, Proc* p2, int64_t activation_count, solvers::SolverKind solver,
    bool optimize_merged_ir,
    absl::optional_ref<solvers::EquivalenceTiming> timing) {
  XLS_ASSIGN_OR_RETURN(
      Function * f1,
      UnrollProcToFunction(p1, activation_count, /*include_state=*/false),
      _ << "Unable to unroll: " << p1->DumpIr());
  XLS_ASSIGN_OR_RETURN(
      Function * f2,
      UnrollProcToFunction(p2, activation_count, /*include_state=*/false),
      _ << "Unable to unroll: " << p2->DumpIr());
  return CheckFunctionEquivalence(f1, f2, solver, optimize_merged_ir, timing);
}

absl::StatusOr<std::vector<std::string>> CounterexampleParams(
    FunctionBase* f, const solvers::ProvenFalse& proven_false) {
  std::vector<std::string> counterexample;
  using InputValues = absl::flat_hash_map<Node*, Value>;
  XLS_ASSIGN_OR_RETURN(InputValues counterexample_map,
                       proven_false.counterexample);
  if (f->IsFunction()) {
    for (const xls::Param* param : f->params()) {
      bool missing = true;
      for (const auto& [counterexample_param, value] : counterexample_map) {
        if (counterexample_param->GetName() == param->name()) {
          missing = false;
          counterexample.push_back(value.ToString(FormatPreference::kHex));
          break;
        }
      }
      if (missing) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Counterexample includes no value for param: ", param->name()));
      }
    }
  } else {
    // TODO(allight): Print these out better.
    for (const auto& [param, val] : counterexample_map) {
      counterexample.push_back(absl::StrFormat(
          "%s -> %s", param->ToString(), val.ToString(FormatPreference::kHex)));
    }
    absl::c_sort(counterexample);
  }
  return counterexample;
}

absl::StatusOr<bool> RealMain(const std::vector<std::string_view>& ir_paths,
                              const std::string& entry,
                              std::optional<int64_t> activation_count,
                              solvers::SolverKind solver,
                              bool optimize_merged_ir, bool print_timing) {
  auto timeout = StartTimeoutTimer();
  Stopwatch total_sw;
  absl::Duration inlining_duration = absl::ZeroDuration();
  solvers::EquivalenceTiming equiv_timing{};
  absl::Cleanup print_timing_cleanup([&]() {
    if (print_timing) {
      double inlining_ms = absl::ToDoubleMilliseconds(inlining_duration);
      double merge_ms = absl::ToDoubleMilliseconds(equiv_timing.merge_duration);
      double opt_pipeline_ms =
          absl::ToDoubleMilliseconds(equiv_timing.transform_duration);
      double merge_and_opt_ms = merge_ms + opt_pipeline_ms;
      double solver_ms =
          absl::ToDoubleMilliseconds(equiv_timing.solver_duration);
      double total_ms = absl::ToDoubleMilliseconds(total_sw.GetElapsedTime());
      std::cerr << absl::StreamFormat(
          "timing: inlining_ms=%.3f merge_ms=%.3f opt_pipeline_ms=%.3f "
          "merge_and_opt_ms=%.3f solver_ms=%.3f total_ms=%.3f\n",
          inlining_ms, merge_ms, opt_pipeline_ms, merge_and_opt_ms, solver_ms,
          total_ms);
    }
  });

  std::vector<std::unique_ptr<Package>> packages;
  for (const auto ir_path : ir_paths) {
    XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
    XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
    if (!entry.empty()) {
      XLS_RETURN_IF_ERROR(package->SetTopByName(entry));
    }
    packages.push_back(std::move(package));
  }

  // Z3 doesn't handle directly-specified (as AST nodes) subroutines - there is
  // some support for recursive functions (with bodies), but it hasn't behaved
  // like we've expected.
  //
  // To work around this, we have to inline such calls.
  // Fortunately, inlining is pretty simple and unlikely to change semantics.
  Stopwatch inlining_sw;
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<OptimizationPass> inlining_passes,
      // NB All except full-inlining and dce are defined in this file.
      GetOptimizationPipelineGenerator().GeneratePipeline(
          "full-inlining proc_state_legalization_shim dce literalize_zero_bits "
          "dce assert_and_cover_removal dce"));
  OptimizationPassOptions options;
  PassResults results;
  OptimizationContext context;
  for (const auto& package : packages) {
    XLS_RETURN_IF_ERROR(
        inlining_passes->Run(package.get(), options, &results, context)
            .status());
  }
  inlining_duration = inlining_sw.GetElapsedTime();

  std::vector<FunctionBase*> functions;
  functions.reserve(packages.size());
  for (const auto& package : packages) {
    std::optional<FunctionBase*> top = package->GetTop();
    if (!top.has_value()) {
      return absl::InvalidArgumentError("Package has no top entity: " +
                                        package->name());
    }
    functions.push_back(top.value());
  }

  solvers::ProverResult result;
  if (functions[0]->IsFunction()) {
    if (!functions[1]->IsFunction()) {
      return absl::InvalidArgumentError("Both inputs must be functions");
    }
    XLS_ASSIGN_OR_RETURN(result, CheckFunctionEquivalence(
                                     functions[0]->AsFunctionOrDie(),
                                     functions[1]->AsFunctionOrDie(), solver,
                                     optimize_merged_ir, equiv_timing));
  } else if (functions[0]->IsProc()) {
    if (!functions[1]->IsProc()) {
      return absl::InvalidArgumentError("Both inputs must be procs");
    }
    if (!activation_count || *activation_count <= 0) {
      return absl::InvalidArgumentError(
          "a positive activation is required for proc equivalence checking");
    }
    XLS_ASSIGN_OR_RETURN(
        result,
        CheckProcEquivalence(functions[0]->AsProcOrDie(),
                             functions[1]->AsProcOrDie(), *activation_count,
                             solver, optimize_merged_ir, equiv_timing));
  } else {
    return absl::InternalError(
        "Block equivalence checking not supported currently.");
  }

  if (std::holds_alternative<solvers::ProvenTrue>(result)) {
    std::cout << "Verified equivalent\n";
  } else {
    XLS_RET_CHECK(std::holds_alternative<solvers::ProvenFalse>(result));
    XLS_ASSIGN_OR_RETURN(
        std::vector<std::string> params,
        CounterexampleParams(functions[0],
                             std::get<solvers::ProvenFalse>(result)));
    std::cout << "Verified NOT equivalent; results differ for input: "
              << absl::StrJoin(params, ", ") << "\n";
    return false;
  }

  return true;
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  QCHECK_EQ(positional_args.size(), 2) << "Two IR files must be specified!";
  auto result = xls::RealMain(positional_args, absl::GetFlag(FLAGS_top),
                              absl::GetFlag(FLAGS_activation_count),
                              absl::GetFlag(FLAGS_solver),
                              absl::GetFlag(FLAGS_optimize_merged_ir),
                              absl::GetFlag(FLAGS_print_timing));
  if (!result.ok()) {
    return xls::ExitStatus(result.status());
  }
  return *result ? absl::GetFlag(FLAGS_match_exit_code)
                 : absl::GetFlag(FLAGS_mismatch_exit_code);
}
