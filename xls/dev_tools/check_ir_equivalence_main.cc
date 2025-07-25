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
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
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
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_translator.h"

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
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

namespace xls {
namespace {

absl::StatusOr<solvers::z3::ProverResult> CheckFunctionEquivalence(
    Function* f1, Function* f2) {
  return solvers::z3::TryProveEquivalence(f1, f2);
}
absl::StatusOr<solvers::z3::ProverResult> CheckProcEquivalence(
    Proc* p1, Proc* p2, int64_t activation_count) {
  XLS_ASSIGN_OR_RETURN(
      Function * f1,
      UnrollProcToFunction(p1, activation_count, /*include_state=*/false),
      _ << "Unable to unroll: " << p1->DumpIr());
  XLS_ASSIGN_OR_RETURN(
      Function * f2,
      UnrollProcToFunction(p2, activation_count, /*include_state=*/false),
      _ << "Unable to unroll: " << p2->DumpIr());
  return CheckFunctionEquivalence(f1, f2);
}

absl::StatusOr<std::vector<std::string>> CounterexampleParams(
    FunctionBase* f, const solvers::z3::ProvenFalse& proven_false) {
  std::vector<std::string> counterexample;
  using ParamValues = absl::flat_hash_map<const Param*, Value>;
  XLS_ASSIGN_OR_RETURN(ParamValues counterexample_map,
                       proven_false.counterexample);
  if (f->IsFunction()) {
    for (const xls::Param* param : f->params()) {
      bool missing = true;
      for (const auto& [counterexample_param, value] : counterexample_map) {
        if (counterexample_param->name() == param->name()) {
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
                              std::optional<int64_t> activation_count) {
  auto timeout = StartTimeoutTimer();
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

  solvers::z3::ProverResult result;
  if (functions[0]->IsFunction()) {
    if (!functions[1]->IsFunction()) {
      return absl::InvalidArgumentError("Both inputs must be functions");
    }
    XLS_ASSIGN_OR_RETURN(
        result, CheckFunctionEquivalence(functions[0]->AsFunctionOrDie(),
                                         functions[1]->AsFunctionOrDie()));
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
                             functions[1]->AsProcOrDie(), *activation_count));
  } else {
    return absl::InternalError(
        "Block equivalence checking not supported currently.");
  }

  if (std::holds_alternative<solvers::z3::ProvenTrue>(result)) {
    std::cout << "Verified equivalent\n";
  } else {
    XLS_RET_CHECK(std::holds_alternative<solvers::z3::ProvenFalse>(result));
    XLS_ASSIGN_OR_RETURN(
        std::vector<std::string> params,
        CounterexampleParams(functions[0],
                             std::get<solvers::z3::ProvenFalse>(result)));
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
                              absl::GetFlag(FLAGS_activation_count));
  if (!result.ok()) {
    return xls::ExitStatus(result.status());
  }
  return *result ? absl::GetFlag(FLAGS_match_exit_code)
                 : absl::GetFlag(FLAGS_mismatch_exit_code);
}
