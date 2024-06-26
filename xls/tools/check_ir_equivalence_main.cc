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

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/map_inlining_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/unroll_pass.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_translator.h"
#include "external/z3/src/api/z3_api.h"

static constexpr std::string_view kUsage = R"(
Verifies that the two provided XLS IR files are logically equivalent; that is,
they produce the same outputs across all inputs. The most common usage is to
prove that optimizations are safe - that they don't change program outputs.

Example invocation:
  check_ir_equivalence_main <IR file> <IR file>

If there are multiple functions in the specified files, then it's _strongly_
recommended that you specify --function to ensure that the right functions are
compared. If the tool picks the wrong one, a crash may result.
)";

// LINT.IfChange
ABSL_FLAG(std::string, top, "",
          "The top entity to check. If unspecified, an attempt will be made"
          "to find and check a top entity for the package. Currently, only"
          "Functions are supported.");
ABSL_FLAG(absl::Duration, timeout, absl::InfiniteDuration(),
          "How long to wait for any proof to complete.");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)

namespace xls {
namespace {

absl::StatusOr<std::vector<std::string>> CounterexampleParams(
    Function* f, const solvers::z3::ProvenFalse& proven_false) {
  std::vector<std::string> counterexample;
  using ParamValues = absl::flat_hash_map<const Param*, Value>;
  XLS_ASSIGN_OR_RETURN(ParamValues counterexample_map,
                       proven_false.counterexample);
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
  return counterexample;
}

absl::Status RealMain(const std::vector<std::string_view>& ir_paths,
                      const std::string& entry, absl::Duration timeout) {
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
  // TODO(b/154025625): Replace this with a new InliningPass.
  OptimizationCompoundPass inlining_passes("inlining_passes",
                                           "All inlining passes.");
  inlining_passes.Add<MapInliningPass>();
  inlining_passes.Add<UnrollPass>();
  inlining_passes.Add<InliningPass>();
  inlining_passes.Add<DeadCodeEliminationPass>();
  OptimizationPassOptions options;
  PassResults results;
  for (const auto& package : packages) {
    bool keep_going = true;
    while (keep_going) {
      XLS_ASSIGN_OR_RETURN(
          keep_going, inlining_passes.Run(package.get(), options, &results));
    }
  }

  std::vector<Function*> functions;
  for (const auto& package : packages) {
    XLS_ASSIGN_OR_RETURN(Function * func, package->GetTopAsFunction());
    functions.push_back(func);
  }

  XLS_ASSIGN_OR_RETURN(
      solvers::z3::ProverResult result,
      solvers::z3::TryProveEquivalence(functions[0], functions[1], timeout));
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
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  QCHECK_EQ(positional_args.size(), 2) << "Two IR files must be specified!";
  return xls::ExitStatus(xls::RealMain(
      positional_args, absl::GetFlag(FLAGS_top), absl::GetFlag(FLAGS_timeout)));
}
