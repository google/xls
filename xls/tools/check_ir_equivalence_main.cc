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
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/internal/sysinfo.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/map_inlining_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/unroll_pass.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3.h"
#include "../z3/src/api/z3_api.h"

const char kUsage[] = R"(
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

using solvers::z3::IrTranslator;

// To compare, simply take the output nodes of each function and compare them.
static absl::StatusOr<Z3_ast> CreateComparisonFunction(
    absl::Span<std::unique_ptr<IrTranslator>> translators,
    const std::vector<Function*>& functions) {
  Z3_context ctx = translators[0]->ctx();
  Z3_ast result1 = translators[0]->GetReturnNode();
  Z3_ast result2 = translators[1]->GetReturnNode();

  Z3_sort opt_sort = Z3_get_sort(ctx, result1);
  Z3_sort unopt_sort = Z3_get_sort(ctx, result2);
  XLS_RET_CHECK(Z3_is_eq_sort(ctx, opt_sort, unopt_sort));

  return Z3_mk_eq(ctx, result1, result2);
}

static absl::Status RealMain(const std::vector<std::string_view>& ir_paths,
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

  std::vector<std::unique_ptr<IrTranslator>> translators;
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<IrTranslator> translator,
                       IrTranslator::CreateAndTranslate(functions[0]));
  translators.push_back(std::move(translator));

  // Get the params for the first function, so we can map the second function's
  // parameters to them.
  Z3_context ctx = translators[0]->ctx();
  std::vector<Z3_ast> z3_params;
  for (const Param* param : functions[0]->params()) {
    z3_params.push_back(translators[0]->GetTranslation(param));
  }

  XLS_ASSIGN_OR_RETURN(
      translator, IrTranslator::CreateAndTranslate(ctx, functions[1],
                                                   absl::MakeSpan(z3_params)));
  translators.push_back(std::move(translator));

  XLS_ASSIGN_OR_RETURN(
      Z3_ast results_equal,
      CreateComparisonFunction(absl::MakeSpan(translators), functions));
  translators[0]->SetTimeout(timeout);

  Z3_solver solver =
      solvers::z3::CreateSolver(ctx, std::thread::hardware_concurrency());

  // Remember: we try to prove the condition by searching for a model that
  // produces the opposite result. Thus, we want to find a model where the
  // results are _not_ equal.
  Z3_ast objective = Z3_mk_eq(ctx, Z3_mk_false(ctx), results_equal);
  Z3_solver_assert(ctx, solver, objective);

  // Finally, print the output to the terminal in gorgeous two-color ASCII.
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  std::cout << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
            << '\n';

  Z3_solver_dec_ref(ctx, solver);

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  QCHECK_EQ(positional_args.size(), 2) << "Two IR files must be specified!";
  return xls::ExitStatus(xls::RealMain(
      positional_args, absl::GetFlag(FLAGS_top), absl::GetFlag(FLAGS_timeout)));
}
