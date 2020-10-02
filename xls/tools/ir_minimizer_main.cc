// Copyright 2020 Google LLC
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

#include <random>

#include "absl/flags/flag.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/ir/verifier.h"
#include "xls/jit/ir_jit.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/standard_pipeline.h"

const char* kUsage = R"(
Tool for reducing IR to a minimal test case based on an external test.

Selectively removes nodes from the graph and performs various simplifications
(e.g., CSE, DCE, etc.) while preserving the feature/bug indicated by an
external test.

Note that this is currently specialized to reduce fuzz functions, which are
fairly self contained. Reduction may not be as effective with larger samples
(like real user packages).

Currently the algorithm is such that the number of live nodes in the function
should go down monotonically.

The reducer supports two modes of operation. In the first mode, an external test
executable is passed in. This test should return a zero exit status if the IR
exhibits the bug. Example invocation:

  ir_minimizer_main --test_executable=/foo/test.sh IR_FILE

The second mode specifically reduces a test case where the JIT results differ
from the interpreter results. Example invocation:

  ir_minimizer_main --test_llvm_jit --simplify_with_optimization_pipeline \
    --input='bits[32]:42; bits[1]:0' IR_FILE

)";

ABSL_FLAG(bool, can_remove_params, false,
          "Whether parameters can be removed during the minimization process. "
          "If the test executable interprets the IR using a fixed set of "
          "arguments, parameters should not be removed.");
ABSL_FLAG(int64, failed_attempt_limit, 128,
          "Failed simplification attempts (in a row) before we conclude we're "
          "done reducing.");
ABSL_FLAG(int64, total_attempt_limit, 16384,
          "Limit on total number of attempts to try before bailing.");
ABSL_FLAG(
    std::string, test_executable, "",
    "Path to test executable to run during minimization. The test accepts "
    "a single command line argument which is the path to the textual IR "
    "file. The test should return a zero exit code (success) if the IR "
    "exhibits the bug in question. Note, if testing for a crash of a "
    "particular binary this is the opposite return code polarity.");
ABSL_FLAG(bool, test_llvm_jit, false,
          "Tests for differences between results from the JIT and the "
          "interpreter as the reduction test case. Must specify --input with "
          "this flag.");
ABSL_FLAG(std::string, input, "",
          "Input to use when invoking the JIT and the interpreter. Must be "
          "used with --test_llvm_jit.");
ABSL_FLAG(
    std::string, test_only_inject_jit_result, "",
    "Test-only flag for injecting the result produced by the JIT. Used to "
    "force mismatches between JIT and interpreter for testing purposed.");
ABSL_FLAG(bool, simplify_with_optimization_pipeline, false,
          "If true, then include the standard optimization pipeline as a "
          "simplification option during reduction. This option should *not* be "
          "used if trying to reduce a crash in the optimization pipeline "
          "itself as the minimizer will then crash.");

namespace xls {
namespace {

// Checks whether we still fail when attempting to run function "f". Optional
// 'inputs' is required if --test_llvm_jit is used.
absl::StatusOr<bool> StillFails(absl::string_view ir_text,
                                absl::optional<std::vector<Value>> inputs) {
  XLS_VLOG_LINES(
      1, absl::StrCat("=== Verifying contents still fails:\n", ir_text));
  if (!absl::GetFlag(FLAGS_test_executable).empty()) {
    // Test for bug using external executable.
    XLS_ASSIGN_OR_RETURN(TempFile temp_file,
                         TempFile::CreateWithContent(ir_text));
    std::string ir_path = temp_file.path().string();

    XLS_QCHECK(!absl::GetFlag(FLAGS_test_llvm_jit))
        << "Cannot specify --test_llvm_jit with --test_executable";
    XLS_QCHECK(absl::GetFlag(FLAGS_input).empty())
        << "Cannot specify --input with --test_executable";
    absl::StatusOr<std::pair<std::string, std::string>> result =
        InvokeSubprocess({absl::GetFlag(FLAGS_test_executable), ir_path});

    if (result.ok()) {
      const auto& [stdout_str, stderr_str] = *result;
      XLS_LOG(INFO) << "stdout:  \"\"\"" << stdout_str << "\"\"\"";
      XLS_LOG(INFO) << "stderr:  \"\"\"" << stderr_str << "\"\"\"";
      XLS_LOG(INFO) << "retcode: 0";
    } else {
      XLS_LOG(INFO) << result.status();
    }
    return result.ok();
  }

  // Test for bugs by comparing the results of the JIT and interpreter.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  XLS_RET_CHECK(inputs.has_value());
  XLS_ASSIGN_OR_RETURN(Function * main, package->EntryFunction());
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<IrJit> jit,
                       IrJit::Create(main, /*queue_mgr=*/nullptr));
  Value jit_result;
  if (absl::GetFlag(FLAGS_test_only_inject_jit_result).empty()) {
    XLS_ASSIGN_OR_RETURN(jit_result, jit->Run(*inputs));
  } else {
    XLS_ASSIGN_OR_RETURN(jit_result, Parser::ParseTypedValue(absl::GetFlag(
                                         FLAGS_test_only_inject_jit_result)));
  }
  XLS_ASSIGN_OR_RETURN(Value interpreter_result,
                       IrInterpreter::Run(main, *inputs));
  if (jit_result != interpreter_result) {
    return true;
  }
  return false;
}

// Writes the IR out to a temporary file, runs the test executable on it, and
// returns 'true' if the test (still) fails on that IR text.
absl::Status VerifyStillFails(absl::string_view ir_text,
                              absl::optional<std::vector<Value>> inputs,
                              absl::string_view description) {
  XLS_ASSIGN_OR_RETURN(bool still_fails, StillFails(ir_text, inputs));

  if (!still_fails) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unexpected PASS: ", description));
  }

  XLS_VLOG(1) << "Confirmed: sample still fails.";
  return absl::OkStatus();
}

// Removes params with zero users from the function.
absl::StatusOr<bool> RemoveDeadParameters(Function* f) {
  std::vector<Param*> params(f->params().begin(), f->params().end());
  for (Param* p : params) {
    if (p->users().empty() && p != f->return_value()) {
      XLS_RETURN_IF_ERROR(f->RemoveNode(p, /*remove_param_ok=*/true));
    }
  }
  return params.size() != f->params().size();
}

enum class SimplificationResult {
  kCannotChange,  // Cannot simplify.
  kDidNotChange,  // Did not simplify, e.g. because RNG didn't come up that way.
  kDidChange,     // Did simplify in some way.
};

absl::StatusOr<SimplificationResult> SimplifyReturnValue(
    Function* f, std::mt19937* rng, std::string* which_transform) {
  Node* orig = f->return_value();
  SimplificationResult result = SimplificationResult::kDidNotChange;
  if (orig->Is<Tuple>()) {
    Tuple* orig_tup = orig->As<Tuple>();
    const int64 orig_tuple_size = orig_tup->operands().size();
    if (orig_tuple_size == 1) {
      // Unbox the singleton tuple.
      XLS_RETURN_IF_ERROR(f->set_return_value(orig_tup->operand(0)));
      *which_transform = "unbox singleton tuple return value";
      return SimplificationResult::kDidChange;
    }
    std::vector<Node*> new_members;
    for (Node* member : orig_tup->operands()) {
      float p = absl::Uniform<float>(*rng, 0.0f, 1.0f);
      if (p >= 1.0 / orig_tuple_size) {
        new_members.push_back(member);
      } else {
        result = SimplificationResult::kDidChange;
      }
    }
    if (result != SimplificationResult::kDidChange) {
      // We got through all the members without changing any.
      return result;
    }
    *which_transform = absl::StrFormat("return tuple reduction: %d => %d ",
                                       orig_tuple_size, new_members.size());
    XLS_ASSIGN_OR_RETURN(Tuple * new_t,
                         f->MakeNode<Tuple>(orig->loc(), new_members));
    XLS_RETURN_IF_ERROR(f->set_return_value(new_t));
    return result;
  }

  if (orig->operand_count() > 0) {
    int64 which_operand = absl::Uniform<int>(*rng, 0, orig->operand_count());
    XLS_RETURN_IF_ERROR(f->set_return_value(orig->operand(which_operand)));
    *which_transform =
        absl::StrFormat("return operand %d of return value", which_operand);
    return SimplificationResult::kDidChange;
  }

  if (orig->Is<Literal>()) {
    return SimplificationResult::kCannotChange;
  }

  XLS_LOG(WARNING) << "Cannot yet simplify return value node: "
                   << orig->ToString();
  return SimplificationResult::kCannotChange;
}

absl::StatusOr<SimplificationResult> Simplify(
    Function* f, absl::optional<std::vector<Value>> inputs, std::mt19937* rng,
    std::string* which_transform) {
  // Return a uniform random number over the interval [0, 1).
  auto rand_0_to_1 = [&]() { return absl::Uniform<float>(*rng, 0.0f, 1.0f); };

  if (rand_0_to_1() < 0.5) {
    // Attempt to CSE the function and see if it still fails.
    // If so, we'll have a much tidier graph.
    //
    // TODO(leary): 2019-09-25 We could do a "CSE this one node" and randomly
    // select the node to get this to be more incremental, which would likely
    // help avoid perturbing away any essential part of the graph while cleaning
    // it up some.
    CsePass cse;
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         cse.RunOnFunction(f, PassOptions(), &results));
    if (changed) {
      *which_transform = "CSE";
      return SimplificationResult::kDidChange;
    }
  }

  if (rand_0_to_1() < 0.1) {
    // Attempt to do constant folding.
    ConstantFoldingPass folding_pass;
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed, folding_pass.RunOnFunction(f, PassOptions(), &results));
    if (changed) {
      *which_transform = "Constant folding";
      return SimplificationResult::kDidChange;
    }
  }

  if (absl::GetFlag(FLAGS_simplify_with_optimization_pipeline) &&
      rand_0_to_1() < 0.05) {
    // Try to run the sample through the entire optimization pipeline.
    XLS_ASSIGN_OR_RETURN(bool changed, RunStandardPassPipeline(f->package()));
    if (changed) {
      *which_transform = "Optimization pipeline";
      return SimplificationResult::kDidChange;
    }
  }

  if (rand_0_to_1() < 0.1) {
    XLS_ASSIGN_OR_RETURN(auto result,
                         SimplifyReturnValue(f, rng, which_transform));
    if (result == SimplificationResult::kDidChange) {
      return result;
    }
  }

  if (inputs.has_value() && rand_0_to_1() < 0.3) {
    // Try to replace a parameter with a literal equal to the respective input
    // value.
    int64 param_no = absl::Uniform<int64>(*rng, 0, f->params().size());
    Param* param = f->params()[param_no];
    if (!param->users().empty() || param == f->return_value()) {
      XLS_RETURN_IF_ERROR(
          param->ReplaceUsesWithNew<Literal>(inputs->at(param_no)).status());
      *which_transform = absl::StrFormat(
          "random replace parameter %d (%s) with literal of input value: %s",
          param_no, param->GetName(), inputs->at(param_no).ToString());
      return SimplificationResult::kDidChange;
    }
  }

  int64 i = absl::Uniform<int64>(*rng, 0, f->node_count());
  Node* n = *std::next(f->nodes().begin(), i);
  if (n->Is<Literal>() && n->As<Literal>()->value().IsAllZeros()) {
    XLS_VLOG(1) << "Candidate for zero-replacement already a literal zero.";
    return SimplificationResult::kDidNotChange;
  }
  if (n->Is<Param>() && n->users().empty()) {
    // Can't replace unused params with zero.
    XLS_VLOG(1) << "Candidate for zero-replacement is a user-less parameter.";
    return SimplificationResult::kDidNotChange;
  }
  XLS_RETURN_IF_ERROR(
      n->ReplaceUsesWithNew<Literal>(ZeroOfType(n->GetType())).status());
  *which_transform = "random replace with zero: " + n->GetName();
  return SimplificationResult::kDidChange;
}

// Runs removal of dead nodes (transitively), and then any dead parameters.
//
// Note removing dead parameters will not cause any additional nodes to be dead.
absl::Status CleanUp(Function* f, bool can_remove_params) {
  DeadCodeEliminationPass dce;
  DeadFunctionEliminationPass dfe;
  PassResults results;
  XLS_RETURN_IF_ERROR(dce.RunOnFunction(f, PassOptions(), &results).status());
  if (can_remove_params) {
    XLS_RETURN_IF_ERROR(RemoveDeadParameters(f).status());
  }
  XLS_RETURN_IF_ERROR(dfe.Run(f->package(), PassOptions(), &results).status());
  return absl::OkStatus();
}

absl::Status RealMain(absl::string_view path, const int64 failed_attempt_limit,
                      const int64 total_attempt_limit) {
  XLS_ASSIGN_OR_RETURN(std::string knownf_ir_text, GetFileContents(path));

  // Parse inputs, if specified.
  absl::optional<std::vector<xls::Value>> inputs;
  if (!absl::GetFlag(FLAGS_input).empty()) {
    inputs = std::vector<xls::Value>();
    XLS_QCHECK(absl::GetFlag(FLAGS_test_llvm_jit))
        << "Can only specify --input with --test_llvm_jit";
    for (const absl::string_view& value_string :
         absl::StrSplit(absl::GetFlag(FLAGS_input), ';')) {
      XLS_ASSIGN_OR_RETURN(Value input, Parser::ParseTypedValue(value_string));
      inputs->push_back(input);
    }
  }

  // Check what the user gave us actually fails.
  XLS_RETURN_IF_ERROR(VerifyStillFails(
      knownf_ir_text, inputs,
      "Originally-provided main function provided does not fail"));

  const bool can_remove_params = absl::GetFlag(FLAGS_can_remove_params);

  // Clean up any initial garbage and see if it still fails.
  {
    XLS_LOG(INFO) << "=== Cleaning up initial garbage";
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         Parser::ParsePackage(knownf_ir_text));
    XLS_ASSIGN_OR_RETURN(Function * main, package->EntryFunction());
    // const int64 original_node_count = main->node_count();
    XLS_RETURN_IF_ERROR(CleanUp(main, can_remove_params));
    XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
    knownf_ir_text = package->DumpIr();
    XLS_RETURN_IF_ERROR(
        VerifyStillFails(knownf_ir_text, inputs,
                         "Original main function does not fail after cleanup"));
    XLS_LOG(INFO) << "=== Done cleaning up initial garbage";
  }

  // If so, we start simplifying via this seeded RNG.
  std::mt19937 rng;  // Default constructor uses deterministic seed.

  // Smallest version of the function that's known to be failing.
  int64 failed_simplification_attempts = 0;
  int64 total_attempts = 0;

  while (true) {
    if (failed_simplification_attempts >= failed_attempt_limit) {
      XLS_LOG(INFO) << "Hit failed-simplification-attempt-limit: "
                    << failed_simplification_attempts;
      // Used up all our attempts for this state.
      break;
    }

    total_attempts++;
    if (total_attempts >= total_attempt_limit) {
      XLS_LOG(INFO) << "Hit total-attempt-limit: " << total_attempts;
      break;
    }

    XLS_VLOG(1) << "=== Simplification attempt " << total_attempts;

    XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(knownf_ir_text));
    XLS_ASSIGN_OR_RETURN(Function * candidate, package->EntryFunction());
    XLS_VLOG_LINES(1,
                   "=== Candidate for simplification:\n" + candidate->DumpIr());

    // Simplify the function.
    std::string which_transform;
    XLS_ASSIGN_OR_RETURN(SimplificationResult simplification,
                         Simplify(candidate, inputs, &rng, &which_transform));

    // If we cannot change it, we're done.
    if (simplification == SimplificationResult::kCannotChange) {
      XLS_LOG(INFO) << "Cannot simplify any further, done!";
      break;
    }

    // If we happened to not change it (e.g. because the RNG said not to), keep
    // going until we do. We still bump the counter to make sure we don't end up
    // wedged in a state where we can't simplify anything.
    if (simplification == SimplificationResult::kDidNotChange) {
      XLS_VLOG(1) << "Did not change the sample.";
      failed_simplification_attempts++;
      continue;
    }

    // When we changed (simplified) it, clean it up then see if it still fails.
    XLS_CHECK(simplification == SimplificationResult::kDidChange);
    XLS_RETURN_IF_ERROR(CleanUp(candidate, can_remove_params));

    XLS_VLOG_LINES(1, "=== After simplification [" + which_transform + "]\n" +
                          candidate->DumpIr());

    std::string candidate_ir_text = package->DumpIr();
    XLS_ASSIGN_OR_RETURN(bool still_fails,
                         StillFails(candidate_ir_text, inputs));
    if (!still_fails) {
      failed_simplification_attempts++;
      XLS_VLOG(1) << "Failed simplification attempts now: "
                  << failed_simplification_attempts;
      // That simplification caused it to stop failing, but keep going with the
      // last known failing version and seeing if we can find something else
      // from there.
      continue;
    }

    // We found something that definitely fails, update our "knownf" value and
    // reset our failed simplification attempt count since we see we've made
    // some forward progress.
    XLS_RETURN_IF_ERROR(CleanUp(candidate, can_remove_params));

    XLS_RETURN_IF_ERROR(VerifyStillFails(
        knownf_ir_text, inputs, "Known failure does not fail after cleanup!"));

    knownf_ir_text = candidate_ir_text;

    std::cerr << "---\ntransform: " << which_transform << "\n"
              << candidate->DumpIr() << "(" << candidate->node_count()
              << " nodes)" << std::endl;

    failed_simplification_attempts = 0;
  }

  XLS_RETURN_IF_ERROR(VerifyStillFails(knownf_ir_text, inputs,
                                       "Minimized function does not fail!"));

  std::cout << knownf_ir_text;

  XLS_ASSIGN_OR_RETURN(bool fails, StillFails(knownf_ir_text, inputs));
  XLS_RET_CHECK(fails);

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1 || positional_arguments[0].empty()) {
    XLS_LOG(QFATAL) << "Expected path argument with IR: " << argv[0]
                    << " <ir_path>";
  }

  XLS_QCHECK(!absl::GetFlag(FLAGS_test_executable).empty() ^
             absl::GetFlag(FLAGS_test_llvm_jit))
      << "Must specify either --test_executable or --test_llvm_jit";

  // If minimizing a mismatch between the JIT and the interpreter then
  // --simplify_with_optimization_pipeline really should be specified as it much
  // improves the minimization performance and results.
  XLS_QCHECK(!absl::GetFlag(FLAGS_test_llvm_jit) ||
             absl::GetFlag(FLAGS_simplify_with_optimization_pipeline))
      << "Must specify --simplify_with_optimization_pipeline with "
         "--test_llvm_jit";

  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0],
                              absl::GetFlag(FLAGS_failed_attempt_limit),
                              absl::GetFlag(FLAGS_total_attempt_limit)));

  return EXIT_SUCCESS;
}
