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

#include <random>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/ir_jit.h"
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"

const char kUsage[] = R"(
Evaluates an IR file with user-specified or random inputs using the IR
interpreter. Example invocations:

Evaluate IR with a single input:

   eval_ir_main --input='bits[32]:42; (bits[7]:0, bits[20]:4)' IR_FILE

Evaluate IR with a single input and fail if result does not match expected
value:

   eval_ir_main --input='bits[32]:42; (bits[7]:0, bits[20]:4)' \
      --expected='bits[3]:5' IR_FILE

Evaluate an IR function with a batch of arguments, one per line in INPUT_FILE
and compare each result against a corresponding values in EXPECTED_FILE:

   eval_ir_main --input_file=INPUT_FILE --expected_file=EXPECTED_FILE IR_FILE

Evaluate IR with randomly generated inputs:

   eval_ir_main --random_inputs=100 IR_FILE

Evaluate IR before and after optimizations:

    eval_ir_main --random_inputs=100 --optimize_ir IR_FILE

Evaluate IR after every optimization pass:
  eval_ir_main --random_inputs=100 --optimize_ir --eval_after_each_pass IR_FILE

Evaluate IR using the JIT and with the interpreter and compare the results:
  eval_ir_main --test_llvm_jit --random_inputs=100  IR_FILE
)";

ABSL_FLAG(std::string, entry, "", "Entry function name to evaluate.");
ABSL_FLAG(std::string, input, "",
          "The input to the function as a semicolon-separated list of typed "
          "values. For example: \"bits[32]:42; (bits[7]:0, bits[20]:4)\"");
ABSL_FLAG(std::string, input_file, "",
          "Inputs to interpreter, one set per line. Each line should contain a "
          "semicolon-separated set of typed values. Cannot be specified with "
          "--input.");
ABSL_FLAG(int64, random_inputs, 0,
          "If non-zero, this is the number of randomly generated inputs to use "
          "in evaluation. Cannot be specified with --input.");
ABSL_FLAG(std::string, expected, "",
          "The expected result of the evaluation. A non-zero error code is "
          "returned if the evaluated result does not match.");
ABSL_FLAG(
    std::string, expected_file, "",
    "The expected result(s) of the evaluation(s). A non-zero error code is "
    "returned if the evaluated result does not match. Must be specified with "
    "--input_file.");
ABSL_FLAG(bool, optimize_ir, false,
          "Run optimization passes on the input and evaluate before and after "
          "optimizations. A non-zero error status is returned if the results "
          "do not match.");
ABSL_FLAG(
    bool, eval_after_each_pass, false,
    "When specified with --optimize_ir, run evaluation after each pass. "
    "A non-zero error status is returned if any of the results do not match.");
ABSL_FLAG(bool, use_llvm_jit, true, "Use the LLVM IR JIT for execution.");
ABSL_FLAG(bool, test_llvm_jit, false,
          "If true, then run the JIT and compare the results against the "
          "interpereter.");
ABSL_FLAG(int64, llvm_opt_level, 3,
          "The optimization level of the LLVM JIT. Valid values are from 0 (no "
          "optimizations) to 3 (maximum optimizations).");

ABSL_FLAG(
    std::string, test_only_inject_jit_result, "",
    "Test-only flag for injecting the result produced by the JIT. Used to "
    "force mismatches between JIT and interpreter for testing purposed.");

namespace xls {
namespace {

// Excapsulates a set of arguments to pass to the function for evaluation and
// the expected result.
struct ArgSet {
  std::vector<Value> args;
  absl::optional<Value> expected;
};

// Returns the given arguments as a semicolon-separated string.
std::string ArgsToString(absl::Span<const Value> args) {
  return absl::StrJoin(args, "; ", [](std::string* s, const Value& v) {
    absl::StrAppend(s, v.ToString(FormatPreference::kHex));
  });
}

// Evaluates the function with the given ArgSets. Returns an error if the result
// does not match expectations (if any). 'actual_src' and 'expected_src' are
// string descriptions of the sources of the actual results and expected
// results, respectively. These strings are included in error messages.
absl::StatusOr<std::vector<Value>> Eval(
    Function* f, absl::Span<const ArgSet> arg_sets, bool use_jit,
    absl::string_view actual_src = "actual",
    absl::string_view expected_src = "expected") {
  std::unique_ptr<IrJit> jit;
  if (use_jit) {
    // No support for procs yet.
    XLS_ASSIGN_OR_RETURN(jit,
                         IrJit::Create(f, /*queue_mgr=*/nullptr,
                                       absl::GetFlag(FLAGS_llvm_opt_level)));
  }

  std::vector<Value> results;
  for (const ArgSet& arg_set : arg_sets) {
    Value result;
    if (use_jit) {
      if (absl::GetFlag(FLAGS_test_only_inject_jit_result).empty()) {
        XLS_ASSIGN_OR_RETURN(result, jit->Run(arg_set.args));
      } else {
        XLS_ASSIGN_OR_RETURN(result, Parser::ParseTypedValue(absl::GetFlag(
                                         FLAGS_test_only_inject_jit_result)));
      }
    } else {
      XLS_ASSIGN_OR_RETURN(result, IrInterpreter::Run(f, arg_set.args));
    }
    std::cout << result.ToString(FormatPreference::kHex) << std::endl;

    if (arg_set.expected.has_value()) {
      if (result != *arg_set.expected) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Miscompare for input \"%s\"\n  %s: %s\n  %s: %s",
            ArgsToString(arg_set.args), actual_src,
            result.ToString(FormatPreference::kHex), expected_src,
            arg_set.expected->ToString(FormatPreference::kHex)));
      }
    }
    results.push_back(result);
  }
  return results;
}

// An invariant checker which evaluates the entry function with the given
// ArgSets. Raises an error if expectations are not matched.
class EvalInvariantChecker : public InvariantChecker {
 public:
  explicit EvalInvariantChecker(absl::Span<const ArgSet> arg_sets, bool use_jit)
      : arg_sets_(arg_sets.begin(), arg_sets.end()), use_jit_(use_jit) {}
  absl::Status Run(Package* package, const PassOptions& options,
                   PassResults* results) const override {
    if (results->invocations.empty()) {
      std::cerr << "// Evaluating entry function at start of pipeline.\n";
    } else {
      std::cerr << "// Evaluating entry function after pass: "
                << results->invocations.back().pass_name << "\n";
    }
    XLS_ASSIGN_OR_RETURN(Function * f, package->EntryFunction());
    std::string actual_src = "before optimizations";
    XLS_RETURN_IF_ERROR(Eval(f, arg_sets_, use_jit_,
                             /*actual_src=*/results->invocations.empty()
                                 ? std::string("start of pipeline")
                                 : results->invocations.back().pass_name,
                             /*expected_src=*/"before optimizations")
                            .status());
    return absl::OkStatus();
  }

 private:
  std::vector<ArgSet> arg_sets_;
  bool use_jit_;
};

// Runs the given ArgSets through the given package. This includes optionally
// (based on flags) optimizing the IR and evaluating the ArgSets during and
// after optimizations.
absl::Status Run(Package* package, absl::Span<const ArgSet> arg_sets_in) {
  XLS_ASSIGN_OR_RETURN(Function * f, package->EntryFunction());
  // Copy the input ArgSets because we want to write in expected values if they
  // do not exist.
  std::vector<ArgSet> arg_sets(arg_sets_in.begin(), arg_sets_in.end());

  if (absl::GetFlag(FLAGS_test_llvm_jit)) {
    XLS_QCHECK(!absl::GetFlag(FLAGS_optimize_ir))
        << "Cannot specify both --test_llvm_jit and --optimize_ir";
    XLS_ASSIGN_OR_RETURN(std::vector<Value> interpreter_results,
                         Eval(f, arg_sets, /*use_jit=*/false));
    for (int64 i = 0; i < arg_sets.size(); ++i) {
      XLS_QCHECK(!arg_sets[i].expected.has_value())
          << "Cannot specify expected values when using --test_llvm_jit";
      arg_sets[i].expected = interpreter_results[i];
    }
    return Eval(f, arg_sets, /*use_jit=*/true, "JIT", "interpreter").status();
  }

  // Run the argsets through the IR before any optimizations. Write in the
  // results as the expected values if the expected value is not already
  // set. These expected values are used in any later evaluation after
  // optimizations.
  XLS_ASSIGN_OR_RETURN(std::vector<Value> results,
                       Eval(f, arg_sets, absl::GetFlag(FLAGS_use_llvm_jit)));
  for (int64 i = 0; i < arg_sets.size(); ++i) {
    if (!arg_sets[i].expected.has_value()) {
      arg_sets[i].expected = results[i];
    }
  }

  // Run optimizations (optionally) and check the results against expectations
  // (either expected result passed in on the command line or the result
  // produced without optimizations).
  if (absl::GetFlag(FLAGS_optimize_ir)) {
    std::unique_ptr<CompoundPass> pipeline = CreateStandardPassPipeline();
    if (absl::GetFlag(FLAGS_eval_after_each_pass)) {
      pipeline->AddInvariantChecker<EvalInvariantChecker>(
          arg_sets, absl::GetFlag(FLAGS_use_llvm_jit));
    }
    PassResults results;
    XLS_RETURN_IF_ERROR(
        pipeline->Run(package, PassOptions(), &results).status());

    XLS_RETURN_IF_ERROR(Eval(f, arg_sets, absl::GetFlag(FLAGS_use_llvm_jit),
                             "before optimizations", "after optimizations")
                            .status());
  } else {
    XLS_RET_CHECK(!absl::GetFlag(FLAGS_eval_after_each_pass))
        << "Must specify --optimize_ir with --eval_after_each_pass";
  }
  return absl::OkStatus();
}

// Parse the given string as a semi-colon separated list of Values.
absl::StatusOr<ArgSet> ArgSetFromString(absl::string_view args_string) {
  ArgSet arg_set;
  for (const absl::string_view& value_string :
       absl::StrSplit(args_string, ';')) {
    XLS_ASSIGN_OR_RETURN(Value arg, Parser::ParseTypedValue(value_string));
    arg_set.args.push_back(arg);
  }
  return arg_set;
}

absl::Status RealMain(absl::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(input_path));
  std::unique_ptr<Package> package;
  if (absl::GetFlag(FLAGS_entry).empty()) {
    XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackage(contents, input_path));
  } else {
    XLS_ASSIGN_OR_RETURN(package,
                         Parser::ParsePackageWithEntry(
                             contents, absl::GetFlag(FLAGS_entry), input_path));
  }
  XLS_ASSIGN_OR_RETURN(Function * f, package->EntryFunction());

  std::vector<ArgSet> arg_sets;
  if (!absl::GetFlag(FLAGS_input).empty()) {
    XLS_QCHECK_EQ(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Cannot specify both --input and --random_inputs";
    XLS_QCHECK(absl::GetFlag(FLAGS_input_file).empty())
        << "Cannot specify both --input and --input_file";
    absl::StatusOr<ArgSet> arg_set_status =
        ArgSetFromString(absl::GetFlag(FLAGS_input));
    XLS_QCHECK_OK(arg_set_status.status())
        << "Failed to parse input: " << absl::GetFlag(FLAGS_input);

    arg_sets.push_back(arg_set_status.value());
  } else if (!absl::GetFlag(FLAGS_input_file).empty()) {
    XLS_QCHECK_EQ(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Cannot specify both --input_file and --random_inputs";
    absl::StatusOr<std::string> args_input_file =
        GetFileContents(absl::GetFlag(FLAGS_input_file));
    XLS_QCHECK_OK(args_input_file.status());
    for (const auto& arg_line : absl::StrSplit(args_input_file.value(), '\n',
                                               absl::SkipWhitespace())) {
      absl::StatusOr<ArgSet> arg_set_status = ArgSetFromString(arg_line);
      XLS_QCHECK_OK(arg_set_status.status())
          << absl::StreamFormat("Invalid line in input file %s: %s",
                                absl::GetFlag(FLAGS_input_file), arg_line);
      arg_sets.push_back(arg_set_status.value());
    }
  } else {
    XLS_QCHECK_NE(absl::GetFlag(FLAGS_random_inputs), 0)
        << "Must specify --input, --input_file, or --random_inputs.";
    arg_sets.resize(absl::GetFlag(FLAGS_random_inputs));
    std::minstd_rand rng_engine;
    for (ArgSet& arg_set : arg_sets) {
      for (Param* param : f->params()) {
        arg_set.args.push_back(RandomValue(param->GetType(), &rng_engine));
      }
    }
  }

  if (!absl::GetFlag(FLAGS_expected).empty()) {
    XLS_QCHECK(absl::GetFlag(FLAGS_expected_file).empty())
        << "Cannot specify both --expected_file and --expected";
    absl::StatusOr<Value> expected_status =
        Parser::ParseTypedValue(absl::GetFlag(FLAGS_expected));
    XLS_QCHECK_OK(expected_status.status())
        << "Failed to parse expected value: " << absl::GetFlag(FLAGS_expected);
    // Set the expected value of every sample to 'expected'.
    for (ArgSet& arg_set : arg_sets) {
      arg_set.expected = expected_status.value();
    }
  } else if (!absl::GetFlag(FLAGS_expected_file).empty()) {
    absl::StatusOr<std::string> expected_file =
        GetFileContents(absl::GetFlag(FLAGS_expected_file));
    XLS_QCHECK_OK(expected_file.status());
    std::vector<Value> expecteds;
    for (const auto& expected_line :
         absl::StrSplit(expected_file.value(), '\n', absl::SkipWhitespace())) {
      absl::StatusOr<Value> expected_status =
          Parser::ParseTypedValue(expected_line);
      XLS_QCHECK_OK(expected_status.status())
          << absl::StreamFormat("Failed to parse line in expected file %s: %s",
                                expected_line, expected_line);
      expecteds.push_back(expected_status.value());
    }
    XLS_QCHECK_EQ(expecteds.size(), arg_sets.size())
        << "Number of values in expected file does not match the number of "
           "inputs.";
    for (int64 i = 0; i < arg_sets.size(); ++i) {
      arg_sets[i].expected = expecteds[i];
    }
  }

  return Run(package.get(), arg_sets);
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  if (positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <ir-path>",
                                          argv[0]);
  }
  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0]));
}
