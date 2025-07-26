// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

ABSL_FLAG(bool, print_ir_fuzzer_info, false,
          "Force the ir fuzzer to print ir fuzzer and args.");
ABSL_FLAG(bool, print_ir_fuzzer_info_on_fail, false,
          "Force the ir fuzzer to print ir fuzzer and args on failure.");
ABSL_FLAG(std::string, print_ir_fuzzer_info_file, "/dev/stderr",
          "File to print ir fuzzer info to.");

namespace xls {
namespace {

// Evaluates the IR function with a set of parameter values.
absl::StatusOr<std::vector<InterpreterResult<Value>>> EvaluateArgSets(
    Function* f, absl::Span<const std::vector<Value>> param_sets) {
  std::vector<InterpreterResult<Value>> results;
  for (const std::vector<Value>& param_set : param_sets) {
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> result,
                         InterpretFunction(f, param_set));
    results.push_back(result);
  }
  return results;
}

// Returns true if the results are different.
bool DoResultsChange(absl::Span<const InterpreterResult<Value>> before_results,
                     absl::Span<const InterpreterResult<Value>> after_results) {
  for (int64_t i = 0; i < before_results.size(); i += 1) {
    if (before_results[i].value != after_results[i].value) {
      return true;
    }
  }
  return false;
}

// Returns a human-readable string representation of the argument sets.
std::string StringifyArgSets(absl::Span<const std::vector<Value>> arg_sets) {
  std::stringstream ss;
  ss << "[";
  // Iterate over the number of param sets.
  for (int64_t i = 0; i < arg_sets.size(); i += 1) {
    // Iterate over the param set elements.
    for (int64_t j = 0; j < arg_sets[i].size(); j += 1) {
      ss << (arg_sets[i][j].ToHumanString())
         << (j != arg_sets[i].size() - 1 ? ", " : "");
    }
    ss << (i != arg_sets.size() - 1 ? "], [" : "]");
  }
  return ss.str();
}

// Returns a human-readable string representation of the results.
std::string StringifyResults(
    absl::Span<const InterpreterResult<Value>> results) {
  std::stringstream ss;
  ss << "[";
  for (int64_t i = 0; i < results.size(); i += 1) {
    ss << results[i].value;
    ss << (i != results.size() - 1 ? ", " : "]");
  }
  return ss.str();
}

}  // namespace

namespace {
void RecordFuzzInfoString(std::string_view package,
                          std::optional<std::string_view> args) {
  std::ostringstream oss;
  oss << "IR: \n";
  oss << package << "\n";
  if (args) {
    oss << "args: " << *args << "\n";
  }
  auto status = AppendStringToFile(
      absl::GetFlag(FLAGS_print_ir_fuzzer_info_file), std::move(oss).str());
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write output: " << status;
  }
}

class PrintIrOnFailure : public testing::EmptyTestEventListener {
 public:
  PrintIrOnFailure(const testing::TestInfo* test, std::string_view ir,
                   std::optional<std::string_view> args)
      : test_(test),
        ir_(ir),
        args_(args ? std::optional<std::string>(std::string(*args))
                   : std::nullopt) {}
  void OnTestEnd(const testing::TestInfo& result) override {
    if (&result == test_ && result.result()->Failed()) {
      RecordFuzzInfoString(ir_, args_);
    }
  }

 private:
  const testing::TestInfo* test_;
  std::string ir_;
  std::optional<std::string> args_;
};
void MaybeRecordFuzzInfoString(std::string_view initial_ir,
                               std::optional<std::string_view> args) {
  auto inst = testing::UnitTest::GetInstance();
  const testing::TestInfo* cur_test = inst->current_test_info();
  if (absl::GetFlag(FLAGS_print_ir_fuzzer_info)) {
    RecordFuzzInfoString(initial_ir, args);
  } else if (absl::GetFlag(FLAGS_print_ir_fuzzer_info_on_fail) && cur_test) {
    inst->listeners().Append(new PrintIrOnFailure(cur_test, initial_ir, args));
  }
}
}  // namespace
void RecordFuzzInfo(const FuzzPackageWithArgs& args) {
  MaybeRecordFuzzInfoString(args.fuzz_package.p->DumpIr(),
                            StringifyArgSets(args.arg_sets));
}
void RecordFuzzInfo(const FuzzPackage& args) {
  MaybeRecordFuzzInfoString(args.p->DumpIr(), std::nullopt);
}
void RecordFuzzInfo(std::shared_ptr<Package> args) {
  MaybeRecordFuzzInfoString(args->DumpIr(), std::nullopt);
}
void RecordFuzzInfo(const Package& args) {
  MaybeRecordFuzzInfoString(args.DumpIr(), std::nullopt);
}

// Takes an IR function from a Package object and puts it through an
// optimization pass. It then evaluates the IR function with a set of argument
// values before and after the pass. It returns the boolean value of whether the
// results changed.
void OptimizationPassChangesOutputs(FuzzPackageWithArgs fuzz_package_with_args,
                                    OptimizationPass& pass) {
  std::unique_ptr<Package>& p = fuzz_package_with_args.fuzz_package.p;
  FuzzProgramProto& fuzz_program =
      fuzz_package_with_args.fuzz_package.fuzz_program;
  RecordFuzzInfo(fuzz_package_with_args);
  std::vector<std::vector<Value>>& arg_sets = fuzz_package_with_args.arg_sets;
  VLOG(3) << "IR Fuzzer-2: Before Pass IR:" << "\n" << p->DumpIr() << "\n";
  ScopedMaybeRecord<std::string> pre("before_pass", p->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction(kFuzzTestName));
  VLOG(3) << "IR Fuzzer-3: Argument Sets: " << StringifyArgSets(arg_sets)
          << "\n";
  // Interpret the IR function with the arguments before optimization.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> before_pass_results,
      EvaluateArgSets(f, arg_sets));
  // Run the optimization pass over the IR.
  PassResults results;
  OptimizationContext context;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool ir_changed,
      pass.Run(p.get(), OptimizationPassOptions(), &results, context));
  VLOG(3) << "IR Fuzzer-4: After Pass IR:" << "\n" << p->DumpIr() << "\n";
  ScopedMaybeRecord<std::string> post("after_pass", p->DumpIr());
  // Interpret the IR function with the arguments after optimization.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> after_pass_results,
      EvaluateArgSets(f, arg_sets));
  VLOG(3) << "IR Fuzzer-5: IR Changed: " << (ir_changed ? "TRUE" : "FALSE")
          << "\n";
  VLOG(3) << "IR Fuzzer-6: Before Pass Results: "
          << StringifyResults(before_pass_results) << "\n";
  VLOG(3) << "IR Fuzzer-7: After Pass Results: "
          << StringifyResults(after_pass_results) << "\n";
  // Check if the results are the same before and after optimization.
  bool results_changed =
      DoResultsChange(before_pass_results, after_pass_results);
  VLOG(3) << "IR Fuzzer-8: Results Changed: "
          << (results_changed ? "TRUE" : "FALSE") << "\n";
  ASSERT_FALSE(results_changed)
      << "\n"
      << "Expected: " << StringifyResults(before_pass_results) << "\n"
      << "Actual:   " << StringifyResults(after_pass_results) << "\n"
      << "IR:\n"
      << p->DumpIr() << "\n"
      << "Fuzz Protobuf:\n"
      << fuzz_program.DebugString() << "\n";
}

// Accepts a FuzzProgramProto string and runs the given optimization pass over
// the generated IR. It verifies that the results are the same before and after
// the pass. Useful for debugging OptimizationPassChangesOutputs failures.
absl::Status PassChangesOutputsWithProto(std::string proto_string,
                                         int64_t arg_set_count,
                                         OptimizationPass& pass) {
  XLS_ASSIGN_OR_RETURN(FuzzPackage fuzz_package,
                       BuildPackageFromProtoString(proto_string));
  FuzzPackageWithArgs fuzz_package_with_args =
      GenArgSetsForPackage(std::move(fuzz_package), arg_set_count);
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
  return absl::OkStatus();
}

// Performs tests on the IrFuzzBuilder by manually creating a FuzzProgramProto,
// instantiating it into its IR version, and manually verifying the IR is
// correct.
absl::Status EquateProtoToIrTest(
    std::string proto_string, testing::Matcher<const Node*> expected_ir_node) {
  XLS_ASSIGN_OR_RETURN(FuzzPackage fuzz_package,
                       BuildPackageFromProtoString(proto_string));
  std::unique_ptr<Package>& p = fuzz_package.p;
  XLS_ASSIGN_OR_RETURN(Function * f, p->GetFunction(kFuzzTestName));
  VLOG(3) << "IR Fuzzer-2: IR:" << "\n" << p->DumpIr() << "\n";
  // Verify that the proto_ir_node matches the expected_ir_node.
  EXPECT_THAT(f->return_value(), expected_ir_node);
  return absl::OkStatus();
}

// Accepts a string which contains information to instantiate a FuzzProgram
// protobuf object. This object is then used to generate IR which is returned as
// a Package object.
absl::StatusOr<FuzzPackage> BuildPackageFromProtoString(
    std::string proto_string) {
  // Create the package.
  std::unique_ptr<Package> p = std::make_unique<VerifiedPackage>(kFuzzTestName);
  FunctionBuilder fb(kFuzzTestName, p.get());
  // Create the proto from the string.
  FuzzProgramProto fuzz_program;
  XLS_RET_CHECK(
      google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  // Generate the IR from the proto.
  IrFuzzBuilder ir_fuzz_builder(fuzz_program, p.get(), &fb);
  BValue proto_ir = ir_fuzz_builder.BuildIr();
  XLS_RET_CHECK_OK(fb.BuildWithReturnValue(proto_ir));
  return FuzzPackage(std::move(p), fuzz_program);
}

// Returns multiple sets of randomly generated arguments that are compatible for
// a given package function.
FuzzPackageWithArgs GenArgSetsForPackage(FuzzPackage fuzz_package,
                                         int64_t arg_set_count) {
  Function* f = fuzz_package.p->GetFunction(kFuzzTestName).value();
  std::vector<std::vector<Value>> arg_sets;
  std::string args_bytes = fuzz_package.fuzz_program.args_bytes();
  // Convert the args_bytes into a Bits object.
  Bits args_bits = Bits::FromBytes(
      absl::MakeSpan(reinterpret_cast<const uint8_t*>(args_bytes.data()),
                     args_bytes.size()),
      args_bytes.size() * 8);
  int64_t bits_idx = 0;
  // Retrieve arg_set_count amount of arguments for each parameter.
  for (Param* param : f->params()) {
    arg_sets.push_back(
        GenArgsForParam(arg_set_count, param->GetType(), args_bits, bits_idx));
  }
  std::vector<std::vector<Value>> transposed_arg_sets;
  transposed_arg_sets.reserve(arg_set_count);
  // Transpose the arg_sets matrix so that they are column-row aligned.
  for (int64_t col_idx = 0; col_idx < arg_set_count; col_idx += 1) {
    std::vector<Value> arg_set_row;
    for (int64_t row_idx = 0; row_idx < f->params().size(); row_idx += 1) {
      arg_set_row.push_back(arg_sets[row_idx][col_idx]);
    }
    transposed_arg_sets.push_back(arg_set_row);
  }
  // Create the FuzzPackageWithArgs struct.
  FuzzPackageWithArgs fuzz_package_with_args =
      FuzzPackageWithArgs(std::move(fuzz_package), transposed_arg_sets);
  return fuzz_package_with_args;
}

// TODO(allight): Should we bother to serialize the proto. I don't think its
// useful very often.
void FuzzTestPrintSourceCode(const FuzzPackage& fp, std::ostream* os) {
  *os << "FuzzPackage{ .p = ";
  FuzzTestPrintSourceCode(fp.p, os);
  *os << " }";
}
}  // namespace xls
