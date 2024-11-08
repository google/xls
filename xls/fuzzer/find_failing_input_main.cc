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
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/function_jit.h"
#include "xls/tests/testvector.pb.h"

static constexpr std::string_view kUsage = R"(
Runs an IR function with a set of inputs through both the JIT and the
interpreter. Prints the first input which results in a mismatch between the JIT
and the interpreter. Returns a non-zer error code otherwise. Usage:

    find_failing_input_main --input-file=INPUT_FILE IR_FILE
)";

ABSL_FLAG(std::string, input_file, "",  // deprecated
          "Inputs to JIT and interpreter, one set per line. Each line should "
          "contain a semicolon-separated set of typed values. Cannot be "
          "specified with --input.");
ABSL_FLAG(std::string, testvector_textproto, "",
          "A textproto file containing the function argument or proc "
          "channel test vectors.");
ABSL_FLAG(
    std::string, test_only_inject_jit_result, "",
    "Test-only flag for injecting the result produced by the JIT. Used to "
    "force mismatches between JIT and interpreter for testing purposed.");

using ArgSet = std::vector<xls::Value>;

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_path,
                      const std::vector<ArgSet>& inputs) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text, ir_path));
  XLS_ASSIGN_OR_RETURN(Function * f, package->GetTopAsFunction());

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<FunctionJit> jit,
                       FunctionJit::Create(f));
  for (const std::vector<Value>& args : inputs) {
    InterpreterResult<Value> jit_result;
    if (absl::GetFlag(FLAGS_test_only_inject_jit_result).empty()) {
      XLS_ASSIGN_OR_RETURN(jit_result, jit->Run(args));
    } else {
      XLS_ASSIGN_OR_RETURN(jit_result.value,
                           Parser::ParseTypedValue(absl::GetFlag(
                               FLAGS_test_only_inject_jit_result)));
    }
    // TODO(https://github.com/google/xls/issues/506): 2021-10-12 Also compare
    // events once the JIT fully supports events (and we have decided how to
    // handle event mismatches).
    XLS_ASSIGN_OR_RETURN(InterpreterResult<Value> interpreter_result,
                         InterpretFunction(f, args));
    if (jit_result.value != interpreter_result.value) {
      std::cout << absl::StrJoin(args, "; ", ValueFormatterHex);
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(
      "No input found which results in a mismatch between the JIT and "
      "interpreter.");
}

absl::StatusOr<ArgSet> ArgSetFromString(std::string_view args_string) {
  ArgSet arg_set;
  for (const std::string_view& value_string :
       absl::StrSplit(args_string, ';')) {
    XLS_ASSIGN_OR_RETURN(Value arg, Parser::ParseTypedValue(value_string));
    arg_set.push_back(arg);
  }
  return arg_set;
}

absl::StatusOr<std::vector<ArgSet>> InputArgsFromTestvector(
    const testvector::SampleInputsProto& testvector) {
  if (!testvector.has_function_args()) {
    return absl::InvalidArgumentError("Expected function_args in testvector");
  }
  std::vector<ArgSet> result;
  for (std::string_view arg_line : testvector.function_args().args()) {
    XLS_ASSIGN_OR_RETURN(ArgSet arg_set, ArgSetFromString(arg_line));
    result.push_back(arg_set);
  }
  return result;
}

absl::StatusOr<std::vector<ArgSet>> InputArgsFromInputFile(
    std::string_view input_file) {
  std::vector<ArgSet> result;
  XLS_ASSIGN_OR_RETURN(std::string inputs_text, GetFileContents(input_file));
  for (const auto& args_line :
       absl::StrSplit(inputs_text, '\n', absl::SkipWhitespace())) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> args, ArgSetFromString(args_line));
    result.push_back(args);
  }
  return result;
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  if (positional_arguments.empty()) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <ir-path>",
                                      argv[0]);
  }

  QCHECK(absl::GetFlag(FLAGS_testvector_textproto).empty() ^
         absl::GetFlag(FLAGS_input_file).empty())
      << "Need to provide either --input_file or --testvector_textproto";

  std::vector<ArgSet> inputs;
  if (!absl::GetFlag(FLAGS_testvector_textproto).empty()) {
    xls::testvector::SampleInputsProto data;
    QCHECK_OK(xls::ParseTextProtoFile(absl::GetFlag(FLAGS_testvector_textproto),
                                      &data));
    auto args_status = xls::InputArgsFromTestvector(data);
    QCHECK_OK(args_status.status())
        << "Failed to parse testvector "
        << absl::GetFlag(FLAGS_testvector_textproto);
    inputs = args_status.value();
  } else if (!absl::GetFlag(FLAGS_input_file).empty()) {
    auto args_status =
        xls::InputArgsFromInputFile(absl::GetFlag(FLAGS_input_file));
    QCHECK_OK(args_status.status()) << "Failed to parse input";
    inputs = args_status.value();
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments[0], inputs));
}
