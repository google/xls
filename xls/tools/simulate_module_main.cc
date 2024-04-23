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
#include <filesystem>  // NOLINT
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulators.h"
#include "xls/tools/eval_utils.h"

static constexpr std::string_view kUsage = R"(
Runs an Verilog block emitted by XLS through a Verilog simulator. Requires both
the Verilog text and the module signature which includes metadata about the
block.

Simulate a module with a single set of arguments:
  simulate_module_main --signature_file=SIG_FILE \
      --args='bits[32]:0x42; bits[8]:123' VERILOG_FILE

Simulate a module with a batch of arguments, one argument set per line in
ARGS_FILE:
  simulate_module_main  --signature_file=SIG_FILE \
      --args_file=ARGS_FILE VERILOG_FILE
)";

ABSL_FLAG(
    std::string, signature_file, "",
    "The path to the file containing the text-encoded ModuleSignatureProto "
    "describing the interface of the module to simulate.");
ABSL_FLAG(std::string, args_file, "",
          "Batch of arguments to pass to the module, one set per line. Each "
          "line should contain a semicolon-separated set of arguments. Cannot "
          "be specified with --args or --channel_values_file.");
ABSL_FLAG(std::string, args, "",
          "The semicolon-separated arguments to pass to the module. The "
          "number of arguments must match the number of and types of the "
          "inputs of the module. Cannot be specified with --args_file or "
          "--channel_values_file.");
ABSL_FLAG(
    std::string, channel_values_file, "",
    "Path to file containing inputs for the channels.\n"
    "The file format is:\n"
    "CHANNEL_NAME : {\n"
    "  VALUE\n"
    "}\n"
    "where CHANNEL_NAME is the name of the channel and VALUE is one XLS Value "
    "in human-readable form. There is one VALUE per line. There may be zero or "
    "more occurrences of VALUE for a channel. The file may contain one or more "
    "channels. Cannot be specified with --args_file or --args.");
ABSL_FLAG(std::vector<std::string>, output_channel_counts, {},
          "Comma separated list of output_channel_name=count pairs, for "
          "example: result=2. 'output_channel_name' represents an output "
          "channel name, and 'count' is an integer representing the number of "
          "values expected from the given channel during simulation. Must be "
          "specified with 'channel_values_file'.");
ABSL_FLAG(std::string, verilog_simulator, "",
          "The Verilog simulator to use. If not specified, the default "
          "simulator is used.");
ABSL_FLAG(std::string, file_type, "",
          "The type of input file, may be either 'verilog' or "
          "'system_verilog'. If not specified the file type is determined by "
          "the file extensoin of the input file");

namespace xls {
namespace {

struct FunctionInput {
  std::vector<std::string> args_strings;
};

struct ProcInput {
  absl::flat_hash_map<std::string, std::vector<Value>> channel_inputs;
  absl::flat_hash_map<std::string, int64_t> output_channel_counts;
};

using InputType = std::variant<FunctionInput, ProcInput>;

absl::Status RunProc(const verilog::ModuleSimulator& simulator,
                     const verilog::ModuleSignature& signature,
                     ProcInput proc_input) {
  using MapT = absl::flat_hash_map<std::string, std::vector<Value>>;
  XLS_ASSIGN_OR_RETURN(
      MapT channel_outputs,
      simulator.RunInputSeriesProc(proc_input.channel_inputs,
                                   proc_input.output_channel_counts));
  std::cout << ChannelValuesToString(channel_outputs) << '\n';
  return absl::OkStatus();
}

absl::Status RunFunction(const verilog::ModuleSimulator& simulator,
                         const verilog::ModuleSignature& signature,
                         FunctionInput function_input) {
  std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
  for (std::string_view args_string : function_input.args_strings) {
    std::vector<Value> arg_values;
    for (std::string_view arg : absl::StrSplit(args_string, ';')) {
      XLS_ASSIGN_OR_RETURN(Value v, Parser::ParseTypedValue(arg));
      arg_values.push_back(v);
    }
    using MapT = absl::flat_hash_map<std::string, Value>;
    XLS_ASSIGN_OR_RETURN(MapT args_set, signature.ToKwargs(arg_values));
    args_sets.push_back(std::move(args_set));
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Value> outputs,
                       simulator.RunBatched(args_sets));

  for (const Value& output : outputs) {
    std::cout << output.ToString(FormatPreference::kHex) << '\n';
  }
  return absl::OkStatus();
}

absl::Status RealMain(std::string_view verilog_text,
                      verilog::FileType file_type,
                      const verilog::ModuleSignature& signature,
                      InputType inputs,
                      const verilog::VerilogSimulator* verilog_simulator) {
  verilog::ModuleSimulator simulator(signature, verilog_text, file_type,
                                     verilog_simulator);

  if (std::holds_alternative<FunctionInput>(inputs)) {
    return RunFunction(simulator, signature, std::get<FunctionInput>(inputs));
  }
  return RunProc(simulator, signature, std::get<ProcInput>(inputs));
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  const xls::verilog::VerilogSimulator* verilog_simulator;
  if (absl::GetFlag(FLAGS_verilog_simulator).empty()) {
    verilog_simulator = &xls::verilog::GetDefaultVerilogSimulator();
  } else {
    absl::StatusOr<const xls::verilog::VerilogSimulator*>
        verilog_simulator_status = xls::verilog::GetVerilogSimulator(
            absl::GetFlag(FLAGS_verilog_simulator));
    std::string verilog_simulator_names = absl::StrJoin(
        xls::verilog::GetVerilogSimulatorManagerSingleton().simulator_names(),
        ", ");
    QCHECK_OK(verilog_simulator_status.status())
        << "Available simulators: " << verilog_simulator_names;
    verilog_simulator = verilog_simulator_status.value();
  }
  QCHECK_EQ(positional_arguments.size(), 1)
      << "Expected single Verilog file argument.";
  std::filesystem::path verilog_path(positional_arguments.at(0));
  absl::StatusOr<std::string> verilog_text = xls::GetFileContents(verilog_path);
  QCHECK_OK(verilog_text.status());

  xls::verilog::FileType file_type;
  if (absl::GetFlag(FLAGS_file_type).empty()) {
    if (verilog_path.extension() == ".v") {
      file_type = xls::verilog::FileType::kVerilog;
    } else if (verilog_path.extension() == ".sv") {
      file_type = xls::verilog::FileType::kSystemVerilog;
    } else {
      LOG(QFATAL) << absl::StreamFormat(
          "Unable to determine file type from filename `%s`. Expected `.v` or "
          "`.sv` file extension.",
          verilog_path);
    }
  } else {
    if (absl::GetFlag(FLAGS_file_type) == "verilog") {
      file_type = xls::verilog::FileType::kVerilog;
    } else if (absl::GetFlag(FLAGS_file_type) == "system_verilog") {
      file_type = xls::verilog::FileType::kSystemVerilog;
    } else {
      LOG(QFATAL) << "Invalid value for --file_type. Expected `verilog` or "
                     "`system_verilog`.";
    }
  }

  int64_t arg_count = absl::GetFlag(FLAGS_args).empty() ? 0 : 1;
  arg_count += absl::GetFlag(FLAGS_args_file).empty() ? 0 : 1;
  arg_count += absl::GetFlag(FLAGS_channel_values_file).empty() ? 0 : 1;
  QCHECK_EQ(arg_count, 1)
      << "Must specify one of: --args_file or --args or --channel_values_file.";

  if (absl::GetFlag(FLAGS_channel_values_file).empty()) {
    QCHECK(absl::GetFlag(FLAGS_output_channel_counts).empty())
        << "'--output_channel_counts' can only be specified with "
           "'--channel_values_file'.";
  }

  xls::InputType input;
  if (!absl::GetFlag(FLAGS_args).empty()) {
    input =
        xls::FunctionInput{std::vector<std::string>{absl::GetFlag(FLAGS_args)}};
  } else if (!absl::GetFlag(FLAGS_args_file).empty()) {
    absl::StatusOr<std::string> args_file_contents_or =
        xls::GetFileContents(absl::GetFlag(FLAGS_args_file));
    QCHECK_OK(args_file_contents_or.status());
    input = xls::FunctionInput{absl::StrSplit(args_file_contents_or.value(),
                                              '\n', absl::SkipWhitespace())};
  } else {
    absl::StatusOr<std::string> channel_values_file_contents =
        xls::GetFileContents(absl::GetFlag(FLAGS_channel_values_file));
    QCHECK_OK(channel_values_file_contents.status());
    absl::StatusOr<absl::btree_map<std::string, std::vector<xls::Value>>>
        channel_values_or =
            xls::ParseChannelValues(channel_values_file_contents.value());
    QCHECK_OK(channel_values_or.status());
    absl::flat_hash_map<std::string, int64_t> output_channel_counts;
    for (std::string_view output_channel_count :
         absl::GetFlag(FLAGS_output_channel_counts)) {
      std::vector<std::string> split =
          absl::StrSplit(output_channel_count, '=');
      QCHECK_EQ(split.size(), 2) << "Format of 'output_channel_counts' "
                                    "should be output_channel_name=count";
      int64_t count;
      bool successful = absl::SimpleAtoi(split[1], &count);
      QCHECK(successful) << absl::StrFormat(
          "For entry '%s', '%s' is expected to be an integer value.",
          output_channel_count, split[1]);
      output_channel_counts[split[0]] = count;
    }
    absl::flat_hash_map<std::string, std::vector<xls::Value>> channel_values;
    channel_values.reserve(channel_values_or->size());
    absl::c_move(*std::move(channel_values_or),
                 std::inserter(channel_values, channel_values.end()));
    input = xls::ProcInput{.channel_inputs = channel_values,
                           .output_channel_counts = output_channel_counts};
  }

  QCHECK(!absl::GetFlag(FLAGS_signature_file).empty())
      << "Must specify --signature_file";
  xls::verilog::ModuleSignatureProto signature_proto;
  QCHECK_OK(xls::ParseTextProtoFile(absl::GetFlag(FLAGS_signature_file),
                                    &signature_proto));
  auto signature_status =
      xls::verilog::ModuleSignature::FromProto(signature_proto);
  QCHECK_OK(signature_status.status());

  return xls::ExitStatus(xls::RealMain(verilog_text.value(), file_type,
                                       signature_status.value(), input,
                                       verilog_simulator));
}
