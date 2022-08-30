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

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_simulators.h"

const char kUsage[] = R"(
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
          "be specified with --args.");
ABSL_FLAG(std::string, args, "",
          "The semicolon-separated arguments to pass to the module. The "
          "number of arguments must match the number of and types of the "
          "inputs of the module. Cannot be specified with --args_file.");
ABSL_FLAG(std::string, verilog_simulator, "",
          "The Verilog simulator to use. If not specified, the default "
          "simulator is used.");
ABSL_FLAG(std::string, file_type, "",
          "The type of input file, may be either 'verilog' or "
          "'system_verilog'. If not specified the file type is determined by "
          "the file extensoin of the input file");

namespace xls {
namespace {

absl::Status RealMain(absl::string_view verilog_text,
                      verilog::FileType file_type,
                      const verilog::ModuleSignature& signature,
                      absl::Span<const std::string> args_strings,
                      const verilog::VerilogSimulator* verilog_simulator) {
  verilog::ModuleSimulator simulator(signature, verilog_text, file_type,
                                     verilog_simulator);

  std::vector<absl::flat_hash_map<std::string, Value>> args_sets;
  for (absl::string_view args_string : args_strings) {
    std::vector<Value> arg_values;
    for (absl::string_view arg : absl::StrSplit(args_string, ';')) {
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
    std::cout << output.ToString(FormatPreference::kHex) << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
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
    XLS_QCHECK_OK(verilog_simulator_status.status())
        << "Available simulators: " << verilog_simulator_names;
    verilog_simulator = verilog_simulator_status.value();
  }
  XLS_QCHECK_EQ(positional_arguments.size(), 1)
      << "Expected single Verilog file argument.";
  std::filesystem::path verilog_path(positional_arguments.at(0));
  absl::StatusOr<std::string> verilog_text = xls::GetFileContents(verilog_path);
  XLS_QCHECK_OK(verilog_text.status());

  xls::verilog::FileType file_type;
  if (absl::GetFlag(FLAGS_file_type).empty()) {
    if (verilog_path.extension() == ".v") {
      file_type = xls::verilog::FileType::kVerilog;
    } else if (verilog_path.extension() == ".sv") {
      file_type = xls::verilog::FileType::kSystemVerilog;
    } else {
      XLS_LOG(QFATAL) << absl::StreamFormat(
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
      XLS_LOG(QFATAL) << "Invalid value for --file_type. Expected `verilog` or "
                         "`system_verilog`.";
    }
  }

  std::vector<std::string> args_strings;
  XLS_QCHECK(!absl::GetFlag(FLAGS_args).empty() ^
             !absl::GetFlag(FLAGS_args_file).empty())
      << "Must specify either --args_file or --args, but not both.";
  if (!absl::GetFlag(FLAGS_args).empty()) {
    args_strings.push_back(absl::GetFlag(FLAGS_args));
  } else {
    absl::StatusOr<std::string> args_file_contents =
        xls::GetFileContents(absl::GetFlag(FLAGS_args_file));
    XLS_QCHECK_OK(args_file_contents.status());
    args_strings = absl::StrSplit(args_file_contents.value(), '\n',
                                  absl::SkipWhitespace());
  }

  XLS_QCHECK(!absl::GetFlag(FLAGS_signature_file).empty())
      << "Must specify --signature_file";
  xls::verilog::ModuleSignatureProto signature_proto;
  XLS_QCHECK_OK(xls::ParseTextProtoFile(absl::GetFlag(FLAGS_signature_file),
                                        &signature_proto));
  auto signature_status =
      xls::verilog::ModuleSignature::FromProto(signature_proto);
  XLS_QCHECK_OK(signature_status.status());

  XLS_QCHECK_OK(xls::RealMain(verilog_text.value(), file_type,
                              signature_status.value(), args_strings,
                              verilog_simulator));

  return EXIT_SUCCESS;
}
