// Copyright 2026 The XLS Authors
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
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/experimental/busperf/busperf_yaml_generator.h"

static constexpr std::string_view kUsage = R"(
Generates a busperf YAML bus description from an XLS ModuleSignatureProto
(codegen_main --output_signature_path=...). Child block instantiations carry
their own signature inline, so a single top-level signature textproto
is enough to cover the whole design.

Usage:
  xls_sig_to_busperf --scope=tb_top.dut SIGNATURE.textproto
  xls_sig_to_busperf --scope=tb_top.dut --output=bus.yaml top.sig.textproto
)";

ABSL_FLAG(std::string, scope, "",
          "Dot-separated VCD scope path to the DUT instance, e.g. "
          "'tb_passthrough.dut'. Required.");
ABSL_FLAG(std::string, output, "",
          "Output YAML path. If empty, writes to stdout.");

namespace xls::busperf {
namespace {

// Parses `signature_path`, generates the busperf YAML, and writes it to
// `--output` (or stdout if unset).
absl::Status RealMain(std::string_view signature_path) {
  verilog::ModuleSignatureProto signature;
  XLS_RETURN_IF_ERROR(ParseTextProtoFile(signature_path, &signature));

  std::string scope_flag = absl::GetFlag(FLAGS_scope);
  QCHECK(!scope_flag.empty()) << "Must specify --scope";
  std::vector<std::string> scope =
      absl::StrSplit(scope_flag, '.', absl::SkipEmpty());

  XLS_ASSIGN_OR_RETURN(std::string yaml, GenerateBusperfYaml(signature, scope));

  std::string output_path = absl::GetFlag(FLAGS_output);
  if (output_path.empty()) {
    std::cout << yaml;
    return absl::OkStatus();
  }
  return SetFileContents(output_path, yaml);
}

}  // namespace
}  // namespace xls::busperf

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  QCHECK_EQ(positional_arguments.size(), 1)
      << "Expected a single positional argument: SIGNATURE.textproto. See "
         "--help";
  return xls::ExitStatus(xls::busperf::RealMain(positional_arguments[0]));
}
