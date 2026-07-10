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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/experimental/busperf/busperf_yaml_generator.h"

static constexpr std::string_view kUsage = R"(
Generates a busperf (https://github.com/antmicro/busperf) YAML bus
description from an XLS ModuleSignatureProto (codegen_main
--output_signature_path=...).

Usage:
  xls_sig_to_busperf --scope=tb_top.dut SIGNATURE.textproto
  xls_sig_to_busperf --scope=tb_top.dut --output=bus.yaml \
      --child_signature=child_a.sig.textproto,child_b.sig.textproto \
      top.sig.textproto
)";

ABSL_FLAG(std::string, scope, "",
          "Dot-separated VCD scope path to the DUT instance, e.g. "
          "'tb_passthrough.dut'. Required.");
ABSL_FLAG(std::vector<std::string>, child_signature, {},
          "Standalone ModuleSignatureProto textprotos for spawned child "
          "blocks (codegen'd with --top=<ChildProc> on the same package, "
          "without --module_name so each keeps its mangled block name). "
          "Matched against instantiations().block_instantiation()"
          ".block_name() by the child signature's own module_name.");
ABSL_FLAG(std::string, output, "",
          "Output YAML path. If empty, writes to stdout.");

namespace xls::busperf {
namespace {

// Parses `signature_path`/`--child_signature` files, generates the busperf
// YAML, and writes it to `--output` (or stdout if unset).
absl::Status RealMain(std::string_view signature_path) {
  verilog::ModuleSignatureProto signature;
  XLS_RETURN_IF_ERROR(ParseTextProtoFile(signature_path, &signature));

  absl::flat_hash_map<std::string, verilog::ModuleSignatureProto>
      child_signatures;
  for (const std::string& child_path : absl::GetFlag(FLAGS_child_signature)) {
    verilog::ModuleSignatureProto child_signature;
    XLS_RETURN_IF_ERROR(ParseTextProtoFile(child_path, &child_signature));
    const std::string& module_name = child_signature.module_name();
    if (child_signatures.contains(module_name)) {
      LOG(WARNING) << "multiple --child_signature files have module_name '"
                   << module_name << "'; only the last one (" << child_path
                   << ") will be used";
    }
    child_signatures[module_name] = std::move(child_signature);
  }

  std::string scope_flag = absl::GetFlag(FLAGS_scope);
  QCHECK(!scope_flag.empty()) << "Must specify --scope";
  std::vector<std::string> scope =
      absl::StrSplit(scope_flag, '.', absl::SkipEmpty());

  XLS_ASSIGN_OR_RETURN(
      std::string yaml,
      GenerateBusperfYaml(signature, scope, child_signatures));

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
