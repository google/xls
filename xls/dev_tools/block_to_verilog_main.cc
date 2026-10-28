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

#include <cstdint>
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
#include "xls/codegen/codegen_result.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/verifier.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.h"
#include "xls/tools/codegen_flags.pb.h"

static constexpr std::string_view kUsage = R"(
Generates Verilog from a given block IR file. Example invocation:

  block_to_verilog_main --output_verilog_path=OUT BLOCK_IR
)";

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));

  XLS_ASSIGN_OR_RETURN(CodegenFlagsProto codegen_flags_proto,
                       GetCodegenFlags());

  XLS_ASSIGN_OR_RETURN(verilog::CodegenResult codegen_result,
                       BlockToVerilog(p.get(), codegen_flags_proto));

  std::string verilog_path = absl::GetFlag(FLAGS_output_verilog_path);
  if (!verilog_path.empty()) {
    for (int64_t i = 0; i < codegen_result.verilog_line_map.mapping_size();
         ++i) {
      codegen_result.verilog_line_map.mutable_mapping(i)->set_verilog_file(
          verilog_path);
    }
  }

  std::string verilog_line_map_path =
      absl::GetFlag(FLAGS_output_verilog_line_map_path);
  if (!verilog_line_map_path.empty()) {
    XLS_RETURN_IF_ERROR(SetTextProtoFile(verilog_line_map_path,
                                         codegen_result.verilog_line_map));
  }

  if (!absl::GetFlag(FLAGS_output_signature_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_signature_path),
                         codegen_result.signature.proto()));
  }

  // Optionally write residual data textproto capturing node emission order.
  if (!absl::GetFlag(FLAGS_output_residual_data_path).empty()) {
    XLS_RETURN_IF_ERROR(
        SetTextProtoFile(absl::GetFlag(FLAGS_output_residual_data_path),
                         codegen_result.residual_data));
  }

  if (verilog_path.empty()) {
    std::cout << codegen_result.verilog_text;
  } else {
    XLS_RETURN_IF_ERROR(
        SetFileContents(verilog_path, codegen_result.verilog_text));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s IR_FILE",
                                      argv[0]);
  }
  std::string_view ir_path = positional_arguments[0];
  return xls::ExitStatus(xls::RealMain(ir_path));
}
