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
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/annotate_ir.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

static constexpr std::string_view kUsage = R"(
Emits valid XLS IR annotated with a configured IrAnnotator. Parameter
annotations are emitted as comment lines above the function signature and node
annotations are emitted as trailing `// ...` comments.

Example invocation:
  annotate_ir_main --annotator=delay path/to/file.ir
)";

ABSL_FLAG(std::string, annotator, "visibility",
          "Annotator to apply to the IR (options: none, visibility, delay, "
          "critical_path).");
ABSL_FLAG(std::string, delay_model, "unit",
          "Delay model name used when --annotator=delay or critical_path.");

namespace xls {
namespace {

absl::Status RealMain(std::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(input_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text, input_path));
  XLS_ASSIGN_OR_RETURN(AnnotatorKind kind,
                       ParseAnnotatorKind(absl::GetFlag(FLAGS_annotator)));

  XLS_ASSIGN_OR_RETURN(
      std::string annotated_ir,
      AnnotateIr(package.get(), kind, absl::GetFlag(FLAGS_delay_model)));
  std::cout << annotated_ir;
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);
  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <ir_path>",
                                      argv[0]);
  }
  return xls::ExitStatus(xls::RealMain(positional_arguments[0]));
}
