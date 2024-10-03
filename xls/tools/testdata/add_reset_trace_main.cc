// Copyright 2024 The XLS Authors
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

#include <array>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

const char kUsage[] = R"(
Test helper to add a trace of the 'reset' port to an existing block-ir.

This is only for eval_proc_main tests.

TODO(allight): We should transition tests away from using this.
)";

ABSL_FLAG(std::string, reset_name, "rst",
          "name of the port to add a trace for");

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_file, std::string_view rst_name) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(std::vector<FormatStep> format,
                       ParseFormatString("rst_n {:x}"));
  for (auto& b : package->blocks()) {
    XLS_ASSIGN_OR_RETURN(InputPort * port, b->GetInputPort(rst_name));
    XLS_ASSIGN_OR_RETURN(Node * token,
                         b->MakeNode<Literal>(SourceInfo(), Value::Token()));
    XLS_ASSIGN_OR_RETURN(Node * lit_one,
                         b->MakeNode<Literal>(SourceInfo(), Value::Bool(true)));
    XLS_RETURN_IF_ERROR(b->MakeNode<Trace>(SourceInfo(), token, lit_one,
                                           std::array<Node*, 1>{port}, format,
                                           /*verbosity=*/2)
                            .status());
  }
  std::cout << package->DumpIr();
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " input.block.ir";
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments.front(),
                                       absl::GetFlag(FLAGS_reset_name)));
}
