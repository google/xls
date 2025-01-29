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

#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/extract_state_element.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/public/ir_parser.h"

const char kUsage[] = R"(
Extract a segment of a graph containing the evolution of a given state element.

The result is a proc with the requested state elements being the only state. All
other state elements are bound together into a receive.
)";

ABSL_FLAG(std::vector<std::string>, state_elements, {},
          "State elements to extract");
ABSL_FLAG(bool, send_state_values, true,
          "Send the value of the state elements to streaming channels");
ABSL_FLAG(std::optional<std::string>, top, std::nullopt, "top proc name");

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_file,
                      absl::Span<const std::string> state_elements,
                      const std::optional<std::string>& top,
                      bool send_state_values) {
  XLS_ASSIGN_OR_RETURN(auto ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, ParsePackage(ir_text, ir_file));
  Proc* proc;
  if (top) {
    XLS_ASSIGN_OR_RETURN(proc, package->GetProc(*top));
  } else {
    XLS_ASSIGN_OR_RETURN(proc, package->GetTopAsProc());
  }
  std::vector<StateElement*> state_elements_to_extract;
  state_elements_to_extract.reserve(state_elements.size());
  for (const std::string& state_element : state_elements) {
    XLS_ASSIGN_OR_RETURN(std::back_inserter(state_elements_to_extract),
                         proc->GetStateElement(state_element));
  }
  XLS_ASSIGN_OR_RETURN(auto new_package,
                       ExtractStateElementsInNewPackage(
                           proc, state_elements_to_extract, send_state_values));
  std::cout << new_package->DumpIr();
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << "Expected invocation: " << argv[0] << " <ir_file>";
  }

  if (absl::GetFlag(FLAGS_state_elements).empty()) {
    LOG(QFATAL) << "Must specify at least one --state_element";
  }

  return xls::ExitStatus(xls::RealMain(
      positional_arguments[0], absl::GetFlag(FLAGS_state_elements),
      absl::GetFlag(FLAGS_top), absl::GetFlag(FLAGS_send_state_values)));
}
