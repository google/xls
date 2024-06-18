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

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/public/ir_parser.h"
#include "xls/tools/extract_segment.h"

const char kUsage[] = R"(
Extract a segment of a graph either emerging from or draining to a set of nodes.

The result is a function containing a subset of the logic of the input function.

Recieves are always considered as function inputs.

Tokens are considered to create edges in the graph and are considered 1-bit
integers.
)";

ABSL_FLAG(
    std::vector<std::string>, sink_nodes, {},
    "Node ids/names which are considered sinks. If present only nodes which "
    "feed into these nodes will be emitted. The values of all nodes will be "
    "emitted as return values. Values are comma separated.");
ABSL_FLAG(std::vector<std::string>, source_nodes, {},
          "Nodes ids/names which are considered sources. If present only nodes "
          "which are fed by these nodes will be emmitted. Other required "
          "values will be emitted as parameters. Values are comma separated.");
ABSL_FLAG(bool, next_nodes_return_inputs, true,
          "Have 'next' nodes which are sink nodes be emitted as (predicate, "
          "value) tuples.");
ABSL_FLAG(std::string, extracted_package_name, "extracted_package",
          "Name of the package holding the extracted segment");
ABSL_FLAG(std::string, extracted_function_name, "extracted_func",
          "Name of the extracted segment");
ABSL_FLAG(std::optional<std::string>, top, std::nullopt, "top function/proc");

namespace xls {
namespace {

absl::Status RealMain(std::string_view ir_file) {
  XLS_ASSIGN_OR_RETURN(auto ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, ParsePackage(ir_text, ir_file));
  FunctionBase* fb;
  if (absl::GetFlag(FLAGS_top)) {
    XLS_ASSIGN_OR_RETURN(
        fb, package->GetFunctionBaseByName(*absl::GetFlag(FLAGS_top)));
  } else {
    XLS_RET_CHECK(package->GetTop());
    fb = *package->GetTop();
  }
  auto get_node = [&](std::string_view s) -> absl::StatusOr<Node*> {
    int64_t id;
    if (absl::SimpleAtoi(s, &id)) {
      XLS_ASSIGN_OR_RETURN(Node * n, fb->GetNodeById(id));
      return n;
    }
    return fb->GetNode(s);
  };
  std::vector<Node*> sources;
  sources.reserve(absl::GetFlag(FLAGS_source_nodes).size());
  for (std::string_view s : absl::GetFlag(FLAGS_source_nodes)) {
    XLS_ASSIGN_OR_RETURN(Node * n, get_node(s));
    sources.push_back(n);
  }
  std::vector<Node*> sinks;
  sinks.reserve(absl::GetFlag(FLAGS_sink_nodes).size());
  Node* ret = nullptr;
  if (fb->IsFunction()) {
    ret = fb->AsFunctionOrDie()->return_value();
  }
  for (std::string_view s : absl::GetFlag(FLAGS_sink_nodes)) {
    XLS_ASSIGN_OR_RETURN(Node * n, get_node(s));
    sinks.push_back(n);
    if (n == ret) {
      LOG(QFATAL) << "node " << s
                  << " is the return value, all nodes sink to it";
    }
  }
  XLS_ASSIGN_OR_RETURN(
      auto new_package,
      ExtractSegmentInNewPackage(
          fb, sources, sinks, absl::GetFlag(FLAGS_extracted_package_name),
          absl::GetFlag(FLAGS_extracted_function_name),
          absl::GetFlag(FLAGS_next_nodes_return_inputs)));
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

  if (absl::GetFlag(FLAGS_sink_nodes).empty() &&
      absl::GetFlag(FLAGS_source_nodes).empty()) {
    LOG(QFATAL)
        << "At least one of --sink_nodes or --source_nodes must be present.";
  }

  return xls::ExitStatus(xls::RealMain(positional_arguments.front()));
}
