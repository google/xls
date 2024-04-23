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
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/find_logic_clouds.h"
#include "xls/netlist/netlist_parser.h"

ABSL_FLAG(bool, show_clusters, false, "Show the logic clusters found.");

namespace xls {
namespace {

absl::Status RealMain(std::string_view netlist_path,
                      std::optional<std::string_view> cell_library_path) {
  netlist::CellLibrary cell_library;
  if (cell_library_path) {
    XLS_ASSIGN_OR_RETURN(
        netlist::CellLibraryProto cell_library_proto,
        ParseTextProtoFile<netlist::CellLibraryProto>(*cell_library_path));
    XLS_ASSIGN_OR_RETURN(cell_library,
                         netlist::CellLibrary::FromProto(cell_library_proto));
  }

  XLS_ASSIGN_OR_RETURN(std::string netlist_text, GetFileContents(netlist_path));
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<netlist::rtl::Netlist> netlist,
      netlist::rtl::Parser::ParseNetlist(&cell_library, &scanner));
  netlist::rtl::Module* module = netlist->modules()[0].get();
  std::cout << "nets:  " << module->nets().size() << '\n';
  std::cout << "cells: " << module->cells().size() << '\n';
  absl::flat_hash_map<netlist::CellKind, int64_t> cell_kind_to_count;
  for (const auto& name_and_cell : module->cells()) {
    cell_kind_to_count[name_and_cell->kind()]++;
  }
  std::cout << "cell-kind breakdown:" << '\n';
  for (int64_t i = static_cast<int64_t>(netlist::CellKind::kFlop);
       i <= static_cast<int64_t>(netlist::CellKind::kOther); ++i) {
    netlist::CellKind cell_kind = static_cast<netlist::CellKind>(i);
    std::cout << absl::StreamFormat(" %8s: %d",
                                    netlist::CellKindToString(cell_kind),
                                    cell_kind_to_count[cell_kind])
              << '\n';
  }

  std::vector<netlist::rtl::Cluster> clusters =
      netlist::rtl::FindLogicClouds(*module);
  std::cout << "logic clusters: " << clusters.size() << '\n';
  if (absl::GetFlag(FLAGS_show_clusters)) {
    std::cout << netlist::rtl::ClustersToString(clusters) << '\n';
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);

  if (positional_arguments.empty()) {
    std::cerr << "Usage: " << argv[0] << " netlist.gv [cell_library.textproto]"
              << '\n';
    return EXIT_FAILURE;
  }

  std::optional<std::string_view> cell_library_path;
  if (positional_arguments.size() == 2) {
    cell_library_path = positional_arguments[1];
  }

  return xls::ExitStatus(
      xls::RealMain(positional_arguments[0], cell_library_path));
}
