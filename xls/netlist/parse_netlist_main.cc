// Copyright 2020 Google LLC
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
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/find_logic_clouds.h"
#include "xls/netlist/netlist_parser.h"

ABSL_FLAG(bool, show_clusters, false, "Show the logic clusters found.");

namespace xls {
namespace {

absl::Status RealMain(absl::string_view netlist_path,
                      absl::string_view cell_library_path) {
  XLS_ASSIGN_OR_RETURN(std::string netlist, GetFileContents(netlist_path));
  XLS_ASSIGN_OR_RETURN(
      netlist::CellLibraryProto cell_library_proto,
      ParseTextProtoFile<netlist::CellLibraryProto>(cell_library_path));
  XLS_ASSIGN_OR_RETURN(auto cell_library,
                       netlist::CellLibrary::FromProto(cell_library_proto));
  netlist::rtl::Scanner scanner(netlist);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<netlist::rtl::Module> module,
      netlist::rtl::Parser::ParseModule(&cell_library, &scanner));
  std::cout << "nets:  " << module->nets().size() << std::endl;
  std::cout << "cells: " << module->cells().size() << std::endl;
  absl::flat_hash_map<netlist::CellKind, int64> cell_kind_to_count;
  for (const auto& name_and_cell : module->cells()) {
    cell_kind_to_count[name_and_cell->kind()]++;
  }
  std::cout << "cell-kind breakdown:" << std::endl;
  for (int64 i = static_cast<int64>(netlist::CellKind::kFlop);
       i <= static_cast<int64>(netlist::CellKind::kOther); ++i) {
    netlist::CellKind cell_kind = static_cast<netlist::CellKind>(i);
    std::cout << absl::StreamFormat(" %8s: %d",
                                    netlist::CellKindToString(cell_kind),
                                    cell_kind_to_count[cell_kind])
              << std::endl;
  }

  std::vector<netlist::rtl::Cluster> clusters =
      netlist::rtl::FindLogicClouds(*module);
  std::cout << "logic clusters: " << clusters.size() << std::endl;
  if (absl::GetFlag(FLAGS_show_clusters)) {
    std::cout << netlist::rtl::ClustersToString(clusters) << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);

  if (positional_arguments.size() < 2) {
    std::cerr << "Usage: " << argv[0] << " <netlist.gv> <cell_library.pbtxt>"
              << std::endl;
    return EXIT_FAILURE;
  }

  XLS_QCHECK_OK(
      xls::RealMain(positional_arguments[0], positional_arguments[1]));

  return EXIT_SUCCESS;
}
