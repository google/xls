// Copyright 2023 The XLS Authors
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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Find all `*.sig.textproto` files in a directory structure and prints out a
// summary of modules used.

#include "absl/container/btree_map.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/tools/bom.h"

namespace {

enum class PrintBomOutputMode : int8_t { kTable, kCsv };

bool AbslParseFlag(std::string_view text, PrintBomOutputMode* mode,
                   std::string* error) {
  if (text == "table") {
    *mode = PrintBomOutputMode::kTable;
    return true;
  }
  if (text == "csv") {
    *mode = PrintBomOutputMode::kCsv;
    return true;
  }
  *error = "unknown value for enumeration";
  return false;
}

std::string AbslUnparseFlag(PrintBomOutputMode mode) {
  switch (mode) {
    case PrintBomOutputMode::kTable:
      return "table";
    case PrintBomOutputMode::kCsv:
      return "csv";
    default:
      return absl::StrCat(mode);
  }
}

}  // namespace

ABSL_FLAG(std::string, root_path, ".",
          "Path to start the recursive search at.");

ABSL_FLAG(std::string, file_pattern, ".*\\.sig\\.(text)?proto",
          "Regex pattern used to find `XlsMetricsProto` (text)proto files.");

ABSL_FLAG(PrintBomOutputMode, output_as, PrintBomOutputMode::kTable,
          "Format to output the BOM.");

ABSL_FLAG(std::vector<std::string>, op_kind, {},
          "Only include the listed op kinds.");

namespace xls {

static absl::Status RealMain() {
  using MapT = absl::flat_hash_map<std::string, verilog::XlsMetricsProto>;
  XLS_ASSIGN_OR_RETURN(MapT metrics_data,
                       CollectMetricsProtos(absl::GetFlag(FLAGS_root_path),
                                            absl::GetFlag(FLAGS_file_pattern)));
  if (metrics_data.empty()) {
    return absl::NotFoundError(
        "No protobuf files, check --root_path and --file_pattern values.");
  }

  std::cerr << "\nFound " << metrics_data.size() << " protobuf files.\n";
  for (const auto& protos : metrics_data) {
    std::cerr << " * " << protos.first << "\n";
  }
  std::cerr << "\n";

  using BMapT = absl::btree_map<BomItem, int64_t>;
  XLS_ASSIGN_OR_RETURN(BMapT bom_summary, BomCalculateSummary(metrics_data));

  std::vector<std::string> op_kinds = absl::GetFlag(FLAGS_op_kind);
  if (!op_kinds.empty()) {
    std::vector<BomItem> to_remove;
    for (std::pair<const BomItem, int64_t>& it : bom_summary) {
      if (std::find(op_kinds.begin(), op_kinds.end(), it.first.op_kind) ==
          op_kinds.end()) {
        to_remove.push_back(it.first);
      }
    }
    for (BomItem& it : to_remove) {
      bom_summary.erase(it);
    }
  }
  switch (absl::GetFlag(FLAGS_output_as)) {
    case PrintBomOutputMode::kTable: {
      BomPrint(std::cout, bom_summary);
      break;
    }
    case PrintBomOutputMode::kCsv: {
      BomToCsv(std::cout, bom_summary);
      break;
    }
  }

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  xls::InitXls(argv[0], argc, argv);
  return xls::ExitStatus(xls::RealMain());
}
