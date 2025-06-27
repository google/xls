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

#include "xls/tools/bom.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/op.h"

namespace xls {

bool operator==(const BomItem& lhs, const BomItem& rhs) {
  return lhs.op_kind == rhs.op_kind && lhs.op_name == rhs.op_name &&
         lhs.width_input == rhs.width_input &&
         lhs.width_output == rhs.width_output;
}
bool operator<(const BomItem& lhs, const BomItem& rhs) {
  if (lhs.op_kind != rhs.op_kind) {
    return lhs.op_kind < rhs.op_kind;
  }
  if (lhs.op_name != rhs.op_name) {
    return lhs.op_name < rhs.op_name;
  }
  if (lhs.width_input != rhs.width_input) {
    return lhs.width_input < rhs.width_input;
  }
  return lhs.width_output < rhs.width_output;
}

absl::StatusOr<absl::btree_map<BomItem, int64_t>> BomCalculateSummary(
    const absl::flat_hash_map<std::string, verilog::XlsMetricsProto>&
        metrics_data) {
  absl::btree_map<BomItem, int64_t> summary;
  for (const auto& [path, metrics] : metrics_data) {
    if (!metrics.has_block_metrics()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Proto ModuleSignatureProto from %s has empty block metrics.", path));
    }
    if (metrics.block_metrics().bill_of_materials().empty()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Proto ModuleSignatureProto from %s has no bill of materials.",
          path));
    }

    for (const verilog::BomEntryProto& item :
         metrics.block_metrics().bill_of_materials()) {
      if (!item.has_op()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("ModuleSignatureProto from %s has a BomEmptyProto "
                            "missing op value.",
                            path));
      }
      if (!item.has_kind()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("ModuleSignatureProto from %s has a BomEmptyProto "
                            "missing klind value.",
                            path));
      }
      std::string op_kind = verilog::BomKindProto_Name(item.kind());
      std::string op_name = OpToString(FromOpProto(item.op()));
      BomItem key = {.op_kind = op_kind,
                     .op_name = op_name,
                     .width_input = item.maximum_input_width(),
                     .width_output = item.output_width()};
      summary[key] += 1;
    }
  }
  return summary;
}

absl::StatusOr<absl::flat_hash_map<std::string, verilog::XlsMetricsProto>>
CollectMetricsProtos(const std::filesystem::path& root,
                     const std::string& match) {
  absl::flat_hash_map<std::string, verilog::XlsMetricsProto> metrics_protos;
  XLS_ASSIGN_OR_RETURN(std::vector<std::filesystem::path> filenames,
                       FindFilesMatchingRegex(root, match));
  for (const std::filesystem::path& path : filenames) {
    verilog::XlsMetricsProto metrics;

    if (absl::EndsWith(path.filename().c_str(), ".textproto")) {
      XLS_RETURN_IF_ERROR(ParseTextProtoFile(path, &metrics));
    } else if (absl::EndsWith(path.filename().c_str(), ".proto")) {
      XLS_RETURN_IF_ERROR(ParseProtobinFile(path.filename(), &metrics));
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat(path.c_str(), " is not a .proto or .textproto file"));
    }
    metrics_protos[std::string{path}] = metrics;
  }
  return metrics_protos;
}

constexpr int64_t kColumns = 5;  // 4 fields in xls::bom::item and the count

// Names for the column headers
const std::array<std::string, kColumns> kHeaders = {"Kind", "Op", "Input Width",
                                                    "Output Width", "Count"};

namespace {

void PrintLine(std::ostream& out, std::array<int64_t, kColumns> max_width) {
  out << " +";
  for (int64_t i = 0; i < kColumns; ++i) {
    out << "-";
    for (int64_t j = 0; j < max_width[i]; ++j) {
      out << "-";
    }
    out << "-+";
  }
  out << "\n";
}

}  // namespace

void BomPrint(std::ostream& out,
              const absl::btree_map<BomItem, int64_t>& summary) {
  std::vector<std::array<std::string, kColumns>> cells = {kHeaders};

  std::array<int64_t, kColumns> max_width = {
      static_cast<int64_t>(kHeaders[0].size()),
      static_cast<int64_t>(kHeaders[1].size()),
      static_cast<int64_t>(kHeaders[2].size()),
      static_cast<int64_t>(kHeaders[3].size()),
      static_cast<int64_t>(kHeaders[4].size())};

  // Stringify the cells and keep track of the max width
  for (const std::pair<const BomItem, int64_t>& it : summary) {
    std::array<std::string, kColumns> row = {
        it.first.op_kind, it.first.op_name,
        std::to_string(it.first.width_input),
        std::to_string(it.first.width_output), std::to_string(it.second)};
    cells.push_back(row);
    for (int i = 0; i < kColumns; ++i) {
      max_width[i] =
          std::max(max_width[i], static_cast<int64_t>(row[i].size()));
    }
  }

  std::array<std::string, kColumns> last_values;
  for (const std::array<std::string, kColumns>& it : cells) {
    if (last_values[0] != it[0]) {
      PrintLine(out, max_width);
    }

    out << " |";
    for (int64_t i = 0; i < kColumns; ++i) {
      out << " " << std::setw(static_cast<int>(max_width[i]));
      if (last_values[i] == it[i] && i < 2) {
        out << "";
      } else {
        out << it[i];
      }
      out << " |";
      last_values[i] = it[i];
    }
    out << "\n";
  }
  PrintLine(out, max_width);
}

void BomToCsv(std::ostream& out,
              const absl::btree_map<BomItem, int64_t>& summary) {
  // Output the column headers
  for (int64_t i = 0; i < kColumns - 1; ++i) {
    out << kHeaders[i] << ",";
  }
  out << kHeaders[kColumns - 1] << "\n";

  // Output the actual data
  for (const std::pair<const BomItem, int64_t>& it : summary) {
    out << it.first.op_kind << ",";
    out << it.first.op_name << ",";
    out << it.first.width_input << ",";
    out << it.first.width_output << ",";
    out << it.second << "\n";
  }
}

}  // namespace xls
