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

#ifndef XLS_TOOLS_BOM_H_
#define XLS_TOOLS_BOM_H_

#include <cstdint>
#include <filesystem>  // NOLINT
#include <ostream>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.pb.h"

namespace xls {

using ::xls::verilog::ModuleSignatureProto;

struct BomItem {
  // The op values are stored as strings because this code isn't performance
  // critical and we ultimately want the items sorted alphabetically by their
  // names rather than their protobuf values.
  std::string op_kind;
  std::string op_name;
  int64_t width_input;
  int64_t width_output;
};

bool operator==(const BomItem& lhs, const BomItem& rhs);
bool operator<(const BomItem& lhs, const BomItem& rhs);

// Calculate a BOM summary from a given list of ModuleSignatureProto files.
// Using absl::btree_map as we *want* the items to be stored in sorted order.
absl::StatusOr<absl::btree_map<BomItem, int64_t>> BomCalculateSummary(
    const absl::flat_hash_map<std::string, ModuleSignatureProto>&
        signature_data);

// Finds all the ModuleSignatureProto files under a given path.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kFailedPrecondition
absl::StatusOr<absl::flat_hash_map<std::string, ModuleSignatureProto>>
CollectSignatureProtos(const std::filesystem::path& root,
                       const std::string& match);

// Print the BOM summary as a text table suitable for human consumption.
// The output looks like the following;
// +---------------------+-----+-------------+--------------+-------+
// |                Kind |  Op | Input Width | Output Width | Count |
// +---------------------+-----+-------------+--------------+-------+
// |      BOM_KIND_ADDER | add |           8 |            8 |     1 |
// +---------------------+-----+-------------+--------------+-------+
// | BOM_KIND_COMPARISON |  ne |           8 |            1 |     2 |
// |                     |     |          16 |            1 |     1 |
// +---------------------+-----+-------------+--------------+-------+
void BomPrint(std::ostream& out,
              const absl::btree_map<BomItem, int64_t>& summary);

// Outputs the BOM summary in a machine readable CSV (common separate
// values) file for loading into other tools (like Google Sheets). Useful for
// doing further analysis on the BOM contents.
void BomToCsv(std::ostream& out,
              const absl::btree_map<BomItem, int64_t>& summary);

}  // namespace xls

#endif  // XLS_TOOLS_BOM_H_
