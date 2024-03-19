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

// Extracts logic formula that correspond to cells in the cell library text file
// for use in logical equivalence checking.

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/netlist/lib_parser.h"

const char kUsage[] = R"(
Extracts the boolean formula that governs a cell's output pins.

Example invocation:

   $ cell_library_extract_formula /path/to/cell_library.lib AND2
   AND2::o = (i0 & i1)
)";

ABSL_FLAG(bool, stream_from_file, false,
          "Uses a file stream instead of loading into memory (to reduce memory "
          "usage)");

namespace xls {
namespace netlist {
namespace cell_lib {
namespace {

// Dumps the boolean formula for all output pins of the cell to stdout.
absl::Status DumpOutputPinExpressions(std::string_view cell_name,
                                      const Block& entry) {
  for (const Block* pin_entry : entry.GetSubBlocks("pin")) {
    if (pin_entry->args.size() != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Pin block in cell %s did not note its name.", cell_name));
    }
    if (pin_entry->GetKVOrDie("direction") == "output") {
      std::cout << cell_name << "::" << pin_entry->args[0] << " = "
                << pin_entry->GetKVOrDie("function") << '\n';
    }
  }
  return absl::OkStatus();
}

absl::Status RealMain(std::string_view path, std::string_view cell_name,
                      bool stream_from_file) {
  // Either make a char stream that loads the file entirely into memory or
  // streams it from disk. Since these files can get quite large this can be
  // useful.
  std::function<absl::StatusOr<CharStream>()> make_cs;
  std::optional<std::string> text;
  if (stream_from_file) {
    make_cs = [&] { return CharStream::FromPath(path); };
  } else {
    auto read_start = absl::Now();
    XLS_ASSIGN_OR_RETURN(text, GetFileContents(path));
    auto read_end = absl::Now();
    LOG(INFO) << "Read delta: " << (read_end - read_start);
    make_cs = [&] { return CharStream::FromText(std::move(text.value())); };
  }

  XLS_ASSIGN_OR_RETURN(CharStream cs, make_cs());
  Scanner scanner(&cs);
  auto allowlist = absl::flat_hash_set<std::string>{"library", "cell", "pin",
                                                    "function", "direction"};
  Parser parser(&scanner, allowlist);

  // Actually do the parse.
  auto start = absl::Now();
  XLS_ASSIGN_OR_RETURN(auto library, parser.ParseLibrary());
  auto end = absl::Now();
  LOG(INFO) << "Parse delta: " << (end - start);

  // Look for the cell we're interested in so we can dump its boolean formula.
  for (const Block* entry : library->GetSubBlocks("cell")) {
    if (!entry->args.empty() && entry->args[0] == cell_name) {
      // Note how long it took for us to find the cell.
      auto end2 = absl::Now();
      LOG(INFO) << "Query delta: " << (end2 - end);

      LOG(INFO) << "Found cell with " << entry->CountEntries("pin")
                << " pin entries";
      return DumpOutputPinExpressions(cell_name, *entry);
    }
  }

  std::cout << "No cell named \"" << cell_name << "\" found among "
            << library->CountEntries("cell") << " cells." << '\n';
  return absl::OkStatus();
}

}  // namespace
}  // namespace cell_lib
}  // namespace netlist
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 2 || positional_arguments[0].empty()) {
    LOG(QFATAL) << "Expected arguments: " << argv[0]
                << " <lib_path> <cell_name>";
  }

  return xls::ExitStatus(xls::netlist::cell_lib::RealMain(
      positional_arguments[0], positional_arguments[1],
      absl::GetFlag(FLAGS_stream_from_file)));
}
