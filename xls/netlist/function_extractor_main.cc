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

// Simple driver function for FunctionExtractor; preprocesses a netlist into a
// CellLibraryProto for colocation with the original library.

#include <string>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.pb.h"

ABSL_FLAG(std::string, cell_library, "", "Cell library to preprocess.");
ABSL_FLAG(std::string, output_path, "",
          "Path to the file in which to write the output.");
ABSL_FLAG(bool, output_textproto, false,
          "If true, write the output as a text-format protobuf.");

namespace xls::netlist::function {

static absl::Status RealMain(const std::string& cell_library_path,
                             const std::string& output_path,
                             bool output_textproto) {
  XLS_ASSIGN_OR_RETURN(std::string cell_library_text,
                       GetFileContents(cell_library_path));
  XLS_ASSIGN_OR_RETURN(
      auto char_stream,
      netlist::cell_lib::CharStream::FromText(cell_library_text));
  XLS_ASSIGN_OR_RETURN(netlist::CellLibraryProto lib_proto,
                       netlist::function::ExtractFunctions(&char_stream));

  if (output_textproto) {
    std::string output;
    XLS_RET_CHECK(google::protobuf::TextFormat::PrintToString(lib_proto, &output));
    return SetFileContents(output_path, output);
  }
  return SetFileContents(output_path, lib_proto.SerializeAsString());
}

}  // namespace xls::netlist::function

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string cell_library_path = absl::GetFlag(FLAGS_cell_library);
  QCHECK(!cell_library_path.empty()) << "--cell_library must be specified.";

  std::string output_path = absl::GetFlag(FLAGS_output_path);
  QCHECK(!output_path.empty()) << "--output_path must be specified.";

  return xls::ExitStatus(xls::netlist::function::RealMain(
      cell_library_path, output_path, absl::GetFlag(FLAGS_output_textproto)));
}
