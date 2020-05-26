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

// Driver for NetlistInterpreter: loads a netlist from disk, feeds Value input
// (taken from the command line) into it, and prints the result.

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/flattening.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/interpreter.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.pb.h"
#include "xls/netlist/netlist_parser.h"

ABSL_FLAG(std::string, cell_library, "",
          "Cell library to use for interpretation.");
ABSL_FLAG(std::string, cell_library_proto, "",
          "Preprocessed cell library proto to use for interpretation.");
// TODO(rspringer): Eliminate the need for this flag.
// This one is a hidden temporary flag until we can properly handle cells
// with state_function attributes (e.g., some latches).
ABSL_FLAG(std::string, dump_cells, "",
          "Comma-separated list of cells (not cell library entries!) whose "
          "values to dump.");
ABSL_FLAG(std::string, high_cells, "",
          "Comma-separated list of cells for which to assume \"1\" values on "
          "all outputs.");
ABSL_FLAG(std::string, input, "",
          "The input to the function as a semicolon-separated list of typed "
          "values. For example: \"bits[32]:42; (bits[7]:0, bits[20]:4)\". "
          "Values must be listed in the same order as the module inputs.");
ABSL_FLAG(std::string, output_type, "",
          "Type of the value as an XLS-formatted string. If un-set, then the "
          "output will be printed as flat uninterpreted bits.");
ABSL_FLAG(std::string, module_name, "", "Module in the netlist to interpret.");
ABSL_FLAG(std::string, netlist, "", "Path to the netlist to interpret.");

namespace xls {

xabsl::StatusOr<netlist::CellLibrary> GetCellLibrary(
    const std::string& cell_library_path,
    const std::string& cell_library_proto_path) {
  if (!cell_library_proto_path.empty()) {
    XLS_ASSIGN_OR_RETURN(std::string proto_text,
                         GetFileContents(cell_library_proto_path));
    netlist::CellLibraryProto lib_proto;
    XLS_RET_CHECK(lib_proto.ParseFromString(proto_text));
    return netlist::CellLibrary::FromProto(lib_proto);
  } else {
    XLS_ASSIGN_OR_RETURN(std::string cell_library_text,
                         GetFileContents(cell_library_path));
    XLS_ASSIGN_OR_RETURN(
        auto char_stream,
        netlist::cell_lib::CharStream::FromText(cell_library_text));
    XLS_ASSIGN_OR_RETURN(netlist::CellLibraryProto lib_proto,
                         netlist::function::ExtractFunctions(&char_stream));
    return netlist::CellLibrary::FromProto(lib_proto);
  }
}

absl::Status RealMain(const std::string& netlist_path,
                      const std::string& cell_library_path,
                      const std::string& cell_library_proto_path,
                      const absl::flat_hash_set<std::string>& high_cells,
                      const std::string& module_name,
                      absl::Span<const std::string> inputs,
                      const std::string& output_type_string,
                      absl::Span<const std::string> dump_cells) {
  XLS_ASSIGN_OR_RETURN(
      netlist::CellLibrary cell_library,
      GetCellLibrary(cell_library_path, cell_library_proto_path));

  XLS_ASSIGN_OR_RETURN(std::string netlist_text, GetFileContents(netlist_path));
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSIGN_OR_RETURN(auto netlist, netlist::rtl::Parser::ParseNetlist(
                                         &cell_library, &scanner));
  XLS_ASSIGN_OR_RETURN(const auto* module, netlist->GetModule(module_name));

  // Input values are listed in the same order as inputs are declared by
  // Module::inputs().
  Bits input_bits;
  for (const auto& input_string : inputs) {
    XLS_ASSIGN_OR_RETURN(Value input, Parser::ParseTypedValue(input_string));
    Bits flat_value = FlattenValueToBits(input);
    input_bits = bits_ops::Concat({input_bits, flat_value});
  }
  input_bits = bits_ops::Reverse(input_bits);

  absl::flat_hash_map<const netlist::rtl::NetRef, bool> input_nets;
  const std::vector<netlist::rtl::NetRef>& module_inputs = module->inputs();
  XLS_RET_CHECK(module_inputs.size() == input_bits.bit_count());

  for (int i = 0; i < module->inputs().size(); i++) {
    input_nets[module_inputs[i]] = input_bits.Get(i);
  }

  netlist::Interpreter interpreter(netlist.get(), high_cells);
  XLS_ASSIGN_OR_RETURN(auto output_nets, interpreter.InterpretModule(
                                             module, input_nets, dump_cells));

  BitsRope rope(output_nets.size());
  for (const netlist::rtl::NetRef ref : module->outputs()) {
    rope.push_back(output_nets[ref]);
  }
  Bits output_bits = bits_ops::Reverse(rope.Build());

  Value output;
  if (!output_type_string.empty()) {
    // This is a disposable package - it only exists to hold the type below.
    Package package("foo");
    XLS_ASSIGN_OR_RETURN(Type * output_type,
                         Parser::ParseType(output_type_string, &package));
    XLS_ASSIGN_OR_RETURN(output,
                         UnflattenBitsToValue(output_bits, output_type));
  } else {
    output = Value(output_bits);
  }

  std::cout << "Results: " << output.ToString(FormatPreference::kHex)
            << std::endl;
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string cell_library_path = absl::GetFlag(FLAGS_cell_library);
  std::string cell_library_proto_path = absl::GetFlag(FLAGS_cell_library_proto);
  XLS_QCHECK(!cell_library_path.empty() ^ !cell_library_proto_path.empty())
      << "One (and only one) of --cell_library or --cell_library_proto "
         "must be specified.";

  std::string netlist_path = absl::GetFlag(FLAGS_netlist);
  XLS_QCHECK(!netlist_path.empty()) << "--netlist must be specified.";

  std::string module_name = absl::GetFlag(FLAGS_module_name);
  XLS_QCHECK(!module_name.empty()) << "--module_name must be specified.";

  std::string high_cells_string = absl::GetFlag(FLAGS_high_cells);
  absl::flat_hash_set<std::string> high_cells;
  for (const auto& high_cell : absl::StrSplit(high_cells_string, ',')) {
    high_cells.insert(std::string(high_cell));
  }

  std::string input = absl::GetFlag(FLAGS_input);
  XLS_QCHECK(!input.empty()) << "--input must be specified.";
  std::vector<std::string> inputs = absl::StrSplit(input, ';');

  std::string dump_cells_str = absl::GetFlag(FLAGS_dump_cells);
  std::vector<std::string> dump_cells = absl::StrSplit(dump_cells_str, ',');

  std::string output_type = absl::GetFlag(FLAGS_output_type);

  XLS_QCHECK_OK(xls::RealMain(netlist_path, cell_library_path,
                              cell_library_proto_path, high_cells, module_name,
                              inputs, output_type, dump_cells));

  return 0;
}
