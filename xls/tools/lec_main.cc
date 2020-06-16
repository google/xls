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

// Tool to prove or disprove logical equivalence of XLS IR and a netlist.

#include "absl/base/internal/sysinfo.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/common/subprocess.h"
#include "xls/ir/ir_parser.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist.pb.h"
#include "xls/netlist/netlist_parser.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/solvers/z3_lec.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

ABSL_FLAG(std::string, cell_lib_path, "",
          "Path to the cell library. "
          "Either this or cell_proto_path should be set.");
ABSL_FLAG(std::string, cell_proto_path, "",
          "Path to the preprocessed cell library proto. "
          "This is a whole bunch faster than specifiying an unprocessed "
          "cell library and should be favored.\n"
          "Either this or --cell_lib_path should be set.");
ABSL_FLAG(std::string, constraints_file, "",
          "Optional path to a DSLX file containing a input parameter "
          "constraint function. This function must have the same signature as "
          "the function being compared and must be called either \"main\" or "
          "have the same name as its containing package or else "
          "be the only function in the file. The "
          "function must return 1 for cases where all constraints are "
          "satisfied, and 0 otherwise.\n"
          "For example, a constraint function to restrict arg 0 to be == 5 "
          "and arg 1, tuple element 0 to be > 6 would be as follows:\n"
          "  fn constraints(a: bits[3], b: (bits[8], bits[16])) -> bits[1] {\n"
          "    let x0 = a == 5;\n"
          "    let x1 = b[0] > 6;\n"
          "    x0 & x1;\n"
          "  }");
ABSL_FLAG(std::string, entry_function_name, "",
          "Function (in the IR) to compare. If unset, the program will attempt "
          "to find and use an appropriate entry function.");
// TODO(rspringer): Eliminate the need for this flag.
// This one is a hidden temporary flag until we can properly handle cells
// with state_function attributes (e.g., some latches).
ABSL_FLAG(std::string, high_cells, "",
          "Comma-separated list of cells for which to assume \"1\" values on "
          "all outputs.");
ABSL_FLAG(std::string, netlist_module_name, "",
          "Module name (in the netlist) to compare. If unset, the program will "
          "use the name of the entry function in the IR.");
ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR to compare against.");
ABSL_FLAG(std::string, netlist_path, "", "Path to the netlist.");
ABSL_FLAG(std::string, schedule_path, "",
          "Path to a PipelineSchedule textproto containing the schedule.\n"
          "!IMPORTANT! If the netlist spans multiple stages, a schedule MUST "
          "be specified. Otherwise, mapping IR nodes to netlist cells is "
          "impossible.");
ABSL_FLAG(int32, stage, -1,
          "Pipeline stage to evaluate. Requires --schedule.\n"
          "If \"schedule\" is set, but this is not, then the entire module "
          "will be evaluated.");

namespace xls {
namespace {

using solvers::z3::IrTranslator;

constexpr const char kIrConverterPath[] = "xls/dslx/ir_converter_main";

// Loads a cell library, either from a raw Liberty file or a preprocessed
// CellLibraryProto proto.
xabsl::StatusOr<netlist::CellLibrary> GetCellLibrary(
    absl::string_view cell_lib_path, absl::string_view cell_proto_path) {
  if (!cell_proto_path.empty()) {
    XLS_ASSIGN_OR_RETURN(std::string cell_proto_text,
                         GetFileContents(cell_proto_path));
    netlist::CellLibraryProto cell_proto;
    XLS_RET_CHECK(cell_proto.ParseFromString(cell_proto_text));
    return netlist::CellLibrary::FromProto(cell_proto);
  } else {
    XLS_ASSIGN_OR_RETURN(std::string lib_text, GetFileContents(cell_lib_path));
    XLS_ASSIGN_OR_RETURN(auto stream,
                         netlist::cell_lib::CharStream::FromText(lib_text));
    XLS_ASSIGN_OR_RETURN(netlist::CellLibraryProto proto,
                         netlist::function::ExtractFunctions(&stream));
    return netlist::CellLibrary::FromProto(proto);
  }
}

// Loads and parses a netlist from a file.
xabsl::StatusOr<std::unique_ptr<netlist::rtl::Netlist>> GetNetlist(
    absl::string_view netlist_path, netlist::CellLibrary* cell_library) {
  XLS_ASSIGN_OR_RETURN(std::string netlist_text, GetFileContents(netlist_path));
  netlist::rtl::Scanner scanner(netlist_text);
  return netlist::rtl::Parser::ParseNetlist(cell_library, &scanner);
}

// Dumps all Z3 values corresponding to IR nodes in the input function.
void DumpTree(Z3_context ctx, Z3_model model, IrTranslator* translator) {
  std::deque<const Node*> to_process;
  to_process.push_back(translator->xls_function()->return_value());

  absl::flat_hash_set<const Node*> seen;
  while (!to_process.empty()) {
    const Node* node = to_process.front();
    to_process.pop_front();
    Z3_ast translation = translator->GetTranslation(node);
    std::cout << "IR: " << node->ToString() << std::endl;
    std::cout << "Z3: " << solvers::z3::QueryNode(ctx, model, translation)
              << std::endl
              << std::endl;
    seen.insert(node);
    for (const Node* operand : node->operands()) {
      if (!seen.contains(operand)) {
        to_process.push_back(operand);
      }
    }
  }
}

}  // namespace

absl::Status RealMain(absl::string_view ir_path,
                      absl::string_view entry_function_name,
                      absl::string_view netlist_module_name,
                      absl::string_view cell_lib_path,
                      absl::string_view cell_proto_path,
                      const absl::flat_hash_set<std::string>& high_cells,
                      absl::string_view netlist_path,
                      absl::string_view constraints_file,
                      absl::string_view schedule_path, int stage) {
  solvers::z3::LecParams lec_params;
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  lec_params.ir_package = package.get();
  if (entry_function_name.empty()) {
    XLS_ASSIGN_OR_RETURN(lec_params.ir_function,
                         lec_params.ir_package->EntryFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(
        lec_params.ir_function,
        lec_params.ir_package->GetFunction(entry_function_name));
  }
  XLS_ASSIGN_OR_RETURN(auto cell_library,
                       GetCellLibrary(cell_lib_path, cell_proto_path));
  XLS_ASSIGN_OR_RETURN(auto netlist, GetNetlist(netlist_path, &cell_library));
  lec_params.netlist = netlist.get();
  lec_params.netlist_module_name = netlist_module_name;
  lec_params.high_cells = high_cells;

  std::unique_ptr<solvers::z3::Lec> lec;
  if (!schedule_path.empty()) {
    XLS_ASSIGN_OR_RETURN(
        PipelineScheduleProto proto,
        ParseTextProtoFile<PipelineScheduleProto>(schedule_path));
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::FromProto(lec_params.ir_function, proto));
    XLS_ASSIGN_OR_RETURN(lec, solvers::z3::Lec::CreateForStage(
                                  std::move(lec_params), schedule, stage));
  } else {
    XLS_ASSIGN_OR_RETURN(lec, solvers::z3::Lec::Create(std::move(lec_params)));
  }

  std::unique_ptr<Package> constraints_pkg;
  if (!constraints_file.empty()) {
    std::filesystem::path ir_converter_path =
        GetXlsRunfilePath(kIrConverterPath);
    std::vector<std::string> args;
    args.push_back(ir_converter_path);
    args.push_back(std::string(constraints_file));
    XLS_ASSIGN_OR_RETURN(auto stdout_and_stderr, InvokeSubprocess(args));

    XLS_ASSIGN_OR_RETURN(constraints_pkg,
                         Parser::ParsePackage(stdout_and_stderr.first));
    XLS_ASSIGN_OR_RETURN(Function * function, constraints_pkg->EntryFunction());
    XLS_RETURN_IF_ERROR(lec->AddConstraints(function));
  }

  lec->Run();
  XLS_ASSIGN_OR_RETURN(std::string output, lec->ResultToString());
  std::cout << output << std::endl;

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  XLS_QCHECK(!ir_path.empty()) << "--ir_path must be set.";

  std::string netlist_path = absl::GetFlag(FLAGS_netlist_path);
  XLS_QCHECK(!netlist_path.empty()) << "--netlist_path must be set.";

  std::string cell_lib_path = absl::GetFlag(FLAGS_cell_lib_path);
  std::string cell_proto_path = absl::GetFlag(FLAGS_cell_proto_path);
  XLS_QCHECK(cell_lib_path.empty() ^ cell_proto_path.empty())
      << "One (and only one) of --cell_lib_path and --cell_proto_path "
         "should be set.";

  std::string high_cells_string = absl::GetFlag(FLAGS_high_cells);
  absl::flat_hash_set<std::string> high_cells;
  for (const auto& high_cell : absl::StrSplit(high_cells_string, ',')) {
    high_cells.insert(std::string(high_cell));
  }

  std::string schedule_path = absl::GetFlag(FLAGS_schedule_path);
  int stage = absl::GetFlag(FLAGS_stage);
  XLS_QCHECK(stage == -1 || !schedule_path.empty())
      << "--schedule_path must be specified with --stage.";

  XLS_QCHECK_OK(xls::RealMain(
      ir_path, absl::GetFlag(FLAGS_entry_function_name),
      absl::GetFlag(FLAGS_netlist_module_name), cell_lib_path, cell_proto_path,
      high_cells, netlist_path, absl::GetFlag(FLAGS_constraints_file),
      schedule_path, stage));
  return 0;
}
