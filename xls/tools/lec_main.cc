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
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/ir_parser.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist.pb.h"
#include "xls/netlist/netlist_parser.h"
#include "xls/netlist/z3_translator.h"
#include "xls/solvers/z3_ir_translator.h"
#include "../z3/src/api/z3_api.h"

ABSL_FLAG(std::string, cell_lib_path, "",
          "Path to the cell library. "
          "Either this or cell_proto_path should be set.");
ABSL_FLAG(std::string, cell_proto_path, "",
          "Path to the preprocessed cell library proto. "
          "This is a whole bunch faster than specifiying an unprocessed "
          "cell library and should be favored.\n"
          "Either this or --cell_lib_path should be set.");
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

namespace xls {
namespace {

// Convenience struct for IR translation data.
struct IrData {
  // The package holding all IR data.
  std::unique_ptr<Package> package;

  // The function translated by the below translator.
  Function* function;

  // The translator for a given function in the package.
  std::unique_ptr<solvers::z3::IrTranslator> translator;
};

// Reads in XLS IR and returns a Z3Translator for the desired function.
xabsl::StatusOr<IrData> GetIrTranslator(absl::string_view ir_path,
                                        absl::string_view entry_function_name) {
  IrData ir_data;
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(ir_data.package, Parser::ParsePackage(ir_text));
  if (entry_function_name.empty()) {
    XLS_ASSIGN_OR_RETURN(ir_data.function, ir_data.package->EntryFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(ir_data.function,
                         ir_data.package->GetFunction(entry_function_name));
  }

  XLS_ASSIGN_OR_RETURN(
      ir_data.translator,
      solvers::z3::IrTranslator::CreateAndTranslate(ir_data.function));
  return ir_data;
}

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
xabsl::StatusOr<netlist::rtl::Netlist> GetNetlist(
    absl::string_view netlist_path, netlist::CellLibrary* cell_library) {
  XLS_ASSIGN_OR_RETURN(std::string netlist_text, GetFileContents(netlist_path));
  netlist::rtl::Scanner scanner(netlist_text);
  return netlist::rtl::Parser::ParseNetlist(cell_library, &scanner);
}

}  // namespace

absl::Status RealMain(absl::string_view ir_path,
                      absl::string_view entry_function_name,
                      absl::string_view netlist_module_name,
                      absl::string_view cell_lib_path,
                      absl::string_view cell_proto_path,
                      const absl::flat_hash_set<std::string>& high_cells,
                      absl::string_view netlist_path) {
  XLS_ASSIGN_OR_RETURN(IrData ir_data,
                       GetIrTranslator(ir_path, entry_function_name));
  if (entry_function_name.empty()) {
    entry_function_name = ir_data.function->name();
  }
  if (netlist_module_name.empty()) {
    netlist_module_name = entry_function_name;
  }

  Z3_context ctx = ir_data.translator->ctx();
  Function* entry_function = ir_data.function;

  // Get the inputs to the IR function, and flatten them into values as
  // expected by the netlist function (to "tie together" the inputs to the two
  // translations.
  absl::flat_hash_map<std::string, Z3_ast> inputs;
  for (const Param* param : entry_function->params()) {
    // Explode each param into individual bits.
    std::vector<Z3_ast> bits = ir_data.translator->FlattenValue(
        param->GetType(), ir_data.translator->GetTranslation(param));
    if (bits.size() > 1) {
      for (int i = 0; i < bits.size(); i++) {
        // Param names are formatted by the parser as
        // <param_name>[<bit_index>] or <param_name> (for single-bit)
        std::string name = absl::StrCat(param->name(), "[", i, "]");
        inputs[name] = bits[i];
      }
    } else {
      inputs[param->name()] = bits[0];
    }
  }

  XLS_ASSIGN_OR_RETURN(netlist::CellLibrary cell_library,
                       GetCellLibrary(cell_lib_path, cell_proto_path));
  XLS_ASSIGN_OR_RETURN(auto netlist, GetNetlist(netlist_path, &cell_library));
  XLS_ASSIGN_OR_RETURN(const netlist::rtl::Module* netlist_module,
                       netlist.GetModule(std::string{netlist_module_name}));

  absl::flat_hash_map<std::string, const netlist::rtl::Module*> module_refs;
  for (const std::unique_ptr<netlist::rtl::Module>& module :
       netlist.modules()) {
    if (module->name() == netlist_module_name) {
      netlist_module = module.get();
    } else {
      module_refs[module->name()] = module.get();
    }
  }

  XLS_ASSIGN_OR_RETURN(
      auto netlist_translator,
      netlist::Z3Translator::CreateAndTranslate(
          ctx, netlist_module, module_refs, inputs, high_cells));

  // Now do the opposite of the param flattening above - collect the outputs
  // from the netlist translation and unflatten them into [the higher-level]
  // IR types.
  std::vector<Z3_ast> z3_outputs;
  z3_outputs.reserve(netlist_module->outputs().size());
  for (const auto& output : netlist_module->outputs()) {
    // Drop output wires not part of the original signature.
    if (output->name() == "output_valid") {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Z3_ast z3_output,
                         netlist_translator->GetTranslation(output));
    z3_outputs.push_back(z3_output);
  }
  std::reverse(z3_outputs.begin(), z3_outputs.end());

  Z3_ast netlist_output = ir_data.translator->UnflattenZ3Ast(
      entry_function->GetType()->return_type(), absl::MakeSpan(z3_outputs));

  // Create the final equality checks. Since we're trying to prove the opposite,
  // we're aiming for NOT(EQ(ir, netlist))
  Z3_ast eq_node =
      Z3_mk_eq(ctx, ir_data.translator->GetReturnNode(), netlist_output);
  eq_node = Z3_mk_not(ctx, eq_node);

  // TODO(rspringer): This code below _really_needs_ to be commonized. It's used
  // ~verbatim in quite a few places by now.
  // Push all that work into z3, and have the solver do its work.
  Z3_params params = Z3_mk_params(ctx);
  Z3_params_inc_ref(ctx, params);
  Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "sat.threads"),
                     absl::base_internal::NumCPUs());
  Z3_params_set_uint(ctx, params, Z3_mk_string_symbol(ctx, "threads"),
                     absl::base_internal::NumCPUs());

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_set_params(ctx, solver, params);
  Z3_solver_assert(ctx, solver, eq_node);

  std::cout << solvers::z3::SolverResultToString(ctx, solver) << std::endl;

  Z3_solver_dec_ref(ctx, solver);
  Z3_params_dec_ref(ctx, params);

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

  XLS_QCHECK_OK(xls::RealMain(ir_path, absl::GetFlag(FLAGS_entry_function_name),
                              absl::GetFlag(FLAGS_netlist_module_name),
                              cell_lib_path, cell_proto_path, high_cells,
                              netlist_path));
  return 0;
}
