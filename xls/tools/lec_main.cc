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
#include "xls/netlist/z3_translator.h"
#include "xls/solvers/z3_ir_translator.h"
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

namespace xls {
namespace {

using solvers::z3::IrTranslator;

constexpr const char kIrConverterPath[] = "xls/dslx/ir_converter_main";

// Convenience struct for IR translation data.
struct IrData {
  // The package holding all IR data.
  std::unique_ptr<Package> package;

  // The function translated by the below translator.
  Function* function;

  // The translator for a given function in the package.
  std::unique_ptr<IrTranslator> translator;
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

  XLS_ASSIGN_OR_RETURN(ir_data.translator,
                       IrTranslator::CreateAndTranslate(ir_data.function));
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

// Applies the constraints in the specified constraint file to the solver, using
// the parameters in the main translator.
absl::Status ApplyConstraints(Z3_solver solver, IrTranslator* main_translator,
                              absl::string_view constraints_file) {
  if (constraints_file.empty()) {
    return absl::OkStatus();
  }

  // Convert the input DSLX to XLS IR.
  std::filesystem::path ir_converter_path = GetXlsRunfilePath(kIrConverterPath);
  std::vector<std::string> args;
  args.push_back(ir_converter_path);
  args.push_back(std::string(constraints_file));
  XLS_ASSIGN_OR_RETURN(auto stdout_and_stderr, InvokeSubprocess(args));

  XLS_ASSIGN_OR_RETURN(auto package,
                       Parser::ParsePackage(stdout_and_stderr.first));
  XLS_ASSIGN_OR_RETURN(Function * function, package->EntryFunction());
  std::vector<Z3_ast> params;
  for (const auto& param : main_translator->xls_function()->params()) {
    params.push_back(main_translator->GetTranslation(param));
  }

  // Assert that the constraint function's outputs are 1, and slap that into the
  // solver.
  Z3_context ctx = main_translator->ctx();
  XLS_ASSIGN_OR_RETURN(auto constraint_translator,
                       IrTranslator::CreateAndTranslate(ctx, function, params));
  Z3_ast eq_node = Z3_mk_eq(ctx, constraint_translator->GetReturnNode(),
                            Z3_mk_int(ctx, 1, Z3_mk_bv_sort(ctx, 1)));
  Z3_solver_assert(ctx, solver, eq_node);

  return absl::OkStatus();
}

}  // namespace

absl::Status RealMain(absl::string_view ir_path,
                      absl::string_view entry_function_name,
                      absl::string_view netlist_module_name,
                      absl::string_view cell_lib_path,
                      absl::string_view cell_proto_path,
                      const absl::flat_hash_set<std::string>& high_cells,
                      absl::string_view netlist_path,
                      absl::string_view constraints_file) {
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
    // Explode each param into individual bits. XLS IR and parsed netlist data
    // layouts are different in that:
    //  1) Netlists list input values from high-to-low bit, i.e.,
    //  input_value_31_, input_value_30_, ... input_value_0, for a 32-bit input.
    //  2) Netlists interpret their input values as "little-endian", i.e.,
    //  input_value_7_ for an 8-bit input will be the MSB and
    //  input_value_0_ will be the LSB.
    // These two factors are why we need to reverse elements here. Item 1 is why
    // we reverse the entire bits vector, and item 2 is why we pass
    // little_endian as true to FlattenValue.
    std::vector<Z3_ast> bits = ir_data.translator->FlattenValue(
        param->GetType(), ir_data.translator->GetTranslation(param),
        /*little_endian=*/true);
    std::reverse(bits.begin(), bits.end());
    if (bits.size() > 1) {
      for (int i = 0; i < bits.size(); i++) {
        // Param names are formatted by the parser as
        // <param_name>[<bit_index>] or <param_name> (for single-bit)
        std::string name = absl::StrCat(param->name(), "_", i, "_");
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

  // Specify little endian here as with FlattenValue() above.
  Z3_ast netlist_output = ir_data.translator->UnflattenZ3Ast(
      entry_function->GetType()->return_type(), absl::MakeSpan(z3_outputs),
      /*little_endian=*/true);

  // Create the final equality checks. Since we're trying to prove the opposite,
  // we're aiming for NOT(EQ(ir, netlist))
  Z3_ast eq_node =
      Z3_mk_eq(ctx, ir_data.translator->GetReturnNode(), netlist_output);
  eq_node = Z3_mk_not(ctx, eq_node);

  // Push all that work into z3, and have the solver do its work.
  Z3_solver solver =
      solvers::z3::CreateSolver(ctx, absl::base_internal::NumCPUs());
  Z3_solver_assert(ctx, solver, eq_node);
  XLS_RETURN_IF_ERROR(
      ApplyConstraints(solver, ir_data.translator.get(), constraints_file));

  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  std::cout << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
            << std::endl;

  // If the condition was satisifiable, display sample inputs.
  if (satisfiable == Z3_L_TRUE) {
    Z3_model model = Z3_solver_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);

    std::cout << "IR result: "
              << solvers::z3::QueryNode(ctx, model,
                                        ir_data.translator->GetReturnNode())
              << std::endl;
    std::cout << "Netlist result: "
              << solvers::z3::QueryNode(ctx, model, netlist_output)
              << std::endl;

    if (XLS_VLOG_IS_ON(2)) {
      DumpTree(ctx, model, ir_data.translator.get());
    }

    Z3_model_dec_ref(ctx, model);
  }

  Z3_solver_dec_ref(ctx, solver);

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

  XLS_QCHECK_OK(xls::RealMain(
      ir_path, absl::GetFlag(FLAGS_entry_function_name),
      absl::GetFlag(FLAGS_netlist_module_name), cell_lib_path, cell_proto_path,
      high_cells, netlist_path, absl::GetFlag(FLAGS_constraints_file)));
  return 0;
}
