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

#ifndef THIRD_PARTY_XLS_SOLVERS_Z3_LEC_H_
#define THIRD_PARTY_XLS_SOLVERS_Z3_LEC_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "xls/ir/package.h"
#include "xls/netlist/netlist.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_netlist_translator.h"

namespace xls {
namespace solvers {
namespace z3 {

// Simple utility struct to keep Lec::Create()'s arg list from being unwieldy.
struct LecParams {
  // The XLS IR package containing the function to compare.
  // Note that this is a borrowed reference that must outlive the Lec object.
  Package* ir_package;

  // The comparison function itself.
  Function* ir_function;

  // The netlist to compare.
  // Note that this is a borrowed reference that must outlive the Lec object.
  netlist::rtl::Netlist* netlist;

  // The name of the module (inside "netlist") to compare.
  std::string netlist_module_name;

  // Set of [netlist] cells to assume are always high.
  absl::flat_hash_set<std::string> high_cells;
};

// Class for performing logical equivalence checks between a function specified
// in XLS IR (perhaps converted from DSLX) and a netlist.
class Lec {
 public:
  static xabsl::StatusOr<std::unique_ptr<Lec>> Create(LecParams params);
  ~Lec();

  // Applies additional constraints (aside from the LEC itself), such as
  // restricting the input space.
  // This function must have the same signature as the function being compared
  // and must be called either "main" or have the same name as its containing
  // package or else be the only function in the file.
  // The function must return 1 for cases where all constraints are
  // satisfied, and 0 otherwise.
  absl::Status AddConstraints(Function* constraints);

  // Returns true of the netlist and IR are proved to be equivalent.
  bool Run();

  // Dumps all Z3 values corresponding to IR nodes in the input function.
  void DumpIrTree();

  // Returns a textual description of the result.
  xabsl::StatusOr<std::string> ResultToString();

  Z3_context ctx() { return ir_translator_->ctx(); }

 private:
  Lec(Package* ir_package, Function* ir_function,
      netlist::rtl::Netlist* netlist, const std::string& netlist_module_name);
  absl::Status Init(const absl::flat_hash_set<std::string>& high_cells);
  absl::Status CreateIrTranslator();
  absl::Status CreateNetlistTranslator(
      const absl::flat_hash_set<std::string>& high_cells);

  // Explodes each param into individual bits. XLS IR and parsed netlist data
  // layouts are different in that:
  //  1) Netlists list input values from high-to-low bit, i.e.,
  //  input_value_31_, input_value_30_, ... input_value_0, for a 32-bit input.
  //  2) Netlists interpret their input values as "little-endian", i.e.,
  //  input_value_7_ for an 8-bit input will be the MSB and
  //  input_value_0_ will be the LSB.
  absl::flat_hash_map<std::string, Z3_ast> FlattenNetlistInputs();

  // The opposite of FlattenNetlistInputs - taken the output pins from the
  // module, reconstruct the composite value as seen by the IR translator.
  xabsl::StatusOr<Z3_ast> UnflattenNetlistOutputs();

  Package* ir_package_;
  Function* ir_function_;
  std::unique_ptr<IrTranslator> ir_translator_;

  netlist::rtl::Netlist* netlist_;
  std::string netlist_module_name_;
  std::unique_ptr<NetlistTranslator> netlist_translator_;

  // Z3 elements are, under the hood, void pointers, but let's respect the
  // interface and use absl::optional to determine live-ness.
  absl::optional<Z3_solver> solver_;

  // Satisfiable is equivalent to "model_.has_value()", but having an explicit
  // value is more understandable.
  bool satisfiable_;
  absl::optional<Z3_model> model_;
};

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // THIRD_PARTY_XLS_SOLVERS_Z3_LEC_H_
