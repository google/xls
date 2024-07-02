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

#ifndef XLS_SOLVERS_Z3_LEC_H_
#define XLS_SOLVERS_Z3_LEC_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/netlist/netlist.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_netlist_translator.h"
#include "external/z3/src/api/z3.h"  // IWYU pragma: keep
#include "external/z3/src/api/z3_api.h"

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
};

// Class for performing logical equivalence checks between a function specified
// in XLS IR (perhaps converted from DSLX) and a netlist.
class Lec {
 public:
  // Creates a LEC object for checking across the entire specified function and
  // module.
  static absl::StatusOr<std::unique_ptr<Lec>> Create(const LecParams& params);

  // Creates a LEC object for a particular pipeline stage. The schedule only
  // directly applies to the specified XLS Function; the mapping of Netlist
  // cell/wire to stage is derived from there.
  static absl::StatusOr<std::unique_ptr<Lec>> CreateForStage(
      const LecParams& params, const PipelineSchedule& schedule, int stage);
  ~Lec();

  // Applies additional constraints (aside from the LEC itself), such as
  // restricting the input space.
  // This function must have the same signature as the function being compared
  // and must be called either "main" or have the same name as its containing
  // package or else be the only function in the file.
  // The function must return 1 for cases where all constraints are
  // satisfied, and 0 otherwise.
  // Constraints can not be currently specified with per-stage evaluation.
  absl::Status AddConstraints(Function* constraints);

  // Returns true of the netlist and IR are proved to be equivalent.
  bool Run();

  // Dumps all Z3 values corresponding to IR nodes in the input function.
  void DumpIrTree();

  // Returns a textual description of the result.
  std::string ResultToString();

  Z3_context ctx() { return ir_translator_->ctx(); }

 private:
  Lec(Function* ir_function, netlist::rtl::Netlist* netlist,
      const std::string& netlist_module_name,
      std::optional<PipelineSchedule> schedule, int stage);
  absl::Status Init();
  absl::Status CreateIrTranslator();
  absl::Status CreateNetlistTranslator();

  // Collects the XLS IR nodes that are inputs to this evaluation - either the
  // original function inputs for whole-function or first-stage equivalence
  // checks, or the stage inputs for all others.
  absl::Status CollectIrInputs();

  // And the above, but for outputs - either the function outputs for
  // whole-function or last-stage checks, or stage outputs for all others.
  void CollectIrOutputNodes();

  // Connects IR Params to netlist inputs.
  absl::Status BindNetlistInputs();

  // Explodes each param into individual bits. XLS IR and parsed netlist data
  // layouts are different in that:
  //  1) Netlists list input values from high-to-low bit, i.e.,
  //  input_value_31_, input_value_30_, ... input_value_0, for a 32-bit input.
  //  2) Netlists interpret their input values as "little-endian", i.e.,
  //  input_value_7_ for an 8-bit input will be the MSB and
  //  input_value_0_ will be the LSB.
  absl::flat_hash_map<std::string, Z3_ast> FlattenNetlistInputs();

  // The opposite of BindNetlistInputs - given the output nodes from the
  // stage, collect the corresponding NetRefs and use them to reconstruct
  // the composite output value.
  absl::StatusOr<std::vector<Z3_ast>> GetNetlistZ3ForIr(const Node* node);

  // Retrieves the set of NetRefs corresponding to the output value of the given
  // node.
  absl::StatusOr<std::vector<netlist::rtl::NetRef>> GetIrNetrefs(
      const Node* node);

  // Returns the name of the netlist wire corresponding to the input node.
  std::string NodeToNetlistName(const Node* node, std::optional<int> bit_index,
                                bool is_cell = true);

  // Gets strings comparing the IR to netlist values of the given node under the
  // current model.
  std::pair<std::string, std::string> GetComparisonStrings(const Node* node);

  // Replaces any values in the given string (nl_string) with "don't care"
  // markers if the corresponding AST nodes aren't present, i.e., if the source
  // wires aren't present in the netlist.
  void MarkDontCareBits(const std::vector<Z3_ast>& nl_bits,
                        std::string& nl_string);

  Function* ir_function_;
  std::unique_ptr<IrTranslator> ir_translator_;

  netlist::rtl::Netlist* netlist_;
  std::string netlist_module_name_;
  const netlist::rtl::Module* module_;
  std::unique_ptr<NetlistTranslator> netlist_translator_;

  // Cached copies of translation data (cached for post-proof output).
  absl::flat_hash_map<const Node*, Z3_ast> input_mapping_;
  std::vector<const Node*> ir_output_nodes_;
  std::vector<Z3_ast> ir_outputs_;
  std::vector<Z3_ast> netlist_outputs_;

  std::optional<PipelineSchedule> schedule_;
  int stage_;

  // Z3 elements are, under the hood, void pointers, but let's respect the
  // interface and use std::optional to determine live-ness.
  std::optional<Z3_solver> solver_;

  // Satisfiable is equivalent to "model_.has_value()", but having an explicit
  // value is more understandable.
  bool satisfiable_;
  std::optional<Z3_model> model_;
};

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_SOLVERS_Z3_LEC_H_
