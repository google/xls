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

#ifndef XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_
#define XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/netlist/function_parser.h"
#include "xls/netlist/netlist.h"
#include "../z3/src/api/z3.h"

namespace xls {
namespace solvers {
namespace z3 {

// Z3Translator converts a netlist into a Z3 AST suitable for use in Z3 proofs
// (correctness, equality, etc.).
// It does this by converting the logical ops described in a Cell's "function"
// attribute into trees Z3 operations, then combining those trees, as described
// in the Module's nets, into one comprehensive tree.
class NetlistTranslator {
 public:
  // Inputs must be provided here so that the same "values" can be used for
  // other trees, e.g., in an equivalence check (we need the same input nodes to
  // feed into two separate trees in that case).
  //  - module_refs is a collection of modules that may be present as Cell
  //    references in the module being processed.
  //  - inputs is a map of wire/net name to Z3 one-bit vectors; this requires
  //    "exploding" values, such as a bits[8] into 8 single-bit inputs.
  static absl::StatusOr<std::unique_ptr<NetlistTranslator>> CreateAndTranslate(
      Z3_context ctx, const netlist::rtl::Module* module,
      const absl::flat_hash_map<std::string, const netlist::rtl::Module*>&
          module_refs);

  // Returns the Z3 equivalent for the specified net.
  absl::StatusOr<Z3_ast> GetTranslation(netlist::rtl::NetRef ref);

  // Retranslates the netlist, replacing the named netrefs with their paired Z3
  // ASTs.
  absl::Status Retranslate(
      const absl::flat_hash_map<std::string, Z3_ast>& inputs);

  // An individual node in a "value cone": the transitive set of nodes on which
  // one particular node depends to determine its value.
  struct ValueCone {
    Z3_ast node;
    netlist::rtl::NetRef ref;
    const netlist::rtl::Cell* parent_cell;
    std::vector<ValueCone> parents;
  };

  // Calculates the value cone for the given NetRef. "terminals" is the set of
  // nodes at which to stop processing, e.g., a set of fixed inputs or other
  // values past which the cone is unnecessary.
  ValueCone GetValueCone(netlist::rtl::NetRef ref,
                         const absl::flat_hash_set<Z3_ast>& terminals);

  // Prints the given value cone, when evaluated with the given model, to the
  // terminal. Assumes (generally safely) that the associated Z3_context is that
  // held by this object.
  void PrintValueCone(const ValueCone& value_cone, Z3_model model,
                      int level = 0);

 private:
  NetlistTranslator(
      Z3_context ctx, const netlist::rtl::Module* module,
      const absl::flat_hash_map<std::string, const netlist::rtl::Module*>&
          module_refs);
  absl::Status Init();

  // Translates the module, cell, or cell function, respectively, into Z3-space.
  absl::Status Translate();
  absl::Status TranslateCell(const netlist::rtl::Cell& cell);
  absl::StatusOr<Z3_ast> TranslateFunction(
      const netlist::rtl::Cell& cell, netlist::function::Ast ast,
      const absl::flat_hash_map<std::string, Z3_ast>& state_table_values);
  absl::StatusOr<absl::flat_hash_map<std::string, Z3_ast>> TranslateStateTable(
      const netlist::rtl::Cell& cell);

  Z3_context ctx_;
  const netlist::rtl::Module* module_;

  // Maps a NetDef to a Z3 entity.
  absl::flat_hash_map<netlist::rtl::NetRef, Z3_ast> translated_;

  const absl::flat_hash_map<std::string, const netlist::rtl::Module*>
      module_refs_;
};

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_
