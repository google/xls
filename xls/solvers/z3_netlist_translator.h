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

#ifndef THIRD_PARTY_XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_
#define THIRD_PARTY_XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "xls/common/status/statusor.h"
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
  static xabsl::StatusOr<std::unique_ptr<NetlistTranslator>> CreateAndTranslate(
      Z3_context ctx, const netlist::rtl::Module* module,
      const absl::flat_hash_map<std::string, const netlist::rtl::Module*>&
          module_refs,
      const absl::flat_hash_map<std::string, Z3_ast>& inputs,
      const absl::flat_hash_set<std::string>& high_cells);

  // Returns the Z3 equivalent for the specified net.
  xabsl::StatusOr<Z3_ast> GetTranslation(netlist::rtl::NetRef ref);

 private:
  NetlistTranslator(
      Z3_context ctx, const netlist::rtl::Module* module,
      const absl::flat_hash_map<std::string, const netlist::rtl::Module*>&
          module_refs,
      const absl::flat_hash_set<std::string>& high_cells);
  absl::Status Init(const absl::flat_hash_map<std::string, Z3_ast>& inputs);

  // Translates the module, cell, or cell function, respectively, into Z3-space.
  absl::Status Translate();
  absl::Status TranslateCell(const netlist::rtl::Cell& cell);
  xabsl::StatusOr<Z3_ast> TranslateFunction(const netlist::rtl::Cell& cell,
                                            const netlist::function::Ast ast);

  Z3_context ctx_;
  const netlist::rtl::Module* module_;

  // Maps a NetDef to a Z3 entity.
  absl::flat_hash_map<netlist::rtl::NetRef, Z3_ast> translated_;

  const absl::flat_hash_map<std::string, const netlist::rtl::Module*>&
      module_refs_;

  // TODO(rspringer): Eliminate the need for this by properly handling cells
  // with state_function attributes.
  // List of cells for which all outputs are unconditionally set to 1.
  absl::flat_hash_set<std::string> high_cells_;
};

}  // namespace z3
}  // namespace solvers
}  // namespace xls

#endif  // THIRD_PARTY_XLS_SOLVERS_Z3_NETLIST_TRANSLATOR_H_
