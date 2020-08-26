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
#ifndef XLS_NETLIST_INTERPRETER_H_
#define XLS_NETLIST_INTERPRETER_H_

#include <deque>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/netlist/function_parser.h"
#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {

// Interprets Netlists/Modules given a set of input values and returns the
// resulting value.
class Interpreter {
 public:
  explicit Interpreter(rtl::Netlist* netlist);

  // Interprets the given module with the given input mapping.
  //  - inputs: Mapping of module input wire to value. Must have the same size
  //    as module->inputs();
  //  - dump_cells: List of cells whose inputs and outputs should be dumped
  //    on evaluation.
  xabsl::StatusOr<absl::flat_hash_map<const rtl::NetRef, bool>> InterpretModule(
      const rtl::Module* module,
      const absl::flat_hash_map<const rtl::NetRef, bool>& inputs,
      absl::Span<const std::string> dump_cells = {});

 private:
  // Returns true if the specified NetRef is an output of the given cell.
  bool IsCellOutput(const rtl::Cell& cell, const rtl::NetRef ref);

  absl::Status InterpretCell(
      const rtl::Cell& cell,
      absl::flat_hash_map<const rtl::NetRef, bool>* processed_wires);

  xabsl::StatusOr<bool> InterpretFunction(
      const rtl::Cell& cell, const function::Ast& ast,
      const absl::flat_hash_map<const rtl::NetRef, bool>& processed_wires);

  rtl::Netlist* netlist_;
};

}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_INTERPRETER_H_
