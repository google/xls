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

// Data structure that represents netlists (e.g. ones that have been parsed in
// from the synthesis flow).

#ifndef XLS_NETLIST_NETLIST_H_
#define XLS_NETLIST_NETLIST_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/common/integral_types.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/statusor.h"
#include "xls/netlist/cell_library.h"

namespace xls {
namespace netlist {
namespace rtl {

// Forward declaration for use in NetRef.
class NetDef;

// Refers to an ID inside the module's wires_ array.
using NetRef = NetDef*;

// Represents a cell instantiated in the netlist.
class Cell {
 public:
  // Simple utility struct to capture data for a Cell's input or output pins.
  struct Pin {
    // Name of the pin in the cell's function description.
    std::string name;

    // The associated net from the netlist.
    NetRef netref;
  };

  // In this class, both "inputs" and "outputs" are maps of cell input/output
  // pin name to the NetDef/Ref used as that input in a given instance.
  // For outputs, if a pin isn't used, then it won't be present in the provided
  // map.
  // "dummy_net" is a ref to the "dummy" cell used by the containing module for
  // output wires that aren't connected to any cells.
  static xabsl::StatusOr<Cell> Create(
      const CellLibraryEntry* cell_library_entry, absl::string_view name,
      const absl::flat_hash_map<std::string, NetRef>&
          named_parameter_assignments,
      absl::optional<NetRef> clock, NetRef dummy_net);

  const CellLibraryEntry* cell_library_entry() const {
    return cell_library_entry_;
  }
  const std::string& name() const { return name_; }
  CellKind kind() const { return cell_library_entry_->kind(); }

  absl::Span<const Pin> inputs() const { return inputs_; }
  absl::Span<const Pin> outputs() const { return outputs_; }
  const absl::optional<NetRef>& clock() const { return clock_; }

 private:
  Cell(const CellLibraryEntry* cell_library_entry, absl::string_view name,
       const std::vector<Pin>& inputs, const std::vector<Pin>& outputs,
       absl::optional<NetRef> clock)
      : cell_library_entry_(cell_library_entry),
        name_(name),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        clock_(clock) {}

  const CellLibraryEntry* cell_library_entry_;
  std::string name_;  // Instance name.
  std::vector<Pin> inputs_;
  std::vector<Pin> outputs_;
  absl::optional<NetRef> clock_;
};

// Definition of a net. Note this may be augmented with a def/use chain in the
// future.
class NetDef {
 public:
  explicit NetDef(absl::string_view name) : name_(name) {}

  const std::string& name() const { return name_; }

  // Called to note that a cell is connected to this net.
  void NoteConnectedCell(Cell* cell) { connected_cells_.push_back(cell); }

  absl::Span<Cell* const> connected_cells() const { return connected_cells_; }

  // Helper for getting the connected cells without one that is known to be
  // connected (e.g. a driver). Note: could be optimized to give a smart
  // view/iterator object that filters out to_remove without instantiating
  // storage.
  xabsl::StatusOr<std::vector<Cell*>> GetConnectedCellsSans(
      Cell* to_remove) const;

 private:
  std::string name_;
  std::vector<Cell*> connected_cells_;
};

// Kinds of wire declarations that can be made in the netlist module.
enum class NetDeclKind {
  kInput,
  kOutput,
  kWire,
};

// Represents the module containing the netlist info.
class Module {
 public:
  explicit Module(absl::string_view name) : name_(name) {
    // Zero and one values are present in netlists as cell inputs (which we
    // interpret as wires), but aren't explicitly declared, so we create them as
    // wires here.
    zero_ = AddOrResolveNumber(0).value();
    one_ = AddOrResolveNumber(1).value();

    // We need a "dummy" wire to serve as the sink for any unused cell outputs.
    // Even if a cell output is unused, we need some dummy value there to
    // maintain the correspondance between a CellLibraryEntry's pinout and that
    // of a Cell [object].
    // TODO(rspringer): Improve APIs so we don't have to match array indexes
    // between these types.
    constexpr const char kDummyName[] = "__dummy__net_decl__";
    XLS_CHECK_OK(AddNetDecl(NetDeclKind::kWire, kDummyName));
    dummy_ = ResolveNet(kDummyName).value();
  }

  const std::string& name() const { return name_; }

  // Returns a representation of this module as a CellLibraryEntry.
  // This does not currently support stateful modules, e.g., those with
  // "state_table"-like attributes.
  const CellLibraryEntry* AsCellLibraryEntry() const;

  xabsl::StatusOr<Cell*> AddCell(Cell cell);

  absl::Status AddNetDecl(NetDeclKind kind, absl::string_view name);

  // Returns a NetRef to the given number, creating a NetDef if necessary.
  xabsl::StatusOr<NetRef> AddOrResolveNumber(int64 number);

  xabsl::StatusOr<NetRef> ResolveNumber(int64 number) const;

  xabsl::StatusOr<NetRef> ResolveNet(absl::string_view name) const;

  // Returns a reference to a "dummy" net - this is needed for cases where one
  // of a cell's output pins isn't actually used.
  NetRef GetDummyRef() const { return dummy_; }

  xabsl::StatusOr<Cell*> ResolveCell(absl::string_view name) const;

  absl::Span<const std::unique_ptr<NetDef>> nets() const { return nets_; }
  absl::Span<const std::unique_ptr<Cell>> cells() const { return cells_; }

  const std::vector<NetRef>& inputs() const { return inputs_; }
  const std::vector<NetRef>& outputs() const { return outputs_; }

 private:
  std::string name_;
  std::vector<NetRef> inputs_;
  std::vector<NetRef> outputs_;
  std::vector<NetRef> wires_;
  std::vector<std::unique_ptr<NetDef>> nets_;
  std::vector<std::unique_ptr<Cell>> cells_;
  NetRef zero_;
  NetRef one_;
  NetRef dummy_;

  mutable absl::optional<CellLibraryEntry> cell_library_entry_;
};

// A Netlist contains all modules present in a single file.
class Netlist {
 public:
  void AddModule(std::unique_ptr<Module> module);
  xabsl::StatusOr<const Module*> GetModule(
      const std::string& module_name) const;
  const absl::Span<const std::unique_ptr<Module>> modules() { return modules_; }

 private:
  std::vector<std::unique_ptr<Module>> modules_;
};

}  // namespace rtl
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_NETLIST_H_
