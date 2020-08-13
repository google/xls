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

#ifndef XLS_NETLIST_CELL_LIBRARY_H_
#define XLS_NETLIST_CELL_LIBRARY_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/common/status/statusor.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {

enum class CellKind {
  kFlop,
  kInverter,
  kBuffer,
  kNand,
  kNor,
  kMultiplexer,
  kXor,
  kOther,
};

std::string CellKindToString(CellKind kind);

// Captures useful information about (one of) a cell's output pin(s). Currently,
// that's just its name and string description of its calculating function.
struct OutputPin {
  std::string name;
  std::string function;
};
inline bool operator==(const OutputPin& lhs, const OutputPin& rhs) {
  return lhs.name == rhs.name && lhs.function == rhs.function;
}
template <typename H>
H AbslHashValue(H state, const OutputPin& p) {
  return H::combine(std::move(state), p.name, p.function);
}

// StateTable provides methods for querying Liberty "statetable" attributes, as
// captured by the StateTableProto structure.
// This table does not currently handle stateful operations, i.e., those that
// require information outside a given row of the table. Consider the row:
// "A B C : D : D"
// "H - - : - : N"; this indicates that the value of the internal signal ("D")
// is unchanged by that stimulus...but the output value depends on state not
// captured here. To model that, a StateTable should be wrapped in a stateful
// class (not implemented here).
class StateTable {
 public:
  // InputStimulus provides one input for evaluation to the table.
  using InputStimulus = absl::flat_hash_map<std::string, bool>;

  // Constructs a StateTable object from the matching proto.
  static xabsl::StatusOr<StateTable> FromProto(const StateTableProto& proto);

  // Gets the value of the given signal when the table is presented with the
  // specified stimulus.
  // return true/false
  xabsl::StatusOr<bool> GetSignalValue(const InputStimulus& stimulus,
                                       absl::string_view signal);

 private:
  using RowStimulus = absl::flat_hash_map<std::string, StateTableSignal>;
  using RowResponse = absl::flat_hash_map<std::string, StateTableSignal>;
  using Row = std::pair<RowStimulus, RowResponse>;

  StateTable(const std::vector<Row>& rows,
             const absl::flat_hash_set<std::string>& signals,
             const StateTableProto& proto);

  // Returns true if the given input stimulus matches the given table row.
  xabsl::StatusOr<bool> MatchRow(const Row& row,
                                 const InputStimulus& input_stimulus);

  // True if the value of "name" in stimulus matches the given bool value or is
  // "don't care".
  bool SignalMismatch(absl::string_view name, bool value,
                      const RowStimulus& stimulus);

  absl::flat_hash_set<std::string> signals_;

  // We preprocess the proto to combine input signals (both true inputs and
  // internal state signals) for ease-of-lookup.
  std::vector<Row> rows_;
  StateTableProto proto_;
};

// Represents an entry in the cell library, listing inputs/outputs an the name
// of the cell module.
class CellLibraryEntry {
 public:
  static xabsl::StatusOr<CellLibraryEntry> FromProto(
      const CellLibraryEntryProto& proto);

  // InputNamesContainer and OutputNamesContainer are expected to be containers
  // of std::strings.
  template <typename InputNamesContainer, typename OutputPinsContainer>
  CellLibraryEntry(CellKind kind, absl::string_view name,
                   const InputNamesContainer& input_names,
                   const OutputPinsContainer& output_pins,
                   absl::optional<std::string> clock_name = absl::nullopt)
      : kind_(kind),
        name_(name),
        input_names_(input_names.begin(), input_names.end()),
        output_pins_(output_pins.begin(), output_pins.end()),
        clock_name_(clock_name) {}

  CellLibraryEntry(CellKind kind, absl::string_view name,
                   const google::protobuf::RepeatedPtrField<std::string>& input_names,
                   const google::protobuf::RepeatedPtrField<OutputPinProto>& output_pins,
                   absl::optional<std::string> clock_name = absl::nullopt)
      : kind_(kind),
        name_(name),
        input_names_(input_names.begin(), input_names.end()),
        clock_name_(clock_name) {
    output_pins_.reserve(output_pins.size());
    for (const auto& proto : output_pins) {
      output_pins_.push_back({proto.name(), proto.function()});
    }
  }

  CellKind kind() const { return kind_; }
  const std::string& name() const { return name_; }
  absl::Span<const std::string> input_names() const { return input_names_; }
  absl::Span<const OutputPin> output_pins() const { return output_pins_; }
  absl::optional<std::string> clock_name() const { return clock_name_; }

  CellLibraryEntryProto ToProto() const;

 private:
  CellKind kind_;
  std::string name_;
  std::vector<std::string> input_names_;
  std::vector<OutputPin> output_pins_;
  absl::optional<std::string> clock_name_;
};

// Represents a library of cells. The definitions (represented in
// "CellLibraryEntry"s) are referred to from Cell instances in the netlist
// Module.
class CellLibrary {
 public:
  static xabsl::StatusOr<CellLibrary> FromProto(const CellLibraryProto& proto);

  // Returns a NOT_FOUND status if there is not entry with the given name.
  xabsl::StatusOr<const CellLibraryEntry*> GetEntry(
      absl::string_view name) const;

  absl::Status AddEntry(CellLibraryEntry entry);

  CellLibraryProto ToProto() const;

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<CellLibraryEntry>> entries_;
};

}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_CELL_LIBRARY_H_
