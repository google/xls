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

#ifndef XLS_NETLIST_CELL_LIBRARY_H_
#define XLS_NETLIST_CELL_LIBRARY_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
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

// Definition of the "statetable" group of values from section 2.1.4 of the
// Liberty file standard (2007.03).
// Please keep this declaration in sync with that in netlist.proto!
enum class StateTableSignal {
  kInvalid,
  kLow,
  kHigh,
  kDontCare,
  kHighOrLow,
  kLowOrHigh,
  kRising,
  kFalling,
  kNotRising,
  kNotFalling,
  kNoChange,
  kX,
};

// Captures useful information about (one of) a cell's output pin(s). Currently,
// that's just its name and string description of its calculating function.
struct OutputPin {
  std::string name;
  std::string function;
};

// AbstractStateTable provides methods for querying Liberty "statetable"
// attributes, as captured by the StateTableProto structure.  This table does
// not currently handle stateful operations, i.e., those that
// require information outside a given row of the table. Consider the row:
// "A B C : D : D"
// "H - - : - : N"; this indicates that the value of the internal signal ("D")
// is unchanged by that stimulus...but the next state value depends on state not
// captured here. To model that, a AbstractStateTable should be wrapped in a
// stateful class (not implemented here).
template <typename EvalT = bool>
class AbstractStateTable {
 public:
  // InputStimulus provides one input for evaluation to the table.
  using InputStimulus = absl::flat_hash_map<std::string, EvalT>;

  // Constructs an AbstractStateTable object from the matching proto.
  static absl::StatusOr<AbstractStateTable> FromProto(
      const StateTableProto& proto, EvalT zero, EvalT one);

  // Default form when EvalT can be constructed from a boolean value.
  template <typename = std::is_constructible<EvalT, bool>>
  static absl::StatusOr<AbstractStateTable> FromProto(
      const StateTableProto& proto) {
    return FromProto(proto, EvalT{false}, EvalT{true});
  }

  // Create an AbstractStateTable object for the equivalent LUT4 cell and mask.
  static AbstractStateTable FromLutMask(uint16_t lut_mask, EvalT zero,
                                        EvalT one);

  template <typename = std::is_constructible<EvalT, bool>>
  static AbstractStateTable FromLutMask(uint16_t lut_mask) {
    return FromLutMask(lut_mask, EvalT{false}, EvalT{true});
  }

  // Gets the value of the given signal when the table is presented with the
  // specified stimulus.
  // return true/false
  absl::StatusOr<EvalT> GetSignalValue(const InputStimulus& stimulus,
                                       std::string_view signal) const;

  const absl::flat_hash_set<std::string>& internal_signals() const {
    return internal_signals_;
  }

  // Description of a row in the state table - a mapping of "stimulus", i.e.,
  // input signals, to the "response", the next state of the internal signals.
  using RowStimulus = absl::flat_hash_map<std::string, StateTableSignal>;
  using RowResponse = absl::flat_hash_map<std::string, StateTableSignal>;
  struct Row {
    RowStimulus stimulus;
    RowResponse response;
  };
  const std::vector<Row>& rows() const { return rows_; }

  // Return the protobuf representation of this table.
  absl::StatusOr<StateTableProto> ToProto() const;

 private:
  AbstractStateTable(const std::vector<Row>& rows,
                     const absl::flat_hash_set<std::string>& signals,
                     const absl::flat_hash_set<std::string>& internal_names,
                     EvalT zero, EvalT one);

  // Returns true if the given input stimulus matches the given table row.
  absl::StatusOr<bool> MatchRow(const Row& row,
                                const InputStimulus& input_stimulus) const;

  // True if the value of "name" in stimulus matches the given bool value or is
  // "don't care".
  bool SignalMatches(std::string_view name, EvalT value,
                     const RowStimulus& stimulus) const;

  // The set of all signals (input and internal/output) in this state table.
  absl::flat_hash_set<std::string> signals_;

  // Subset of signals_ - the output/internal signals.
  absl::flat_hash_set<std::string> internal_signals_;

  absl::flat_hash_set<std::string> internal_names_;

  // We preprocess the proto to combine input signals (both true inputs and
  // internal state signals) for ease-of-lookup.
  std::vector<Row> rows_;

  // The values representing zero and one respectively.
  EvalT zero_;
  EvalT one_;
};

using StateTable = AbstractStateTable<>;

// Represents an entry in the cell library, listing inputs/outputs an the name
// of the cell module.
template <typename EvalT = bool>
class AbstractCellLibraryEntry {
 public:
  typedef absl::flat_hash_map<std::string, std::string> OutputPinToFunction;

  static absl::StatusOr<AbstractCellLibraryEntry> FromProto(
      const CellLibraryEntryProto& proto, EvalT zero, EvalT one);

  // InputNamesContainer and OutputNamesContainer are expected to be containers
  // of std::strings.
  template <typename InputNamesContainer = std::vector<std::string>>
  AbstractCellLibraryEntry(
      CellKind kind, std::string_view name,
      const InputNamesContainer& input_names,
      const OutputPinToFunction& output_pin_to_function,
      const std::optional<AbstractStateTable<EvalT>> state_table,
      std::optional<std::string> clock_name = std::nullopt)
      : kind_(kind),
        name_(name),
        input_names_(input_names.begin(), input_names.end()),
        output_pin_to_function_(output_pin_to_function),
        state_table_(state_table),
        clock_name_(clock_name) {}

  CellKind kind() const { return kind_; }
  const std::string& name() const { return name_; }
  absl::Span<const std::string> input_names() const { return input_names_; }
  const OutputPinToFunction& output_pin_to_function() const {
    return output_pin_to_function_;
  }
  const std::optional<AbstractStateTable<EvalT>>& state_table() const {
    return state_table_;
  }
  std::optional<std::string> clock_name() const { return clock_name_; }

  absl::StatusOr<CellLibraryEntryProto> ToProto() const;

 private:
  CellKind kind_;
  std::string name_;
  std::vector<std::string> input_names_;
  OutputPinToFunction output_pin_to_function_;
  std::optional<AbstractStateTable<EvalT>> state_table_;
  std::optional<std::string> clock_name_;
};

using CellLibraryEntry = AbstractCellLibraryEntry<>;

// Represents a library of cells. The definitions (represented in
// "AbstractCellLibraryEntry"s) are referred to from Cell instances in the
// netlist Module.
template <typename EvalT = bool>
class AbstractCellLibrary {
 public:
  AbstractCellLibrary() = default;
  static absl::StatusOr<AbstractCellLibrary<EvalT>> FromProto(
      const CellLibraryProto& proto, EvalT zero, EvalT one);

  template <typename = std::is_constructible<EvalT, bool>>
  static absl::StatusOr<AbstractCellLibrary<EvalT>> FromProto(
      const CellLibraryProto& proto) {
    return FromProto(proto, EvalT{false}, EvalT{true});
  }

  // Returns a NOT_FOUND status if there is not entry with the given name.
  absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*> GetEntry(
      std::string_view name) const;

  absl::Status AddEntry(AbstractCellLibraryEntry<EvalT> entry);

  absl::StatusOr<CellLibraryProto> ToProto() const;

 private:
  absl::flat_hash_map<std::string,
                      std::unique_ptr<AbstractCellLibraryEntry<EvalT>>>
      entries_;
};

using CellLibrary = AbstractCellLibrary<>;

absl::StatusOr<StateTableSignal> StateTableSignalFromProto(
    StateTableSignalProto proto);

// Constructs an AbstractStateTable object from the matching proto.
template <typename EvalT>
/* static */ absl::StatusOr<AbstractStateTable<EvalT>>
AbstractStateTable<EvalT>::FromProto(const StateTableProto& proto, EvalT zero,
                                     EvalT one) {
  std::vector<Row> rows;
  absl::flat_hash_set<std::string> signals;
  absl::flat_hash_set<std::string> internal_names;

  for (const auto& row : proto.rows()) {
    RowStimulus stimulus;
    for (const auto& kv : row.input_signals()) {
      XLS_ASSIGN_OR_RETURN(stimulus[kv.first],
                           StateTableSignalFromProto(kv.second));
      signals.insert(kv.first);
    }
    for (const auto& kv : row.internal_signals()) {
      XLS_ASSIGN_OR_RETURN(stimulus[kv.first],
                           StateTableSignalFromProto(kv.second));
      signals.insert(kv.first);
    }

    RowResponse response;
    for (const auto& kv : row.next_internal_signals()) {
      XLS_ASSIGN_OR_RETURN(response[kv.first],
                           StateTableSignalFromProto(kv.second));
    }

    rows.push_back({stimulus, response});
  }

  for (const auto& name : proto.internal_names()) {
    internal_names.insert(name);
  }

  return AbstractStateTable(rows, signals, internal_names, zero, one);
}

template <typename EvalT>
/* static */ AbstractStateTable<EvalT> AbstractStateTable<EvalT>::FromLutMask(
    uint16_t lut_mask, EvalT zero, EvalT one) {
  std::vector<Row> rows;
  // For a mask with bits b_15, b_14, ... b_1, b_0, the LUT4 with that mask
  // implements the following truth table:
  //
  // I3 | I2 | I1 | I0 | out
  // -----------------------
  // 0  | 0  | 0  | 0  | b_0
  // 0  | 0  | 0  | 1  | b_1
  // 0  | 0  | 1  | 0  | b_2
  // ... etc.
  //
  // Note that this specifically for the iCE40 FPGA, other vendors may have
  // different conventions.
  rows.reserve(16);
  for (int row = 0; row < 16; ++row) {
    StateTableSignal i0 =
        (row & 0b0001) > 0 ? StateTableSignal::kHigh : StateTableSignal::kLow;
    StateTableSignal i1 =
        (row & 0b0010) > 0 ? StateTableSignal::kHigh : StateTableSignal::kLow;
    StateTableSignal i2 =
        (row & 0b0100) > 0 ? StateTableSignal::kHigh : StateTableSignal::kLow;
    StateTableSignal i3 =
        (row & 0b1000) > 0 ? StateTableSignal::kHigh : StateTableSignal::kLow;
    StateTableSignal out = ((lut_mask >> row) & 1) == 1
                               ? StateTableSignal::kHigh
                               : StateTableSignal::kLow;
    // "X" comes from the specified function_name in
    // Netlist::GetOrCreateLut4CellEntry
    rows.push_back(Row{
        .stimulus = {{"I0", i0},
                     {"I1", i1},
                     {"I2", i2},
                     {"I3", i3},
                     {"X", StateTableSignal::kDontCare}},
        .response = {{"X", out}},
    });
  }

  return AbstractStateTable(rows, /*signals*/ {"I0", "I1", "I2", "I3", "X"},
                            /*internal_names*/ {"X"}, zero, one);
}

template <typename EvalT>
AbstractStateTable<EvalT>::AbstractStateTable(
    const std::vector<Row>& rows,
    const absl::flat_hash_set<std::string>& signals,
    const absl::flat_hash_set<std::string>& internal_names, EvalT zero,
    EvalT one)
    : signals_(signals),
      internal_names_(internal_names),
      rows_(rows),
      zero_(zero),
      one_(one) {
  // Get the "output" side ("RowResponse") of the first row in the table,
  // and cache the signal names from there.
  for (const auto& kv : rows_[0].response) {
    internal_signals_.insert(kv.first);
  }
}

// TODO(rspringer): 2020/08/28 - We don't handle transition signals (rising,
// falling, etc.) here yet.
template <typename EvalT>
bool AbstractStateTable<EvalT>::SignalMatches(
    std::string_view name, EvalT value, const RowStimulus& stimulus) const {
  // Input signals can be either high or low - we don't manage persistent state
  // inside this class; we just represent the table.
  if constexpr (std::is_convertible<EvalT, int>()) {
    StateTableSignal value_signal =
        value ? StateTableSignal::kHigh : StateTableSignal::kLow;
    // If the stimulus is L/H or H/L, it has to match; we'll figure out what to
    // do with the output later (in GetSignalValue()).
    if (stimulus.contains(name) &&
        (stimulus.at(name) == StateTableSignal::kLowOrHigh ||
         stimulus.at(name) == StateTableSignal::kHighOrLow ||
         value_signal == stimulus.at(name) ||
         stimulus.at(name) == StateTableSignal::kDontCare)) {
      return true;
    }
  }

  return false;
}

template <typename EvalT>
absl::StatusOr<bool> AbstractStateTable<EvalT>::MatchRow(
    const Row& row, const InputStimulus& input_stimulus) const {
  absl::flat_hash_set<std::string> unspecified_inputs = signals_;
  const RowStimulus& row_stimulus = row.stimulus;
  for (const auto& kv : input_stimulus) {
    const std::string& name = kv.first;
    EvalT value = kv.second;

    XLS_RET_CHECK(signals_.contains(name));
    // Check that, for every stimulus signal, that there's not a mismatch
    // (i.e., the row has "don't care" or a matching value for the input.
    if (!SignalMatches(name, value, row_stimulus)) {
      return false;
    }

    // We've matched this input!
    unspecified_inputs.erase(name);
  }

  // We've matched all inputs - now we have to verify that all unspecified
  // inputs are "don't care".
  for (const std::string& name : unspecified_inputs) {
    if (row_stimulus.at(name) != StateTableSignal::kDontCare) {
      return false;
    }
  }
  return true;
}

template <typename EvalT>
absl::StatusOr<EvalT> AbstractStateTable<EvalT>::GetSignalValue(
    const InputStimulus& input_stimulus, std::string_view signal) const {
  // Find a row matching the stimulus or return error.
  if (!internal_names_.contains(signal)) {
    return absl::InvalidArgumentError("No internal name matched signal");
  }

  for (const Row& row : rows_) {
    XLS_ASSIGN_OR_RETURN(bool match, MatchRow(row, input_stimulus));
    if (match) {
      // If "switched" (e.g., kLowOrHigh) are used, then the next value is
      // identity if both sides are the same, and inversion otherwise.
      StateTableSignal stimulus_signal = row.stimulus.at(signal);
      StateTableSignal response_signal = row.response.at(signal);
      bool switched_signal = stimulus_signal == StateTableSignal::kLowOrHigh ||
                             stimulus_signal == StateTableSignal::kHighOrLow;
      if (switched_signal) {
        if (stimulus_signal == response_signal) {
          return input_stimulus.at(signal);
        }
        EvalT value = input_stimulus.at(signal);
        return !value;
      }

      return response_signal == StateTableSignal::kHigh ? one_ : zero_;
    }
  }

  return absl::NotFoundError("No matching row found in the table.");
}

absl::StatusOr<StateTableSignalProto> ProtoFromStateTableSignal(
    StateTableSignal signal);

template <typename EvalT>
absl::StatusOr<StateTableProto> AbstractStateTable<EvalT>::ToProto() const {
  StateTableProto proto;
  // First, extract the signals (internal and input) from the first row (each
  // row contains every signal).
  for (const auto& kv_stimulus : rows_[0].stimulus) {
    const std::string& signal_name = kv_stimulus.first;
    if (internal_signals_.contains(signal_name)) {
      proto.add_internal_names(signal_name);
    } else {
      proto.add_input_names(signal_name);
    }
  }

  for (const Row& row : rows_) {
    StateTableRow* row_proto = proto.add_rows();
    for (const auto& kv_stimulus : row.stimulus) {
      const std::string& signal_name = kv_stimulus.first;
      XLS_ASSIGN_OR_RETURN(StateTableSignalProto signal_value,
                           ProtoFromStateTableSignal(kv_stimulus.second));
      if (internal_signals_.contains(signal_name)) {
        row_proto->mutable_internal_signals()->insert(
            {signal_name, signal_value});
      } else {
        row_proto->mutable_input_signals()->insert({signal_name, signal_value});
      }
    }

    for (const auto& kv_response : row.response) {
      XLS_ASSIGN_OR_RETURN(StateTableSignalProto signal_value,
                           ProtoFromStateTableSignal(kv_response.second));
      row_proto->mutable_next_internal_signals()->insert(
          {kv_response.first, signal_value});
    }
  }
  return proto;
}

absl::StatusOr<CellKind> CellKindFromProto(CellKindProto proto);

template <typename EvalT>
/* static */ absl::StatusOr<AbstractCellLibraryEntry<EvalT>>
AbstractCellLibraryEntry<EvalT>::FromProto(const CellLibraryEntryProto& proto,
                                           EvalT zero, EvalT one) {
  XLS_ASSIGN_OR_RETURN(CellKind cell_kind, CellKindFromProto(proto.kind()));

  OutputPinListProto output_pin_list = proto.output_pin_list();
  OutputPinToFunction pins;
  pins.reserve(output_pin_list.pins_size());
  for (const auto& proto : output_pin_list.pins()) {
    pins[proto.name()] = proto.function();
  }

  std::optional<AbstractStateTable<EvalT>> state_table;
  if (proto.has_state_table()) {
    XLS_ASSIGN_OR_RETURN(state_table, AbstractStateTable<EvalT>::FromProto(
                                          proto.state_table(), zero, one));
  }

  return AbstractCellLibraryEntry(cell_kind, proto.name(), proto.input_names(),
                                  pins, state_table);
}

template <typename EvalT>
absl::StatusOr<CellLibraryEntryProto> AbstractCellLibraryEntry<EvalT>::ToProto()
    const {
  CellLibraryEntryProto proto;
  switch (kind_) {
    case CellKind::kFlop:
      proto.set_kind(CellKindProto::FLOP);
      break;
    case CellKind::kInverter:
      proto.set_kind(CellKindProto::INVERTER);
      break;
    case CellKind::kBuffer:
      proto.set_kind(CellKindProto::BUFFER);
      break;
    case CellKind::kNand:
      proto.set_kind(CellKindProto::NAND);
      break;
    case CellKind::kNor:
      proto.set_kind(CellKindProto::NOR);
      break;
    case CellKind::kMultiplexer:
      proto.set_kind(CellKindProto::MULTIPLEXER);
      break;
    case CellKind::kXor:
      proto.set_kind(CellKindProto::XOR);
      break;
    case CellKind::kOther:
      proto.set_kind(CellKindProto::OTHER);
      break;
  }
  proto.set_name(name_);
  for (const std::string& input_name : input_names_) {
    proto.add_input_names(input_name);
  }
  if (state_table_.has_value()) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_state_table(),
                         state_table_.value().ToProto());
  }

  OutputPinListProto* pin_list = proto.mutable_output_pin_list();
  for (const auto& kv : output_pin_to_function_) {
    OutputPinProto* pin_proto = pin_list->add_pins();
    pin_proto->set_name(kv.first);
    pin_proto->set_function(kv.second);
  }
  return proto;
}

template <typename EvalT>
/* static */ absl::StatusOr<AbstractCellLibrary<EvalT>>
AbstractCellLibrary<EvalT>::FromProto(const CellLibraryProto& proto, EvalT zero,
                                      EvalT one) {
  AbstractCellLibrary cell_library;
  for (const CellLibraryEntryProto& entry_proto : proto.entries()) {
    XLS_ASSIGN_OR_RETURN(auto entry, AbstractCellLibraryEntry<EvalT>::FromProto(
                                         entry_proto, zero, one));
    XLS_RETURN_IF_ERROR(cell_library.AddEntry(std::move(entry)));
  }
  return cell_library;
}

template <typename EvalT>
absl::StatusOr<CellLibraryProto> AbstractCellLibrary<EvalT>::ToProto() const {
  CellLibraryProto proto;
  for (const auto& entry : entries_) {
    XLS_ASSIGN_OR_RETURN(*proto.add_entries(), entry.second->ToProto());
  }
  return proto;
}

template <typename EvalT>
absl::Status AbstractCellLibrary<EvalT>::AddEntry(
    AbstractCellLibraryEntry<EvalT> entry) {
  if (entries_.find(entry.name()) != entries_.end()) {
    return absl::FailedPreconditionError(
        "Attempting to register a cell library entry with a duplicate name: " +
        entry.name());
  }
  entries_.insert(
      {entry.name(), std::make_unique<AbstractCellLibraryEntry<EvalT>>(entry)});
  return absl::OkStatus();
}

template <typename EvalT>
absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*>
AbstractCellLibrary<EvalT>::GetEntry(std::string_view name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Cell not found in library: ", name));
  }
  return it->second.get();
}

}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_CELL_LIBRARY_H_
