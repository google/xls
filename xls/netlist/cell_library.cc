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

#include "xls/netlist/cell_library.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {
namespace {

xabsl::StatusOr<CellKind> CellKindFromProto(CellKindProto proto) {
  switch (proto) {
    case INVALID:
      break;
    case FLOP:
      return CellKind::kFlop;
    case INVERTER:
      return CellKind::kInverter;
    case BUFFER:
      return CellKind::kBuffer;
    case NAND:
      return CellKind::kNand;
    case NOR:
      return CellKind::kNor;
    case MULTIPLEXER:
      return CellKind::kMultiplexer;
    case XOR:
      return CellKind::kXor;
    case OTHER:
      return CellKind::kOther;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid proto value for conversion to CellKind: %d", proto));
}

}  // namespace

std::string CellKindToString(CellKind kind) {
  switch (kind) {
    case CellKind::kFlop:
      return "flop";
    case CellKind::kInverter:
      return "inverter";
    case CellKind::kBuffer:
      return "buffer";
    case CellKind::kNand:
      return "nand";
    case CellKind::kNor:
      return "nor";
    case CellKind::kXor:
      return "xor";
    case CellKind::kMultiplexer:
      return "multiplexer";
    case CellKind::kOther:
      return "other";
  }
  return absl::StrFormat("<invalid CellKind(%d)>", static_cast<int64>(kind));
}

/* static */ xabsl::StatusOr<StateTable> StateTable::FromProto(
    const StateTableProto& proto) {
  std::vector<Row> rows;
  absl::flat_hash_set<std::string> signals;
  for (const auto& row : proto.rows()) {
    RowStimulus stimulus;
    for (const auto& kv : row.input_signals()) {
      stimulus[kv.first] = kv.second;
      signals.insert(kv.first);
    }
    for (const auto& kv : row.internal_signals()) {
      stimulus[kv.first] = kv.second;
      signals.insert(kv.first);
    }

    RowResponse response;
    for (const auto& kv : row.output_signals()) {
      response[kv.first] = kv.second;
    }

    rows.push_back({stimulus, response});
  }

  return StateTable(rows, signals, proto);
}

StateTable::StateTable(const std::vector<Row>& rows,
                       const absl::flat_hash_set<std::string>& signals,
                       const StateTableProto& proto)
    : signals_(signals), rows_(rows), proto_(proto) {}

bool StateTable::SignalMismatch(absl::string_view name, bool value,
                                const RowStimulus& stimulus) {
  // Input signals can be either high or low - we don't manage persistent state
  // inside this class; we just represent the table.
  StateTableSignal value_signal =
      value ? STATE_TABLE_SIGNAL_HIGH : STATE_TABLE_SIGNAL_LOW;
  if (stimulus.contains(name) && value_signal != stimulus.at(name)) {
    return true;
  }

  return false;
}

xabsl::StatusOr<bool> StateTable::MatchRow(
    const Row& row, const InputStimulus& input_stimulus) {
  absl::flat_hash_set<std::string> unspecified_inputs = signals_;
  const RowStimulus& row_stimulus = row.first;
  for (const auto& kv : input_stimulus) {
    const std::string& name = kv.first;
    bool value = kv.second;

    XLS_RET_CHECK(signals_.contains(name));
    // Check that, for every stimulus signal, that there's not a mismatch
    // (i.e., the row has "don't care" or a matching value for the input.
    if (SignalMismatch(name, value, row_stimulus)) {
      return false;
    }

    // We've matched this input!
    unspecified_inputs.erase(name);
  }

  // We've matched all inputs - now we have to verify that all unspecified
  // inputs are "don't care".
  for (const std::string& name : unspecified_inputs) {
    if (row_stimulus.at(name) != STATE_TABLE_SIGNAL_DONTCARE) {
      return false;
    }
  }
  return true;
}

xabsl::StatusOr<bool> StateTable::GetSignalValue(
    const InputStimulus& input_stimulus, absl::string_view signal) {
  // Find a row matching the stimulus or return error.
  XLS_RET_CHECK(std::find(proto_.internal_names().begin(),
                          proto_.internal_names().end(),
                          signal) != proto_.internal_names().end())
      << "Can't find internal signal \"" << signal << "\"";

  for (const Row& row : rows_) {
    XLS_ASSIGN_OR_RETURN(bool match, MatchRow(row, input_stimulus));
    if (match) {
      return row.second.at(signal) == STATE_TABLE_SIGNAL_HIGH;
    }
  }

  return absl::NotFoundError("No matching row found in the table.");
}

/* static */ xabsl::StatusOr<CellLibraryEntry> CellLibraryEntry::FromProto(
    const CellLibraryEntryProto& proto) {
  XLS_ASSIGN_OR_RETURN(CellKind cell_kind, CellKindFromProto(proto.kind()));
  return CellLibraryEntry(cell_kind, proto.name(), proto.input_names(),
                          proto.output_pins());
}

CellLibraryEntryProto CellLibraryEntry::ToProto() const {
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
  for (const OutputPin& output_pin : output_pins_) {
    OutputPinProto* pin_proto = proto.add_output_pins();
    pin_proto->set_name(output_pin.name);
    pin_proto->set_function(output_pin.function);
  }
  return proto;
}

/* static */ xabsl::StatusOr<CellLibrary> CellLibrary::FromProto(
    const CellLibraryProto& proto) {
  CellLibrary cell_library;
  for (const CellLibraryEntryProto& entry_proto : proto.entries()) {
    XLS_ASSIGN_OR_RETURN(auto entry, CellLibraryEntry::FromProto(entry_proto));
    XLS_RETURN_IF_ERROR(cell_library.AddEntry(std::move(entry)));
  }
  return cell_library;
}

CellLibraryProto CellLibrary::ToProto() const {
  CellLibraryProto proto;
  for (const auto& entry : entries_) {
    *proto.add_entries() = entry.second->ToProto();
  }
  return proto;
}

absl::Status CellLibrary::AddEntry(CellLibraryEntry entry) {
  if (entries_.find(entry.name()) != entries_.end()) {
    return absl::FailedPreconditionError(
        "Attempting to register a cell library entry with a duplicate name: " +
        entry.name());
  }
  entries_.insert({entry.name(), absl::make_unique<CellLibraryEntry>(entry)});
  return absl::OkStatus();
}

xabsl::StatusOr<const CellLibraryEntry*> CellLibrary::GetEntry(
    absl::string_view name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Cell not found in library: ", name));
  }
  return it->second.get();
}

}  // namespace netlist
}  // namespace xls
