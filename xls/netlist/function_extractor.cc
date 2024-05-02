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

#include "xls/netlist/function_extractor.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {
namespace function {
namespace {

constexpr const char kDirectionKey[] = "direction";
constexpr const char kFunctionKey[] = "function";
constexpr const char kNextStateKey[] = "next_state";
constexpr const char kStateFunctionKey[] = "state_function";
constexpr const char kInputValue[] = "input";
constexpr const char kOutputValue[] = "output";
constexpr const char kPinKind[] = "pin";
constexpr const char kFfKind[] = "ff";
constexpr const char kStateTableKind[] = "statetable";

// Translates an individual signal value char to the protobuf equivalent.
absl::StatusOr<StateTableSignalProto> LibertyToTableSignal(
    const std::string& input) {
  if (input == "H/L") {
    return STATE_TABLE_SIGNAL_HIGH_OR_LOW;
  }
  if (input == "L/H") {
    return STATE_TABLE_SIGNAL_LOW_OR_HIGH;
  }
  if (input == "~R") {
    return STATE_TABLE_SIGNAL_NOT_RISING;
  }
  if (input == "~F") {
    return STATE_TABLE_SIGNAL_NOT_FALLING;
  }
  XLS_RET_CHECK(input.size() == 1) << input;

  char signal = input[0];
  XLS_RET_CHECK(signal == '-' || signal == 'H' || signal == 'L' ||
                signal == 'N' || signal == 'R' || signal == 'F' ||
                signal == 'X');
  switch (signal) {
    case '-':
      return STATE_TABLE_SIGNAL_DONTCARE;
    case 'H':
      return STATE_TABLE_SIGNAL_HIGH;
    case 'L':
      return STATE_TABLE_SIGNAL_LOW;
    case 'N':
      return STATE_TABLE_SIGNAL_NOCHANGE;
    case 'R':
      return STATE_TABLE_SIGNAL_RISING;
    case 'F':
      return STATE_TABLE_SIGNAL_FALLING;
    case 'X':
      return STATE_TABLE_SIGNAL_X;
    default:
      LOG(FATAL) << "Invalid input signal: " << signal;
  }
}

std::string_view SanitizeRow(const std::string_view& row) {
  // Remove newlines, whitespace, and occasional extraneous slash characters
  // from a row in the state table: there are rows that appear as, for example:
  //  X X X : Y : Z  ,\
  // It makes life a lot easier if we can just deal with the range from X to Z.
  std::string_view result = absl::StripAsciiWhitespace(row);
  if (result[0] == '\\') {
    result.remove_prefix(1);
  }
  result = absl::StripAsciiWhitespace(result);
  return result;
}

// Parses a textual Liberty statetable entry to a proto.
// This doesn't attempt to _validate_ the specified state table; it only
// proto-izes it.
// Returns a map of internal signal to state table defining its behavior.
absl::Status ProcessStateTable(const cell_lib::Block& table_def,
                               CellLibraryEntryProto* proto) {
  StateTableProto* table = proto->mutable_state_table();
  for (std::string_view name :
       absl::StrSplit(table_def.args[0], ' ', absl::SkipWhitespace())) {
    table->add_input_names(std::string(name));
  }

  for (std::string_view name :
       absl::StrSplit(table_def.args[1], ' ', absl::SkipWhitespace())) {
    table->add_internal_names(std::string(name));
  }

  XLS_RET_CHECK(table_def.entries.size() == 1);
  XLS_RET_CHECK(
      std::holds_alternative<cell_lib::KVEntry>(table_def.entries[0]));
  const cell_lib::KVEntry& table_entry =
      std::get<cell_lib::KVEntry>(table_def.entries[0]);

  // Table entries are, sadly, strings, such as:
  // " L H L : - : H,
  //   H L H : - : L,
  //   ... "
  // So we gotta comma separate then colon separate them to get the important
  // bits. The first column is the input signals, ordered as in the first arg.
  std::vector<std::string> rows =
      absl::StrSplit(table_entry.value, ',', absl::SkipWhitespace());
  for (const std::string& raw_row : rows) {
    std::string_view source_row = SanitizeRow(raw_row);
    std::vector<std::string> fields =
        absl::StrSplit(source_row, ':', absl::SkipWhitespace());
    XLS_RET_CHECK(fields.size() == 3)
        << "Improperly formatted row: " << source_row;
    std::vector<std::string> inputs =
        absl::StrSplit(fields[0], ' ', absl::SkipWhitespace());
    std::vector<std::string> internal_inputs =
        absl::StrSplit(fields[1], ' ', absl::SkipWhitespace());
    std::vector<std::string> internal_outputs =
        absl::StrSplit(fields[2], ' ', absl::SkipWhitespace());

    StateTableRow* row = table->add_rows();
    for (int i = 0; i < inputs.size(); i++) {
      XLS_ASSIGN_OR_RETURN(StateTableSignalProto signal,
                           LibertyToTableSignal(inputs[i]));
      (*row->mutable_input_signals())[table->input_names()[i]] = signal;
    }

    for (int i = 0; i < internal_inputs.size(); i++) {
      XLS_ASSIGN_OR_RETURN(StateTableSignalProto signal,
                           LibertyToTableSignal(internal_inputs[i]));
      (*row->mutable_internal_signals())[table->internal_names()[i]] = signal;
    }

    for (int i = 0; i < internal_inputs.size(); i++) {
      XLS_ASSIGN_OR_RETURN(StateTableSignalProto signal,
                           LibertyToTableSignal(internal_outputs[i]));
      (*row->mutable_next_internal_signals())[table->internal_names()[i]] =
          signal;
    }
  }

  return absl::OkStatus();
}

// Gets input and output pin names and output pin functions and adds them to the
// entry proto.
absl::Status ExtractFromPin(const cell_lib::Block& pin,
                            CellLibraryEntryProto* entry_proto) {
  // I've yet to see an example where this isn't the case.
  std::string name = pin.args[0];

  std::optional<bool> is_output;
  std::string function;
  for (const cell_lib::BlockEntry& entry : pin.entries) {
    const auto* kv_entry = absl::get_if<cell_lib::KVEntry>(&entry);
    if (kv_entry) {
      if (kv_entry->key == kDirectionKey) {
        if (kv_entry->value == kOutputValue) {
          is_output = true;
        } else if (kv_entry->value == kInputValue) {
          is_output = false;
        } else {
          // "internal" is at least one add'l direction.
          // We don't care about such pins.
          return absl::OkStatus();
        }
      } else if (kv_entry->key == kFunctionKey ||
                 kv_entry->key == kStateFunctionKey) {
        // "function" and "state_function" seem to be handlable in the same way:
        // state_function can deal with special cases that function can't (such
        // as use of internal ports)...but the internal logic appears to be the
        // same.
        function = kv_entry->value;
      }
    }
  }

  if (is_output == std::nullopt) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Pin %s has no direction entry!", name));
  }

  if (is_output.value()) {
    OutputPinListProto* output_list = entry_proto->mutable_output_pin_list();
    OutputPinProto* output_pin = output_list->add_pins();
    output_pin->set_name(name);
    if (!function.empty()) {
      // Some output pins lack associated functions.
      // Ignore them for now (during baseline dev). If they turn out to be
      // important, I'll circle back.
      output_pin->set_function(function);
    }
  } else {
    entry_proto->add_input_names(name);
  }

  return absl::OkStatus();
}

// If we see a ff (flop-flop) block, it contains a "next_state" field, with the
// function calculating the output value of the cell after the next clock.
// Since all our current (logic-checking) purposes are clockless, this is
// equivalent to being the "function" of an output pin. All known FF cells have
// a single output pin, so we check for that.
// If so, then we replace that function with the one from the next_state field.
absl::Status ExtractFromFf(const cell_lib::Block& ff,
                           CellLibraryEntryProto* entry_proto) {
  entry_proto->set_kind(netlist::CellKindProto::FLOP);
  for (const cell_lib::BlockEntry& entry : ff.entries) {
    const auto* kv_entry = absl::get_if<cell_lib::KVEntry>(&entry);
    if (kv_entry && kv_entry->key == kNextStateKey) {
      std::string next_state_function = kv_entry->value;
      XLS_RET_CHECK(entry_proto->output_pin_list().pins_size() == 1);
      entry_proto->mutable_output_pin_list()->mutable_pins(0)->set_function(
          next_state_function);
    }
  }

  return absl::OkStatus();
}

absl::Status ExtractFromCell(const cell_lib::Block& cell,
                             CellLibraryEntryProto* entry_proto) {
  // I've yet to see an example where this isn't the case.
  entry_proto->set_name(cell.args[0]);

  // Default kind; overridden only if necessary.
  entry_proto->set_kind(netlist::CellKindProto::OTHER);

  for (const cell_lib::BlockEntry& entry : cell.entries) {
    if (std::holds_alternative<std::unique_ptr<cell_lib::Block>>(entry)) {
      auto& block_entry = std::get<std::unique_ptr<cell_lib::Block>>(entry);
      if (block_entry->kind == kPinKind) {
        XLS_RETURN_IF_ERROR(ExtractFromPin(*block_entry.get(), entry_proto));
      } else if (block_entry->kind == kFfKind) {
        // If it's a flip-flop, we need to replace the pin's output function
        // with its next_state function.
        XLS_RETURN_IF_ERROR(ExtractFromFf(*block_entry.get(), entry_proto));
      } else if (block_entry->kind == kStateTableKind) {
        XLS_RETURN_IF_ERROR(ProcessStateTable(*block_entry.get(), entry_proto));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CellLibraryProto> ExtractFunctions(
    cell_lib::CharStream* stream) {
  cell_lib::Scanner scanner(stream);
  absl::flat_hash_set<std::string> kind_allowlist(
      {"library", "cell", "pin", "direction", "function", "ff", "next_state",
       "statetable"});
  cell_lib::Parser parser(&scanner);

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<cell_lib::Block> block,
                       parser.ParseLibrary());
  CellLibraryProto proto;
  for (const cell_lib::BlockEntry& entry : block->entries) {
    if (std::holds_alternative<std::unique_ptr<cell_lib::Block>>(entry)) {
      auto& block_entry = std::get<std::unique_ptr<cell_lib::Block>>(entry);
      if (block_entry->kind == "cell") {
        CellLibraryEntryProto* entry_proto = proto.add_entries();
        XLS_RETURN_IF_ERROR(ExtractFromCell(*block_entry.get(), entry_proto));
      }
    }
  }

  return proto;
}

}  // namespace function
}  // namespace netlist
}  // namespace xls
