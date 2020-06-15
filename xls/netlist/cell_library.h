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
