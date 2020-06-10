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

#include "xls/netlist/fake_cell_library.h"

#include "xls/common/logging/logging.h"

namespace xls {
namespace netlist {

CellLibrary MakeFakeCellLibrary() {
  CellLibrary cell_library;
  OutputPin pin;
  pin.name = "ZN";
  pin.function = "!A";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kInverter, "INV", std::vector<std::string>{"A"},
      std::vector<OutputPin>{pin})));
  pin.name = "Q";
  pin.function = "D";
  XLS_CHECK_OK(cell_library.AddEntry(
      CellLibraryEntry(CellKind::kFlop, "DFF", std::vector<std::string>{"D"},
                       std::vector<OutputPin>{pin}, "CLK")));
  pin.name = "Z";
  pin.function = "A&B";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kOther, "AND", std::vector<std::string>{"A", "B"},
      std::vector<OutputPin>{pin})));
  pin.name = "Z";
  pin.function = "A|B";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kOther, "OR", std::vector<std::string>{"A", "B"},
      std::vector<OutputPin>{pin})));
  pin.name = "Z";
  pin.function = "A^B";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kOther, "XOR", std::vector<std::string>{"A", "B"},
      std::vector<OutputPin>{pin})));
  pin.name = "ZN";
  pin.function = "!(A&B)";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kNand, "NAND", std::vector<std::string>{"A", "B"},
      std::vector<OutputPin>{pin})));
  pin.name = "ZN";
  pin.function = "!(A|B|C|D)";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kNor, "NOR4", std::vector<std::string>{"A", "B", "C", "D"},
      std::vector<OutputPin>{pin})));
  // A la https://en.wikipedia.org/wiki/AND-OR-Invert
  pin.name = "ZN";
  pin.function = "!((A*B)|C)";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kOther, "AOI21", std::vector<std::string>{"A", "B", "C"},
      std::vector<OutputPin>{pin})));
  // A fixed output-one cell.
  pin.name = "O";
  pin.function = "1";
  XLS_CHECK_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kOther, "LOGIC_ONE", std::vector<std::string>{},
      std::vector<OutputPin>{pin})));
  return cell_library;
}

}  // namespace netlist
}  // namespace xls
