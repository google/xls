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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace netlist {
namespace {

TEST(CellLibraryTest, SerializeToProto) {
  CellLibrary cell_library;
  OutputPin pin;
  pin.name = "Z";
  pin.function = "W";
  XLS_ASSERT_OK(cell_library.AddEntry(CellLibraryEntry(
      CellKind::kInverter, "INV", std::vector<std::string>{"A"},
      std::vector<OutputPin>{pin})));
  CellLibraryProto proto = cell_library.ToProto();
  EXPECT_EQ(R"(entries {
  kind: INVERTER
  name: "INV"
  input_names: "A"
  output_pins {
    name: "Z"
    function: "W"
  }
}
)",
            proto.DebugString());
}

}  // namespace
}  // namespace netlist
}  // namespace xls
