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

#include "google/protobuf/text_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/netlist.pb.h"

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

TEST(CellLibraryTest, EvaluateStateTable) {
  std::string proto_text = R"(input_names: "i0"
  input_names: "i1"
  internal_names: "BLT"
  internal_names: "ham_sandwich"
  rows {
    input_signals {
      key: "i0"
      value: STATE_TABLE_SIGNAL_LOW
    }
    input_signals {
      key: "i1"
      value: STATE_TABLE_SIGNAL_HIGH
    }
    internal_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_DONTCARE
    }
    internal_signals {
      key: "ham_sandwich"
      value: STATE_TABLE_SIGNAL_DONTCARE
    }
    output_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_LOW
    }
    output_signals {
      key: "ham_sandwich"
      value: STATE_TABLE_SIGNAL_HIGH
    }
  }
  rows {
    input_signals {
      key: "i0"
      value: STATE_TABLE_SIGNAL_DONTCARE
    }
    input_signals {
      key: "i1"
      value: STATE_TABLE_SIGNAL_LOW
    }
    internal_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_HIGH
    }
    internal_signals {
      key: "ham_sandwich"
      value: STATE_TABLE_SIGNAL_DONTCARE
    }
    output_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_HIGH
    }
    output_signals {
      key: "ham_sandwich"
      value: STATE_TABLE_SIGNAL_LOW
    }
  }
  )";
  StateTableProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_text, &proto));

  XLS_ASSERT_OK_AND_ASSIGN(StateTable table, StateTable::FromProto(proto));
  StateTable::InputStimulus stimulus;
  stimulus["i0"] = false;
  stimulus["i1"] = true;
  XLS_ASSERT_OK_AND_ASSIGN(bool signal,
                           table.GetSignalValue(stimulus, "ham_sandwich"));
  EXPECT_TRUE(signal);
  XLS_ASSERT_OK_AND_ASSIGN(signal, table.GetSignalValue(stimulus, "BLT"));
  EXPECT_FALSE(signal);

  stimulus.clear();
  stimulus["i1"] = false;
  stimulus["BLT"] = true;
  XLS_ASSERT_OK_AND_ASSIGN(signal,
                           table.GetSignalValue(stimulus, "ham_sandwich"));
  EXPECT_FALSE(signal);
  XLS_ASSERT_OK_AND_ASSIGN(signal, table.GetSignalValue(stimulus, "BLT"));
  EXPECT_TRUE(signal);

  // For the final case, expect an error - there's no row for when i1 is low and
  // BLT is unspecified.
  stimulus.clear();
  stimulus["i1"] = false;
  EXPECT_THAT(
      table.GetSignalValue(stimulus, "ham_sandwich"),
      status_testing::StatusIs(absl::StatusCode::kNotFound,
                               ::testing::HasSubstr("No matching row")));
}

}  // namespace
}  // namespace netlist
}  // namespace xls
