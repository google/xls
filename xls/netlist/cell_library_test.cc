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

#include "xls/netlist/cell_library.h"

#include <memory>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {
namespace {

using google::protobuf::TextFormat;
using status_testing::IsOkAndHolds;

TEST(CellLibraryTest, SerializeToProto) {
  CellLibrary cell_library;
  CellLibraryEntry::OutputPinToFunction pins;
  pins["Z"] = "W";
  XLS_ASSERT_OK(cell_library.AddEntry(
      CellLibraryEntry(CellKind::kInverter, "INV",
                       std::vector<std::string>{"A"}, pins, std::nullopt)));
  XLS_ASSERT_OK_AND_ASSIGN(CellLibraryProto proto, cell_library.ToProto());
  std::string expected_proto_text = R"(entries {
  kind: INVERTER
  name: "INV"
  input_names: "A"
  output_pin_list {
    pins {
      name: "Z"
      function: "W"
    }
  }
}
)";
  CellLibraryProto expected_proto;
  TextFormat::ParseFromString(expected_proto_text, &expected_proto);
  EXPECT_TRUE(google::protobuf::util::MessageDifferencer::Equals(proto, expected_proto));
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
    next_internal_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_LOW
    }
    next_internal_signals {
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
    next_internal_signals {
      key: "BLT"
      value: STATE_TABLE_SIGNAL_HIGH
    }
    next_internal_signals {
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

  EXPECT_THAT(table.GetSignalValue(stimulus, "PB&J"),
              status_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CellLibraryTest, LutStateTable) {
  // 4-way AND
  StateTable table = StateTable::FromLutMask(0x8000);
  StateTable::InputStimulus stimulus;
  stimulus["I0"] = true;
  stimulus["I1"] = true;
  stimulus["I2"] = true;
  stimulus["I3"] = true;
  EXPECT_THAT(table.GetSignalValue(stimulus, "X"), IsOkAndHolds(true));

  stimulus.clear();
  stimulus["I0"] = true;
  stimulus["I1"] = false;
  stimulus["I2"] = true;
  stimulus["I3"] = true;
  EXPECT_THAT(table.GetSignalValue(stimulus, "X"), IsOkAndHolds(false));
}

}  // namespace
}  // namespace netlist
}  // namespace xls
