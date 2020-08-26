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

#include "xls/netlist/function_extractor.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {
namespace function {
namespace {

TEST(FunctionExtractorTest, BasicFunctionality) {
  std::string lib = R"(
library (blah) {
  blah (blah) {
  }
  blah: "blahblahblah";
  cell (cell_1) {
    pin (i0) {
      direction: input;
    }
    pin (i1) {
      direction: input;
    }
    pin (o) {
      direction: output;
      function: "meow";
    }
  }
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto stream, cell_lib::CharStream::FromText(lib));
  XLS_ASSERT_OK_AND_ASSIGN(CellLibraryProto proto, ExtractFunctions(&stream));

  const CellLibraryEntryProto entry = proto.entries(0);
  EXPECT_EQ(entry.name(), "cell_1");
  absl::flat_hash_set<std::string> input_names({"i0", "i1"});
  for (const auto& input_name : entry.input_names()) {
    ASSERT_TRUE(input_names.contains(input_name));
    input_names.erase(input_name);
  }
  ASSERT_EQ(input_names.size(), 0);

  OutputPinProto output_pin = entry.output_pin_list().pins(0);
  ASSERT_EQ(output_pin.name(), "o");
  ASSERT_EQ(output_pin.function(), "meow");
}

TEST(FunctionExtractorTest, HippetyHoppetyTestTheFlippetyFloppety) {
  std::string lib = R"(
library (blah) {
  blah (blah) {
  }
  cell (cell_1) {
    pin (i0) {
      direction: input;
    }
    pin (i1) {
      direction: input;
    }
    pin (q) {
      direction: output;
      function: "bleh";
    }
    ff (whatever) {
      next_state: "i0|i1";
    }
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto stream, cell_lib::CharStream::FromText(lib));
  XLS_ASSERT_OK_AND_ASSIGN(CellLibraryProto proto, ExtractFunctions(&stream));

  const CellLibraryEntryProto entry = proto.entries(0);
  EXPECT_EQ(entry.name(), "cell_1");
  absl::flat_hash_set<std::string> input_names({"i0", "i1"});
  for (const auto& input_name : entry.input_names()) {
    ASSERT_TRUE(input_names.contains(input_name));
    input_names.erase(input_name);
  }
  ASSERT_EQ(input_names.size(), 0);

  OutputPinProto output_pin = entry.output_pin_list().pins(0);
  ASSERT_EQ(output_pin.name(), "q");
  ASSERT_EQ(output_pin.function(), "i0|i1");
}

TEST(FunctionExtractorTest, HandlesStatetables) {
  std::string lib = R"(
library (blah) {
  blah (blah) {
  }
  blah: "blahblahblah";
  cell (cell_1) {
    pin (i0) {
      direction: input;
    }
    pin (i1) {
      direction: input;
    }
    pin (o) {
      direction: output;
      function: "meow";
    }
    statetable ("i0 i1", "ham_sandwich") {
     table: "L L : - : H, \
             L H : T : X, \
             H L : T : N, \
             H H : - : H ";
    }
    pin (ham_sandwich) {
      direction: internal;
      internal_node: "ham_sandwich";
    }
  }
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(auto stream, cell_lib::CharStream::FromText(lib));
  XLS_ASSERT_OK_AND_ASSIGN(CellLibraryProto proto, ExtractFunctions(&stream));

  const CellLibraryEntryProto entry = proto.entries(0);
  EXPECT_EQ(entry.name(), "cell_1");
  ASSERT_TRUE(entry.has_state_table());
  StateTableProto table = entry.state_table();
  absl::flat_hash_set<std::string> input_names({"i0", "i1"});
  for (const auto& input_name : table.input_names()) {
    ASSERT_TRUE(input_names.contains(input_name));
    input_names.erase(input_name);
  }
  ASSERT_EQ(table.internal_names(0), "ham_sandwich");

  // Just validate a single row instead of going through all of them.
  StateTableRow row = table.rows(2);
  ASSERT_EQ(row.input_signals_size(), 2);
  EXPECT_EQ(row.input_signals().at("i0"), STATE_TABLE_SIGNAL_HIGH);
  EXPECT_EQ(row.input_signals().at("i1"), STATE_TABLE_SIGNAL_LOW);

  ASSERT_EQ(row.internal_signals_size(), 1);
  EXPECT_EQ(row.internal_signals().at("ham_sandwich"),
            STATE_TABLE_SIGNAL_TOGGLE);

  ASSERT_EQ(row.output_signals_size(), 1);
  EXPECT_EQ(row.output_signals().at("ham_sandwich"),
            STATE_TABLE_SIGNAL_NOCHANGE);
}

}  // namespace
}  // namespace function
}  // namespace netlist
}  // namespace xls
