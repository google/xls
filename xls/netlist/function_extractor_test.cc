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

  OutputPinProto output_pin = entry.output_pins(0);
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

  OutputPinProto output_pin = entry.output_pins(0);
  ASSERT_EQ(output_pin.name(), "q");
  ASSERT_EQ(output_pin.function(), "i0|i1");
}

}  // namespace
}  // namespace function
}  // namespace netlist
}  // namespace xls
