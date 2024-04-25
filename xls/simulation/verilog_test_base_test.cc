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

#include "xls/simulation/verilog_test_base.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;

constexpr char kTestName[] = "verilog_test_base_test";
constexpr char kTestdataPath[] = "xls/simulation/testdata";

class VerilogTestBaseTest : public VerilogTestBase {};

TEST_P(VerilogTestBaseTest, ValidVerilog) {
  std::string text = R"(module test_module(
  input [1:0] x,
  input [1:0] y,
  output [1:0] z
);
  assign z = x + y;
endmodule
)";
  XLS_ASSERT_OK(ValidateVerilog(text));
}

TEST_P(VerilogTestBaseTest, InvalidVerilog) {
  std::string text = R"(
  10 PRINT "HELLO WORLD"
  20 GOTO 10
)";
  EXPECT_THAT(ValidateVerilog(text), StatusIs(absl::StatusCode::kInternal));
}

TEST_P(VerilogTestBaseTest, ValidVerilogEqual) {
  std::string text = R"(module test_module(
  input [1:0] x,
  input [1:0] y,
  output [1:0] z
);
  assign z = x + y;
endmodule
)";
  ExpectVerilogEqual(text, text);
}

TEST_P(VerilogTestBaseTest, ValidVerilogNotEqual) {
  constexpr char kTextFmt[] = R"(module %s(
  input [1:0] x,
  input [1:0] y,
  output [1:0] z
);
  assign z = x + y;
endmodule
)";
  std::string expected = absl::StrFormat(kTextFmt, "expected_module");
  std::string actual = absl::StrFormat(kTextFmt, "actual_module");
  EXPECT_NONFATAL_FAILURE(ExpectVerilogEqual(expected, actual),
                          "Expected equality of these values");
}

TEST_P(VerilogTestBaseTest, ExpectEqualtoGolden) {
  std::string text = R"(module test_module(
  input [1:0] x,
  input [1:0] y,
  output [1:0] z
);
  assign z = x + y;
endmodule
)";
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 text);
}

TEST_P(VerilogTestBaseTest, InvalidVerilogEqual) {
  EXPECT_NONFATAL_FAILURE(ExpectVerilogEqual("foo", "foo"), "");
}

INSTANTIATE_TEST_SUITE_P(VerilogTestBaseTestInstantiation, VerilogTestBaseTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<VerilogTestBaseTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
