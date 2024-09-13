// Copyright 2023 The XLS Authors
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

#include "xls/fdo/synthesizer.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"

namespace xls {
namespace {

constexpr std::string_view kTestdataPath = "xls/fdo/testdata";

class FakeSynthesizer : public synthesis::Synthesizer {
 public:
  FakeSynthesizer() : synthesis::Synthesizer("FakeSynthesizer") {}

  absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const override {
    return 0;
  }
};

class SynthesizerTest : public IrTestBase {
 public:
  std::filesystem::path GoldenFilePath(std::string_view file_ext) {
    return absl::StrFormat("%s/synthesizer_test_%s.%s", kTestdataPath,
                           TestName(), file_ext);
  }

  FakeSynthesizer synthesizer_;
};

TEST_F(SynthesizerTest, FunctionBaseToVerilogSimple) {
  const std::string ir_text = R"(
package p

fn test(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret add.3: bits[3] = add(i0, i1, id=3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), actual_verilog_text);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_ff_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/true));
  ExpectEqualToGoldenFile(GoldenFilePath("ff.vtxt"), actual_ff_verilog_text);
}

TEST_F(SynthesizerTest, FunctionBaseToVerilogWithLiveIn) {
  std::string ir_text = R"(
package p

fn test(add_1: bits[3], i1: bits[3]) -> bits[3] {
  ret sub.3: bits[3] = sub(add_1, i1, id=3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), actual_verilog_text);
}

TEST_F(SynthesizerTest, FunctionBaseToVerilogWithLiveOut) {
  std::string ir_text = R"(
package p

fn test(i0: bits[3], i1: bits[3]) -> (bits[3], bits[3]) {
  add.3: bits[3] = add(i0, i1, id=3)
  sub.4: bits[3] = sub(add.3, i1, id=4)
  ret tuple.5: (bits[3], bits[3]) = tuple(add.3, sub.4, id=5)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  ExpectEqualToGoldenFile(GoldenFilePath("vtxt"), actual_verilog_text);
}

}  // namespace
}  // namespace xls
