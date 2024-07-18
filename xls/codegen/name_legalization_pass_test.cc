// Copyright 2024 The XLS Authors
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

#include "xls/codegen/name_legalization_pass.h"

#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls::verilog {
namespace {

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Not;
using ::xls::status_testing::IsOkAndHolds;
using ::xls::status_testing::StatusIs;
namespace m = ::xls::op_matchers;

using NameLegalizationPassIrTest = IrTestBase;
using NameLegalizationPassRtlTest = VerilogTestBase;

constexpr std::string_view kTestName = "name_legalization_pass_test";
constexpr std::string_view kTestdataPath = "xls/codegen/testdata";

absl::StatusOr<bool> RunLegalizationPass(Block* block,
                                         bool use_system_verilog) {
  CodegenPassResults results;
  CodegenPassUnit unit(block->package(), block);
  CodegenOptions codegen_options;
  codegen_options.use_system_verilog(use_system_verilog);
  verilog::CodegenPassOptions options;
  options.codegen_options = codegen_options;
  return NameLegalizationPass().Run(&unit, options, &results);
}

TEST_F(NameLegalizationPassIrTest, NoKeywordNamesCausesNoChange) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb(TestName(), &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              IsOkAndHolds(false));
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              IsOkAndHolds(false));
}

TEST_F(NameLegalizationPassIrTest, KeywordPortCausesError) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb(TestName(), &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("output", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Port `output` is a keyword")));
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Port `output` is a keyword")));
}

TEST_F(NameLegalizationPassIrTest,
       SystemVerilogKeywordPortCausesErrorForSystemVerilogOnly) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb(TestName(), &p);
  BValue a = bb.InputPort("a", u32);
  // unique0 is SystemVerilog-only.
  BValue b = bb.InputPort("unique0", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              IsOkAndHolds(false));
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Port `unique0` is a keyword")));
}

TEST_F(NameLegalizationPassIrTest, KeywordModuleNameCausesError) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb("output", &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Module name `output` is a keyword")));
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Module name `output` is a keyword")));
}

TEST_F(NameLegalizationPassIrTest,
       SystemVerilogKeywordModuleNameCausesErrorForSystemVerilogOnly) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  // unique0 is SystemVerilog-only.
  BlockBuilder bb("unique0", &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("unique0", u32);
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              IsOkAndHolds(false));
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Module name `unique0` is a keyword")));
}

TEST_F(NameLegalizationPassIrTest, InternalNodeWithKeywordNameIsRenamed) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb(TestName(), &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue sum = bb.Add(a, b, SourceInfo(), /*name=*/"output");
  bb.OutputPort("out", sum);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              IsOkAndHolds(true));
  EXPECT_THAT(block->nodes(), AllOf(Contains(m::Name("output__1")),
                                    Not(Contains(m::Name("output")))));

  // Already renamed, no change.
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              IsOkAndHolds(false));
}

TEST_F(NameLegalizationPassIrTest,
       InternalNodeWithSystemVerilogKeywordNameIsRenamed) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb(TestName(), &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue sum = bb.Add(a, b, SourceInfo(), /*name=*/"unique0");
  bb.OutputPort("out", sum);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/false),
              IsOkAndHolds(false));
  EXPECT_THAT(block->nodes(), AllOf(Not(Contains(m::Name("unique0__1"))),
                                    Contains(m::Name("unique0"))));

  // Already renamed, no change.
  EXPECT_THAT(RunLegalizationPass(block, /*use_system_verilog=*/true),
              IsOkAndHolds(true));
  EXPECT_THAT(block->nodes(), AllOf(Contains(m::Name("unique0__1")),
                                    Not(Contains(m::Name("unique0")))));
}

TEST_P(NameLegalizationPassRtlTest, InternalNodesRenamedInRTL) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb("test_module", &p);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  // SystemVerilog-only keyword.
  BValue sum1 = bb.Add(a, b, SourceInfo(), /*name=*/"unique0");
  // Verilog + SystemVerilog keyword.
  BValue sum2 = bb.Add(a, b, SourceInfo(), /*name=*/"output");
  bb.OutputPort("out", bb.Add(sum1, sum2));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  CodegenOptions options;
  options.use_system_verilog(UseSystemVerilog());
  EXPECT_THAT(
      RunLegalizationPass(block, /*use_system_verilog=*/UseSystemVerilog()),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(std::string rtl, GenerateVerilog(block, options));
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath), rtl);
}

TEST_P(NameLegalizationPassRtlTest, KeywordFunctionNameCausesError) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  FunctionBuilder fb("output", &p);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(a, b)));

  CodegenOptions options;
  options.use_system_verilog(UseSystemVerilog());
  EXPECT_THAT(GenerateCombinationalModule(f, options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Module name `output` is a keyword")));
}

MATCHER_P(RegisterWithName, name_matcher, "") {
  return ExplainMatchResult(name_matcher, arg->name(), result_listener);
}

TEST_P(NameLegalizationPassRtlTest, KeywordRegisterIsRenamed) {
  VerifiedPackage p(TestName());
  Type* u32 = p.GetBitsType(32);
  BlockBuilder bb("test_module", &p);
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  BValue a = bb.InputPort("a", u32);
  bb.OutputPort("b", bb.InsertRegister("buf", a));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetRegisters(), ElementsAre(RegisterWithName(Eq("buf"))));

  EXPECT_THAT(
      RunLegalizationPass(block, /*use_system_verilog=*/UseSystemVerilog()),
      IsOkAndHolds(true));

  EXPECT_THAT(block->GetRegisters(),
              ElementsAre(RegisterWithName(Eq("buf__1"))));
}

INSTANTIATE_TEST_SUITE_P(NameLegalizationPassRtlTestInstantiation,
                         NameLegalizationPassRtlTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<NameLegalizationPassRtlTest>);

}  // namespace
}  // namespace xls::verilog
