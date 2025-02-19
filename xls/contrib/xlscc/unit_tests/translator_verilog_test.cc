// Copyright 2021 The XLS Authors
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

#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/signature_generator.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xlscc {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

constexpr char kTestName[] = "translator_verilog_test";
constexpr char kTestdataPath[] = "xls/contrib/xlscc/unit_tests/testdata";

class TranslatorVerilogTest : public xls::verilog::VerilogTestBase {};

absl::Status SimplifyAndInline(xls::Package* package) {
  std::unique_ptr<xls::OptimizationCompoundPass> pipeline =
      xls::CreateOptimizationPassPipeline();
  xls::OptimizationPassOptions options;
  xls::PassResults results;
  xls::OptimizationContext context;

  // This pass wants a delay estimator
  options.skip_passes = {"bdd_cse"};

  return pipeline->Run(package, options, &results, &context).status();
}

// What's being tested here is that the IR produced is generatable by the
// combinational generator (after simplification). Simulation tests already
// occur in the combinational_generator_test.
TEST_P(TranslatorVerilogTest, IOProcComboGenOneToNMux) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(int& dir,
              __xls_channel<int>& in,
              __xls_channel<int>& out1,
              __xls_channel<int> &out2) {


      const int ctrl = in.read();

      if (dir == 0) {
        out1.write(ctrl);
      } else {
        out2.write(ctrl);
      }
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(CHANNEL_TYPE_DIRECT_IN);

    HLSChannel* ch_in = block_spec.add_channels();
    ch_in->set_name("in");
    ch_in->set_is_input(true);
    ch_in->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out1");
    ch_out1->set_is_input(false);
    ch_out1->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out2 = block_spec.add_channels();
    ch_out2->set_name("out2");
    ch_out2->set_is_input(false);
    ch_out2->set_type(CHANNEL_TYPE_FIFO);
  }

  auto parser = std::make_unique<xlscc::CCParser>();
  XLS_ASSERT_OK(
      XlsccTestBase::ScanTempFileWithContent(content, {}, parser.get()));

  auto translator = std::make_unique<xlscc::Translator>(
      /*error_on_init_interval=*/false,
      /*error_on_uninitialized=*/false,
      /*generate_fsms_for_pipelined_loops=*/false,
      /*merge_states=*/false, /*split_states_on_channel_ops=*/false,
      /*debug_ir_trace_flags=*/xlscc::DebugIrTraceFlags_None,
      /*warn_unroll_iters=*/100,
      /*max_unroll_iters=*/100,
      /*z3_rlimit=*/-1,
      /*op_ordering=*/xlscc::IOOpOrdering::kNone,
      /*existing_parser=*/std::move(parser));

  xls::Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc,
                           translator->GenerateIR_Block(&package, block_spec));

  VLOG(1) << "Simplifying IR..." << std::endl;
  XLS_ASSERT_OK(SimplifyAndInline(&package));

  XLS_ASSERT_OK_AND_ASSIGN(
      xls::verilog::CodegenPassUnit unit,
      xls::verilog::ProcToCombinationalBlock(proc, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog,
      xls::verilog::GenerateVerilog(unit.top_block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::verilog::ModuleSignature signature,
      xls::verilog::GenerateSignature(codegen_options(), unit.top_block));

  VLOG(1) << package.DumpIr() << std::endl;

  VLOG(1) << verilog << std::endl;

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  xls::verilog::ModuleSimulator simulator(
      signature, verilog, xls::verilog::FileType::kVerilog, GetSimulator());
  using BitsMap = xls::verilog::ModuleSimulator::BitsMap;
  // Output out1 selected, input valid and output ready asserted.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(0, 32)},
                                    {"in", xls::UBits(123, 32)},
                                    {"in_vld", xls::UBits(1, 1)},
                                    {"out1_rdy", xls::UBits(1, 1)},
                                    {"out2_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out1", xls::UBits(123, 32)), Pair("out1_vld", xls::UBits(1, 1)),
          Pair("out2", xls::UBits(123, 32)), Pair("out2_vld", xls::UBits(0, 1)),
          Pair("in_rdy", xls::UBits(1, 1)))));

  // Output out2 selected, input valid and output ready asserted.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in", xls::UBits(123, 32)},
                                    {"in_vld", xls::UBits(1, 1)},
                                    {"out1_rdy", xls::UBits(1, 1)},
                                    {"out2_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out1", xls::UBits(123, 32)), Pair("out1_vld", xls::UBits(0, 1)),
          Pair("out2", xls::UBits(123, 32)), Pair("out2_vld", xls::UBits(1, 1)),
          Pair("in_rdy", xls::UBits(1, 1)))));

  // Output out2 selected, input valid asserted, and output ready *not*
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in", xls::UBits(123, 32)},
                                    {"in_vld", xls::UBits(1, 1)},
                                    {"out1_rdy", xls::UBits(1, 1)},
                                    {"out2_rdy", xls::UBits(0, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out1", xls::UBits(123, 32)), Pair("out1_vld", xls::UBits(0, 1)),
          Pair("out2", xls::UBits(123, 32)), Pair("out2_vld", xls::UBits(1, 1)),
          Pair("in_rdy", xls::UBits(0, 1)))));

  // Output out2 selected, input valid *not* asserted, and output ready
  // asserted. Output valid should be zero.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in", xls::UBits(123, 32)},
                                    {"in_vld", xls::UBits(0, 1)},
                                    {"out1_rdy", xls::UBits(1, 1)},
                                    {"out2_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out1", xls::UBits(123, 32)), Pair("out1_vld", xls::UBits(0, 1)),
          Pair("out2", xls::UBits(123, 32)), Pair("out2_vld", xls::UBits(0, 1)),
          Pair("in_rdy", xls::UBits(1, 1)))));
}

TEST_P(TranslatorVerilogTest, IOProcComboGenNToOneMux) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(int& dir,
              __xls_channel<int>& in1,
              __xls_channel<int>& in2,
              __xls_channel<int>& out) {


      int x;

      if (dir == 0) {
        x = in1.read();
      } else {
        x = in2.read();
      }

      out.write(x);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* dir_in = block_spec.add_channels();
    dir_in->set_name("dir");
    dir_in->set_is_input(true);
    dir_in->set_type(CHANNEL_TYPE_DIRECT_IN);

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in1");
    ch_in1->set_is_input(true);
    ch_in1->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_in2 = block_spec.add_channels();
    ch_in2->set_name("in2");
    ch_in2->set_is_input(true);
    ch_in2->set_type(CHANNEL_TYPE_FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(CHANNEL_TYPE_FIFO);
  }

  auto parser = std::make_unique<xlscc::CCParser>();
  XLS_ASSERT_OK(
      XlsccTestBase::ScanTempFileWithContent(content, {}, parser.get()));

  std::unique_ptr<xlscc::Translator> translator(new xlscc::Translator(
      /*error_on_init_interval=*/false,
      /*error_on_uninitialized=*/false,
      /*generate_fsms_for_pipelined_loops=*/false,
      /*merge_states=*/false, /*split_states_on_channel_ops=*/false,
      /*debug_ir_trace_flags=*/xlscc::DebugIrTraceFlags_None,
      /*max_unroll_iters=*/100,
      /*warn_unroll_iters=*/100,
      /*z3_rlimit=*/-1,
      /*op_ordering=*/xlscc::IOOpOrdering::kNone,
      /*existing_parser=*/std::move(parser)));

  xls::Package package("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc,
                           translator->GenerateIR_Block(&package, block_spec));

  VLOG(1) << "Simplifying IR..." << std::endl;
  XLS_ASSERT_OK(SimplifyAndInline(&package));

  XLS_ASSERT_OK_AND_ASSIGN(
      xls::verilog::CodegenPassUnit unit,
      xls::verilog::ProcToCombinationalBlock(proc, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::verilog::ModuleSignature signature,
      xls::verilog::GenerateSignature(codegen_options(), unit.top_block));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog,
      xls::verilog::GenerateVerilog(unit.top_block, codegen_options()));

  VLOG(1) << package.DumpIr() << std::endl;

  VLOG(1) << verilog << std::endl;

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  xls::verilog::ModuleSimulator simulator(
      signature, verilog, xls::verilog::FileType::kVerilog, GetSimulator());
  using BitsMap = xls::verilog::ModuleSimulator::BitsMap;
  // Input in1 selected, input valid and output ready asserted.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(0, 32)},
                                    {"in1", xls::UBits(123, 32)},
                                    {"in2", xls::UBits(42, 32)},
                                    {"in1_vld", xls::UBits(1, 1)},
                                    {"in2_vld", xls::UBits(1, 1)},
                                    {"out_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(Pair("out_vld", xls::UBits(1, 1)),
                                        Pair("in1_rdy", xls::UBits(1, 1)),
                                        Pair("out", xls::UBits(123, 32)),
                                        Pair("in2_rdy", xls::UBits(0, 1)))));

  // Input in2 selected, input valid and output ready asserted.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in1", xls::UBits(123, 32)},
                                    {"in2", xls::UBits(42, 32)},
                                    {"in1_vld", xls::UBits(0, 1)},
                                    {"in2_vld", xls::UBits(1, 1)},
                                    {"out_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out_vld", xls::UBits(1, 1)), Pair("in1_rdy", xls::UBits(0, 1)),
          Pair("out", xls::UBits(42, 32)), Pair("in2_rdy", xls::UBits(1, 1)))));

  // Input in2 selected, input valid asserted, and output ready *not*
  // asserted. Input ready should be zero.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in1", xls::UBits(123, 32)},
                                    {"in2", xls::UBits(42, 32)},
                                    {"in1_vld", xls::UBits(1, 1)},
                                    {"in2_vld", xls::UBits(1, 1)},
                                    {"out_rdy", xls::UBits(0, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out_vld", xls::UBits(1, 1)), Pair("in1_rdy", xls::UBits(0, 1)),
          Pair("out", xls::UBits(42, 32)), Pair("in2_rdy", xls::UBits(0, 1)))));

  // Input in2 selected, input valid *not* asserted, and output ready
  // asserted. Output valid should be zero.
  EXPECT_THAT(
      simulator.RunFunction(BitsMap{{"dir", xls::UBits(1, 32)},
                                    {"in1", xls::UBits(123, 32)},
                                    {"in2", xls::UBits(42, 32)},
                                    {"in1_vld", xls::UBits(1, 1)},
                                    {"in2_vld", xls::UBits(0, 1)},
                                    {"out_rdy", xls::UBits(1, 1)}}),
      IsOkAndHolds(UnorderedElementsAre(
          Pair("out_vld", xls::UBits(0, 1)), Pair("in1_rdy", xls::UBits(0, 1)),
          Pair("out", xls::UBits(42, 32)), Pair("in2_rdy", xls::UBits(1, 1)))));
}

INSTANTIATE_TEST_SUITE_P(
    TranslatorVerilogTestInstantiation, TranslatorVerilogTest,
    testing::ValuesIn(xls::verilog::kDefaultSimulationTargets),
    xls::verilog::ParameterizedTestName<TranslatorVerilogTest>);

}  // namespace
}  // namespace xlscc
