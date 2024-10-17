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

#include "xls/codegen/block_inlining_pass.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;
namespace xls::verilog {
namespace {
using Instantiation = xls::Instantiation;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

class BlockInliningPassTest : public IrTestBase {};

TEST_F(BlockInliningPassTest, InlineBlocks) {
  auto p = CreatePackage();

  // (define leaf (x y) (+ x y))
  // (define head (a b)
  //   (let ((left (leaf a 2))
  //         (right (leaf b 3)))
  //     (* left right)))
  BlockBuilder bb_leaf("leaf", p.get());
  bb_leaf.OutputPort("res",
                     bb_leaf.Add(bb_leaf.InputPort("x", p->GetBitsType(32)),
                                 bb_leaf.InputPort("y", p->GetBitsType(32))));
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, bb_leaf.Build());

  BlockBuilder bb_top(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * left,
                           bb_top.block()->AddBlockInstantiation("left", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * right,
      bb_top.block()->AddBlockInstantiation("right", leaf));
  bb_top.InstantiationInput(left, "x",
                            bb_top.InputPort("a", p->GetBitsType(32)));
  bb_top.InstantiationInput(left, "y", bb_top.Literal(UBits(2, 32)));
  bb_top.InstantiationInput(right, "x",
                            bb_top.InputPort("b", p->GetBitsType(32)));
  bb_top.InstantiationInput(right, "y", bb_top.Literal(UBits(3, 32)));
  bb_top.OutputPort("res",
                    bb_top.UMul(bb_top.InstantiationOutput(left, "res"),
                                bb_top.InstantiationOutput(right, "res")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb_top.Build());

  ScopedRecordIr sri(p.get());

  BlockInliningPass bip;
  CodegenPassUnit pu(p.get(), top);
  CodegenPassResults results;
  CodegenPassOptions opt;
  ASSERT_THAT(bip.Run(&pu, opt, &results), absl_testing::IsOkAndHolds(true));

  Block* inlined = pu.top_block;
  EXPECT_THAT(inlined->nodes(),
              testing::Not(testing::Contains(m::InstantiationInput())));
  EXPECT_THAT(inlined->nodes(),
              testing::Not(testing::Contains(m::InstantiationOutput())));
  EXPECT_THAT(results.register_renames, IsEmpty());

  InterpreterBlockEvaluator eval;
  XLS_ASSERT_OK_AND_ASSIGN(auto oracle, eval.NewContinuation(top));
  XLS_ASSERT_OK_AND_ASSIGN(auto test, eval.NewContinuation(inlined));
  std::vector<std::pair<int64_t, int64_t>> test_vector{
      {2, 3}, {1, 0}, {0, 3}, {4, 12}, {0, 0}, {0, 0},
  };
  int64_t i = 0;
  for (const auto& [a, b] : test_vector) {
    XLS_ASSERT_OK(test->RunOneCycle(
        {{"a", Value(UBits(a, 32))}, {"b", Value(UBits(b, 32))}}));
    XLS_ASSERT_OK(oracle->RunOneCycle(
        {{"a", Value(UBits(a, 32))}, {"b", Value(UBits(b, 32))}}));
    EXPECT_EQ(oracle->output_ports(), test->output_ports());
    RecordProperty(
        absl::StrFormat("test_out_%d", i),
        absl::StrFormat("(%d, %d) -> %s", a, b,
                        testing::PrintToString(test->output_ports())));
    ++i;
  }
}

TEST_F(BlockInliningPassTest, InlineBlocksWithReg) {
  auto p = CreatePackage();

  // (define leaf (x y)
  //   (cycles
  //      (do (reg_write x_reg x) (reg_write y_reg y)
  //      (+ (reg_read x_reg) (reg_read y_reg)))))
  // (define head (a b)
  //   (let ((left (leaf a 2))
  //         (right (leaf b 3)))
  //     (* left right)))
  BlockBuilder bb_leaf("leaf", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* x_reg, bb_leaf.block()->AddRegister("x_reg", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* y_reg, bb_leaf.block()->AddRegister("y_reg", p->GetBitsType(32)));
  XLS_ASSERT_OK(bb_leaf.AddClockPort("clk"));
  bb_leaf.RegisterWrite(x_reg, bb_leaf.InputPort("x", p->GetBitsType(32)));
  bb_leaf.RegisterWrite(y_reg, bb_leaf.InputPort("y", p->GetBitsType(32)));
  bb_leaf.OutputPort("res", bb_leaf.Add(bb_leaf.RegisterRead(x_reg),
                                        bb_leaf.RegisterRead(y_reg)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, bb_leaf.Build());

  BlockBuilder bb_top(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * left,
                           bb_top.block()->AddBlockInstantiation("left", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * right,
      bb_top.block()->AddBlockInstantiation("right", leaf));
  bb_top.InstantiationInput(left, "x",
                            bb_top.InputPort("a", p->GetBitsType(32)));
  bb_top.InstantiationInput(left, "y", bb_top.Literal(UBits(2, 32)));
  bb_top.InstantiationInput(right, "x",
                            bb_top.InputPort("b", p->GetBitsType(32)));
  bb_top.InstantiationInput(right, "y", bb_top.Literal(UBits(3, 32)));
  bb_top.OutputPort("res",
                    bb_top.UMul(bb_top.InstantiationOutput(left, "res"),
                                bb_top.InstantiationOutput(right, "res")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb_top.Build());

  ScopedRecordIr sri(p.get());

  BlockInliningPass bip;
  CodegenPassUnit pu(p.get(), top);
  CodegenPassResults results;
  CodegenPassOptions opt;
  ASSERT_THAT(bip.Run(&pu, opt, &results), absl_testing::IsOkAndHolds(true));

  Block* inlined = pu.top_block;
  EXPECT_THAT(inlined->nodes(),
              testing::Not(testing::Contains(m::InstantiationInput())));
  EXPECT_THAT(inlined->nodes(),
              testing::Not(testing::Contains(m::InstantiationOutput())));
  EXPECT_THAT(results.register_renames,
              UnorderedElementsAre(Pair("left::x_reg", "left__x_reg"),
                                   Pair("left::y_reg", "left__y_reg"),
                                   Pair("right::x_reg", "right__x_reg"),
                                   Pair("right::y_reg", "right__y_reg")));

  InterpreterBlockEvaluator eval;
  XLS_ASSERT_OK_AND_ASSIGN(auto oracle, eval.NewContinuation(top));
  XLS_ASSERT_OK_AND_ASSIGN(auto test, eval.NewContinuation(inlined));
  std::vector<std::pair<int64_t, int64_t>> test_vector{
      {2, 3}, {1, 0}, {0, 3}, {4, 12}, {0, 0}, {0, 0},
  };
  int64_t i = 0;
  for (const auto& [a, b] : test_vector) {
    XLS_ASSERT_OK(test->RunOneCycle(
        {{"a", Value(UBits(a, 32))}, {"b", Value(UBits(b, 32))}}));
    XLS_ASSERT_OK(oracle->RunOneCycle(
        {{"a", Value(UBits(a, 32))}, {"b", Value(UBits(b, 32))}}));
    EXPECT_EQ(oracle->output_ports(), test->output_ports());
    RecordProperty(
        absl::StrFormat("test_out_%d", i),
        absl::StrFormat("(%d, %d) -> %s", a, b,
                        testing::PrintToString(test->output_ports())));
    ++i;
  }
}

TEST_F(BlockInliningPassTest, InlineBlocksWithFifo) {
  auto p = CreatePackage();

  // (define_fifo foobar 4)
  // (define leaf (x y)
  //   (+ x (foobar y)))
  // (define head (a b)
  //   (let ((left (leaf a 2))
  //         (right (leaf b 3)))
  //     (* left right)))

  BlockBuilder bb_leaf("leaf", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * fifo_inst,
                           bb_leaf.block()->AddFifoInstantiation(
                               "foobar",
                               FifoConfig(/*depth=*/4, /*bypass=*/false,
                                          /*register_push_outputs=*/false,
                                          /*register_pop_outputs=*/false),
                               p->GetBitsType(32)));
  bb_leaf.InstantiationInput(fifo_inst, "push_data",
                             bb_leaf.InputPort("y", p->GetBitsType(32)));
  bb_leaf.OutputPort(
      "res", bb_leaf.Add(bb_leaf.InputPort("x", p->GetBitsType(32)),
                         bb_leaf.InstantiationOutput(fifo_inst, "pop_data")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, bb_leaf.Build());

  BlockBuilder bb_top(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * left,
                           bb_top.block()->AddBlockInstantiation("left", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * right,
      bb_top.block()->AddBlockInstantiation("right", leaf));
  bb_top.InstantiationInput(left, "x",
                            bb_top.InputPort("a", p->GetBitsType(32)));
  bb_top.InstantiationInput(left, "y", bb_top.Literal(UBits(2, 32)));
  bb_top.InstantiationInput(right, "x",
                            bb_top.InputPort("b", p->GetBitsType(32)));
  bb_top.InstantiationInput(right, "y", bb_top.Literal(UBits(3, 32)));
  bb_top.OutputPort("res",
                    bb_top.UMul(bb_top.InstantiationOutput(left, "res"),
                                bb_top.InstantiationOutput(right, "res")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb_top.Build());

  ScopedRecordIr sri(p.get());

  BlockInliningPass bip;
  CodegenPassUnit pu(p.get(), top);
  CodegenPassResults results;
  CodegenPassOptions opt;
  ASSERT_THAT(bip.Run(&pu, opt, &results), absl_testing::IsOkAndHolds(true));

  Block* inlined = pu.top_block;
  XLS_ASSERT_OK_AND_ASSIGN(auto* left_foobar_inst,
                           inlined->GetInstantiation("left::foobar"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* right_foobar_inst,
                           inlined->GetInstantiation("right::foobar"));
  EXPECT_THAT(
      inlined->nodes(),
      testing::AllOf(
          testing::Contains(m::InstantiationInput(
              m::Literal(UBits(2, 32)), "push_data", left_foobar_inst)),
          testing::Contains(m::InstantiationInput(
              m::Literal(UBits(3, 32)), "push_data", right_foobar_inst))));
  EXPECT_THAT(inlined->nodes(),
              testing::AllOf(testing::Contains(m::InstantiationOutput(
                                 "pop_data", left_foobar_inst)),
                             testing::Contains(m::InstantiationOutput(
                                 "pop_data", right_foobar_inst))));
}

TEST_F(BlockInliningPassTest, InlineBlocksWithExtern) {
  auto p = CreatePackage();

  // (define_extern foobar)
  // (define leaf (x y)
  //   (+ x (foobar y)))
  // (define head (a b)
  //   (let ((left (leaf a 2))
  //         (right (leaf b 3)))
  //     (* left right)))
  ForeignFunctionData data;
  data.set_code_template("{fn}");
  FunctionBuilder fb("foobar", p.get());
  fb.Add(fb.Param("a", p->GetBitsType(32)), fb.Literal(UBits(4, 32)));
  fb.SetForeignFunctionData(data);
  XLS_ASSERT_OK_AND_ASSIGN(Function * foobar_impl, fb.Build());

  BlockBuilder bb_leaf("leaf", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * extern_inst,
                           bb_leaf.block()->AddInstantiation(
                               "foobar", std::make_unique<ExternInstantiation>(
                                             "foobar", foobar_impl)));
  bb_leaf.InstantiationInput(extern_inst, "foobar.0",
                             bb_leaf.InputPort("y", p->GetBitsType(32)));
  bb_leaf.OutputPort(
      "res", bb_leaf.Add(bb_leaf.InputPort("x", p->GetBitsType(32)),
                         bb_leaf.InstantiationOutput(extern_inst, "return")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * leaf, bb_leaf.Build());

  BlockBuilder bb_top(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * left,
                           bb_top.block()->AddBlockInstantiation("left", leaf));
  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * right,
      bb_top.block()->AddBlockInstantiation("right", leaf));
  bb_top.InstantiationInput(left, "x",
                            bb_top.InputPort("a", p->GetBitsType(32)));
  bb_top.InstantiationInput(left, "y", bb_top.Literal(UBits(2, 32)));
  bb_top.InstantiationInput(right, "x",
                            bb_top.InputPort("b", p->GetBitsType(32)));
  bb_top.InstantiationInput(right, "y", bb_top.Literal(UBits(3, 32)));
  bb_top.OutputPort("res",
                    bb_top.UMul(bb_top.InstantiationOutput(left, "res"),
                                bb_top.InstantiationOutput(right, "res")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top, bb_top.Build());

  ScopedRecordIr sri(p.get());

  BlockInliningPass bip;
  CodegenPassUnit pu(p.get(), top);
  CodegenPassResults results;
  CodegenPassOptions opt;
  ASSERT_THAT(bip.Run(&pu, opt, &results), absl_testing::IsOkAndHolds(true));

  Block* inlined = pu.top_block;
  XLS_ASSERT_OK_AND_ASSIGN(auto* left_foobar_inst,
                           inlined->GetInstantiation("left::foobar"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* right_foobar_inst,
                           inlined->GetInstantiation("right::foobar"));
  EXPECT_THAT(
      inlined->nodes(),
      testing::AllOf(
          testing::Contains(m::InstantiationInput(
              m::Literal(UBits(2, 32)), "foobar.0", left_foobar_inst)),
          testing::Contains(m::InstantiationInput(
              m::Literal(UBits(3, 32)), "foobar.0", right_foobar_inst))));
  EXPECT_THAT(inlined->nodes(),
              testing::AllOf(testing::Contains(m::InstantiationOutput(
                                 "return", left_foobar_inst)),
                             testing::Contains(m::InstantiationOutput(
                                 "return", right_foobar_inst))));
}
}  // namespace
}  // namespace xls::verilog
