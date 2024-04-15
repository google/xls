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

#include "xls/jit/block_jit.h"

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/block_evaluator_test_base.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/jit_runtime.h"

namespace xls {
namespace {

using testing::ElementsAre;

class BlockJitTest : public IrTestBase {};
TEST_F(BlockJitTest, ConstantToPort) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  auto input = bb.Literal(UBits(42, 8));
  bb.OutputPort("answer", input);

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));

  EXPECT_THAT(cont->GetOutputPorts(), ElementsAre(Value(UBits(42, 8))));
}
TEST_F(BlockJitTest, AddTwoPort) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  auto input1 = bb.InputPort("question", p->GetBitsType(8));
  auto input2 = bb.InputPort("question2", p->GetBitsType(8));
  bb.OutputPort("answer", bb.Add(input1, input2));

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(
      cont->SetInputPorts({Value(UBits(12, 8)), Value(UBits(30, 8))}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));

  EXPECT_THAT(cont->GetOutputPorts(), ElementsAre(Value(UBits(42, 8))));
}
TEST_F(BlockJitTest, ConstantToReg) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r,
                           bb.block()->AddRegister("test", p->GetBitsType(8)));
  auto input = bb.Literal(UBits(42, 8));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  bb.RegisterWrite(r, input);
  bb.RegisterRead(r);

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(cont->SetRegisters({Value(UBits(2, 8))}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));

  EXPECT_THAT(cont->GetRegisters(), ElementsAre(Value(UBits(42, 8))));
}
TEST_F(BlockJitTest, DelaySlot) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto r,
                           bb.block()->AddRegister("test", p->GetBitsType(8)));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  auto input = bb.InputPort("input", p->GetBitsType(8));
  bb.RegisterWrite(r, input);
  auto read = bb.RegisterRead(r);
  bb.OutputPort("output", read);

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();

  XLS_ASSERT_OK(cont->SetRegisters({Value(UBits(2, 8))}));

  XLS_ASSERT_OK(cont->SetInputPorts({Value(UBits(42, 8))}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_THAT(cont->GetRegisters(), ElementsAre(Value(UBits(42, 8))));
  EXPECT_THAT(cont->GetOutputPorts(), ElementsAre(Value(UBits(2, 8))));

  XLS_ASSERT_OK(cont->SetInputPorts({Value(UBits(12, 8))}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_THAT(cont->GetRegisters(), ElementsAre(Value(UBits(12, 8))));
  EXPECT_THAT(cont->GetOutputPorts(), ElementsAre(Value(UBits(42, 8))));
}

TEST_F(BlockJitTest, SetInputsWithViews) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  auto input1 = bb.InputPort("input1", p->GetBitsType(16));
  auto input2 = bb.InputPort("input2", p->GetBitsType(16));
  bb.OutputPort("output", bb.Add(input1, input2));

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();

  int16_t input_bits1 = -12;
  int16_t input_bits2 = 54;
  BitsView<16> bits1(reinterpret_cast<uint8_t*>(&input_bits1));
  BitsView<16> bits2(reinterpret_cast<uint8_t*>(&input_bits2));
  XLS_ASSERT_OK(cont->SetInputPorts(
      absl::Span<const uint8_t* const>{bits1.buffer(), bits2.buffer()}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));

  EXPECT_THAT(cont->GetOutputPorts(),
              testing::ElementsAre(Value(UBits(42, 16))));
}

TEST_F(BlockJitTest, SetRegistersWithViews) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto r1, bb.block()->AddRegister("test1", p->GetBitsType(16)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto r2, bb.block()->AddRegister("test2", p->GetBitsType(16)));
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  bb.RegisterWrite(r1, bb.Literal(UBits(0, 16)));
  bb.RegisterWrite(r2, bb.Literal(UBits(0, 16)));
  auto read1 = bb.RegisterRead(r1);
  auto read2 = bb.RegisterRead(r2);
  bb.OutputPort("output", bb.Add(read1, read2));

  XLS_ASSERT_OK_AND_ASSIGN(Block * b, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto runtime, JitRuntime::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, BlockJit::Create(b, runtime.get()));
  auto cont = jit->NewContinuation();

  int16_t input_bits1 = -12;
  int16_t input_bits2 = 54;
  BitsView<16> bits1(reinterpret_cast<uint8_t*>(&input_bits1));
  BitsView<16> bits2(reinterpret_cast<uint8_t*>(&input_bits2));
  XLS_ASSERT_OK(cont->SetRegisters(
      absl::Span<const uint8_t* const>{bits1.buffer(), bits2.buffer()}));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));

  EXPECT_THAT(cont->GetOutputPorts(),
              testing::ElementsAre(Value(UBits(42, 16))));
  EXPECT_THAT(cont->GetRegistersMap(),
              testing::UnorderedElementsAre(
                  testing::Pair("test1", Value(UBits(0, 16))),
                  testing::Pair("test2", Value(UBits(0, 16)))));
}
INSTANTIATE_TEST_SUITE_P(
    JitBlockCommonTest, BlockEvaluatorTest,
    testing::Values(
        BlockEvaluatorTestParam{.evaluator = &kJitBlockEvaluator,
                                .supports_hierarhical_blocks = false},
        BlockEvaluatorTestParam{.evaluator = &kStreamingJitBlockEvaluator,
                                .supports_hierarhical_blocks = false}),
    [](const auto& v) { return std::string(v.param.evaluator->name()); });

}  // namespace
}  // namespace xls
