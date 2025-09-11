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

#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/float32_mul_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/float32_upcast_jit_wrapper.h"
#include "xls/examples/dslx_module/some_caps_jit_wrapper.h"
#include "xls/examples/dslx_module/some_caps_opt_jit_wrapper.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/ir/value_view.h"
#include "xls/jit/compound_type_jit_wrapper.h"
#include "xls/jit/multi_func_block_wrapper.h"
#include "xls/jit/multi_func_with_trace_block_wrapper.h"
#include "xls/jit/testdata/v1/multi_proc_jit_wrapper.h"
#include "xls/jit/testdata/v1/test_jit_wrapper.h"
#include "xls/jit/testdata/v2/multi_proc_jit_wrapper.h"
#include "xls/jit/testdata/v2/test_jit_wrapper.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::something::cool::CompoundJitWrapper;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(JitWrapperTest, CanCallTargetsWithSameName) {
  XLS_ASSERT_OK_AND_ASSIGN(auto v1_jit, jit_test::v1::TestIr::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto v2_jit, jit_test::v2::TestIr::Create());
  EXPECT_THAT(v1_jit->Run(0), IsOkAndHolds(1));
  EXPECT_THAT(v2_jit->Run(0), IsOkAndHolds(2));
}

TEST(JitWrapperTest, CanCallTargetsWithSameNameMultiProc) {
  XLS_ASSERT_OK_AND_ASSIGN(auto v1_jit, jit_test::v1::TestProc::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto v2_jit, jit_test::v2::TestProc::Create());
  XLS_EXPECT_OK(v1_jit->SendToBytesSrc(1));
  XLS_EXPECT_OK(v1_jit->SendToBytesSrc(20));
  XLS_EXPECT_OK(v2_jit->SendToBytesSrc(1));
  XLS_EXPECT_OK(v2_jit->SendToBytesSrc(20));
  XLS_EXPECT_OK(v1_jit->TickUntilBlocked());
  XLS_EXPECT_OK(v2_jit->TickUntilBlocked());
  EXPECT_THAT(v1_jit->ReceiveFromBytesResult(), IsOkAndHolds(0));
  EXPECT_THAT(v1_jit->ReceiveFromBytesResult(), IsOkAndHolds(200));
  EXPECT_THAT(v2_jit->ReceiveFromBytesResult(), IsOkAndHolds(10));
  EXPECT_THAT(v2_jit->ReceiveFromBytesResult(), IsOkAndHolds(0));
}

TEST(JitWrapperTest, BasicFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto complex_value,
      ValueBuilder::Tuple(
          {Value(UBits(1, 1)), Value(UBits(33, 223)),
           ValueBuilder::Array(
               {ValueBuilder::Tuple(
                    {Value(UBits(3, 3)),
                     ValueBuilder::Array({ValueBuilder::Tuple(
                         {Value(UBits(32, 32)), Value(UBits(2, 2))})})}),
                ValueBuilder::Tuple(
                    {Value(UBits(5, 3)),
                     ValueBuilder::Array({ValueBuilder::Tuple(
                         {Value(UBits(3333, 32)), Value(UBits(0, 2))})})})})})
          .Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, CompoundJitWrapper::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      jit->Run(Value(UBits(0, 32)), Value::Tuple({}), complex_value));
  EXPECT_EQ(result, Value::Tuple({Value::Tuple({}), Value(UBits(1, 32)),
                                  complex_value}));
}

TEST(JitWrapperTest, SpecializedFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::F32ToF64::Create());
  XLS_ASSERT_OK_AND_ASSIGN(double dv, jit->Run(3.14f));
  EXPECT_EQ(dv, static_cast<double>(3.14f));
}

TEST(JitWrapperTest, SpecializedFunctionCall2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::Float32Mul::Create());
  XLS_ASSERT_OK_AND_ASSIGN(float res, jit->Run(3.14f, 1.2345f));
  EXPECT_EQ(res, 3.14f * 1.2345f);
}

TEST(JitWrapperTest, PackedFunctionCall) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::F32ToF64::Create());
  double dv = -1.23;
  float fv = 3.14;
  PackedFloat pfv(std::bit_cast<uint8_t*>(&fv), 0);
  PackedDouble pdv(std::bit_cast<uint8_t*>(&dv), 0);
  XLS_ASSERT_OK(jit->Run(pfv, pdv));
  EXPECT_EQ(dv, static_cast<double>(3.14f));
  EXPECT_EQ(fv, 3.14f);
}

TEST(JitWrapperTest, PackedFunctionCall2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, fp::Float32Mul::Create());
  float lv = 3.14f;
  float rv = 1.2345f;
  float result;
  PackedFloat plv(std::bit_cast<uint8_t*>(&lv), 0);
  PackedFloat prv(std::bit_cast<uint8_t*>(&rv), 0);
  PackedFloat pres(std::bit_cast<uint8_t*>(&result), 0);
  XLS_ASSERT_OK(jit->Run(plv, prv, pres));
  EXPECT_EQ(result, 3.14f * 1.2345f);
  EXPECT_EQ(lv, 3.14f);
  EXPECT_EQ(rv, 1.2345f);
}

std::array<uint8_t, 8> StrArray(std::string_view sv) {
  EXPECT_EQ(sv.size(), 8);
  std::array<uint8_t, 8> ret;
  absl::c_copy_n(sv, ret.size(), ret.begin());
  return ret;
}
TEST(JitWrapperTest, Proc2) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, dslx::examples::SomeCaps::Create());
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->Tick(),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("deadlock")));
}
TEST(JitWrapperTest, Proc) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, dslx::examples::SomeCaps::Create());
  XLS_ASSERT_OK(
      jit->SendToStringInput({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'q', 'r', 's', 't', 'u', 'v', 'w', 'x'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'y', 'z', '0', '1', '2', '3', '4', '5'}));
  // Should take 4 ticks to consume everything.
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("ABCDEFGH"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("ijklmnop"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("QrStUvWx"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("YZ012345"))));
  // No more data right now.
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Next tick succeeds but doesn't finish
  XLS_ASSERT_OK(jit->Tick());
  // No data.
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Another tick gives an error since all procs are blocked.
  EXPECT_THAT(jit->Tick(),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("deadlock")));
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Giving either one a value unblocks however.
  XLS_ASSERT_OK(jit->SendToBlocker(Value::Tuple({})));
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Next tick makes progress but doesn't finish, waiting on channels
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Another tick gives an error since all procs are blocked.
  EXPECT_THAT(jit->Tick(),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("deadlock")));
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));

  XLS_ASSERT_OK(jit->SendToStringInput(StrArray("foobar12")));
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("foobar12"))));
  // Next tick makes progress but doesn't finish, waiting on channels
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
  // Another tick gives an error since all procs are blocked.
  EXPECT_THAT(jit->Tick(),
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::ContainsRegex("deadlock")));
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
}

TEST(JitWrapperTest, ProcTickUnitlBlocked) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, dslx::examples::SomeCaps::Create());
  XLS_ASSERT_OK(
      jit->SendToStringInput({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'q', 'r', 's', 't', 'u', 'v', 'w', 'x'}));
  XLS_ASSERT_OK(
      jit->SendToStringInput({'y', 'z', '0', '1', '2', '3', '4', '5'}));
  XLS_ASSERT_OK(jit->TickUntilBlocked());
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("ABCDEFGH"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("ijklmnop"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("QrStUvWx"))));
  EXPECT_THAT(jit->ReceiveFromStringOutput(),
              IsOkAndHolds(Optional(StrArray("YZ012345"))));
  // No more data right now.
  EXPECT_THAT(jit->ReceiveFromStringOutput(), IsOkAndHolds(std::nullopt));
}

TEST(JitWrapperTest, ProcOptIrWrapper) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit, dslx::examples::SomeCapsOpt::Create());
  XLS_ASSERT_OK(
      jit->SendToExternalInputWire({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}));
  XLS_ASSERT_OK(
      jit->SendToExternalInputWire({'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'}));
  XLS_ASSERT_OK(
      jit->SendToExternalInputWire({'q', 'r', 's', 't', 'u', 'v', 'w', 'x'}));
  XLS_ASSERT_OK(
      jit->SendToExternalInputWire({'y', 'z', '0', '1', '2', '3', '4', '5'}));
  // XLS_ASSERT_OK(jit->TickUntilBlocked());
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  XLS_ASSERT_OK(jit->Tick());
  EXPECT_THAT(jit->ReceiveFromExternalOutputWire(),
              IsOkAndHolds(Optional(StrArray("ABCDEFGH"))));
  EXPECT_THAT(jit->ReceiveFromExternalOutputWire(),
              IsOkAndHolds(Optional(StrArray("ijklmnop"))));
  EXPECT_THAT(jit->ReceiveFromExternalOutputWire(),
              IsOkAndHolds(Optional(StrArray("QrStUvWx"))));
  EXPECT_THAT(jit->ReceiveFromExternalOutputWire(),
              IsOkAndHolds(Optional(StrArray("YZ012345"))));
  // No more data right now.
  EXPECT_THAT(jit->ReceiveFromExternalOutputWire(), IsOkAndHolds(std::nullopt));
}

TEST(JitWrapperTest, BlockRunsOneCycle) {
  XLS_ASSERT_OK_AND_ASSIGN(auto jit,
                           something::cool::MultiFuncBlockJit::Create());
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(
      cont->SetInputPorts(something::cool::MultiFuncBlockJitPorts().SetX(2)));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_EQ(cont->GetOut(), 10);
}

TEST(JitWrapperTest, BlockSetRegisterWorks) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto jit, something::cool::MultiFuncWithTraceBlockJit::Create());
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(cont->SetRegisters({
      {"p1_res", Value(UBits(10, 8))},  // this will output immediately
      {"p0_x", Value(UBits(1, 8))},     // 5*1 will be the output next cycle
  }));
  XLS_ASSERT_OK(cont->SetInputPorts(
      something::cool::MultiFuncWithTraceBlockJitPorts().SetX(
          3)));  // After the pipeline is flushed, 5*3 will be the output.
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_EQ(cont->GetOut(), 10);
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_EQ(cont->GetOut(), 5);
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_EQ(cont->GetOut(), 15);
}

TEST(JitWrapperTest, BlockSetRegisterObservable) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto jit, something::cool::MultiFuncWithTraceBlockJit::Create());
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(cont->SetRegisters({
      {"p1_res", Value(UBits(10, 8))},  // this will output immediately
      {"p0_x", Value(UBits(1, 8))},     // 5*1 will be the output next cycle
  }));
  EXPECT_THAT(cont->GetRegistersMap(),
              UnorderedElementsAre(Pair("p1_res", Value(UBits(10, 8))),
                                   Pair("p0_x", Value(UBits(1, 8)))));
  XLS_ASSERT_OK(cont->SetInputPorts(
      something::cool::MultiFuncWithTraceBlockJitPorts().SetX(
          3)));  // After the pipeline is flushed, 5*3 will be the output.
  EXPECT_THAT(cont->GetRegistersMap(),
              UnorderedElementsAre(Pair("p1_res", Value(UBits(10, 8))),
                                   Pair("p0_x", Value(UBits(1, 8)))));
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_EQ(cont->GetOut(), 10);
  EXPECT_THAT(cont->GetRegistersMap(),
              UnorderedElementsAre(Pair("p1_res", Value(UBits(5, 8))),
                                   Pair("p0_x", Value(UBits(3, 8)))));
}

TEST(JitWrapperTest, BlockTraceEventsWork) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto jit, something::cool::MultiFuncWithTraceBlockJit::Create());
  auto cont = jit->NewContinuation();
  XLS_ASSERT_OK(cont->SetRegisters({
      {"p1_res", Value(UBits(10, 8))},  // this will output immediately
      {"p0_x", Value(UBits(1, 8))},     // 5*1 will be the output next cycle
  }));
  XLS_ASSERT_OK(cont->SetInputPorts(
      something::cool::MultiFuncWithTraceBlockJitPorts().SetX(
          3)));  // After the pipeline is flushed, 5*3 will be the output.
  XLS_ASSERT_OK(jit->RunOneCycle(*cont));
  EXPECT_THAT(cont->interpreter_events().GetAssertMessages(), IsEmpty());
  EXPECT_THAT(cont->interpreter_events().GetTraceMessageStrings(),
              ElementsAre("mf_2(2) -> 4", "mf_1(1) -> 5"));
}

}  // namespace
}  // namespace xls
