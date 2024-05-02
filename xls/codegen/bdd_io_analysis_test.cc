// Copyright 2022 The XLS Authors
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

#include "xls/codegen/bdd_io_analysis.h"

#include <memory>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

class BddIOAnalysisPassTest : public IrTestBase {};

TEST_F(BddIOAnalysisPassTest, SingleStreamingSend) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateSingleValueChannel("out1", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);

  pb.Send(out0, in_val);
  pb.Send(out1, in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(bool mutually_exclusive,
                           AreStreamingOutputsMutuallyExclusive(proc));
  EXPECT_EQ(mutually_exclusive, true);
}

TEST_F(BddIOAnalysisPassTest, StreamingSendWithSendIf) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);

  BValue one = pb.Literal(UBits(1, 32));

  pb.Send(out0, in_val);
  pb.SendIf(out1, pb.Eq(in_val, one), in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(bool mutually_exclusive,
                           AreStreamingOutputsMutuallyExclusive(proc));
  EXPECT_EQ(mutually_exclusive, false);
}

TEST_F(BddIOAnalysisPassTest, MutuallyExclusiveSendIf) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * sel,
      package.CreateStreamingChannel("sel", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out2,
      package.CreateStreamingChannel("out2", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);
  BValue sel_val = pb.Receive(sel);

  BValue zero = pb.Literal(UBits(0, 32));
  BValue one = pb.Literal(UBits(1, 32));
  BValue two = pb.Literal(UBits(2, 32));

  pb.SendIf(out0, pb.Eq(sel_val, zero), in_val);
  pb.SendIf(out1, pb.Eq(sel_val, one), in_val);
  pb.SendIf(out2, pb.Eq(sel_val, two), in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(bool mutually_exclusive,
                           AreStreamingOutputsMutuallyExclusive(proc));
  EXPECT_EQ(mutually_exclusive, true);
}

TEST_F(BddIOAnalysisPassTest, MutuallyExclusiveSendIfWithRange) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * sel,
      package.CreateStreamingChannel("sel", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(),
                          /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);
  BValue sel_val = pb.Receive(sel);

  BValue one_zero_two_four = pb.Literal(UBits(1024, 32));

  pb.SendIf(out0, pb.ULt(sel_val, one_zero_two_four), in_val);
  pb.SendIf(out1, pb.UGt(sel_val, one_zero_two_four), in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(bool mutually_exclusive,
                           AreStreamingOutputsMutuallyExclusive(proc));
  EXPECT_EQ(mutually_exclusive, true);
}

TEST_F(BddIOAnalysisPassTest, NonMutuallyExclusiveSendIf) {
  auto package_ptr = std::make_unique<Package>(TestName());
  Package& package = *package_ptr;

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * sel,
      package.CreateStreamingChannel("sel", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out2,
      package.CreateStreamingChannel("out2", ChannelOps::kSendOnly, u32));

  Value initial_state = Value(UBits(0, 32));
  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue in_val = pb.Receive(in);
  BValue sel_val = pb.Receive(sel);

  BValue zero = pb.Literal(UBits(0, 32));
  BValue one = pb.Literal(UBits(1, 32));

  pb.SendIf(out0, pb.Eq(sel_val, zero), in_val);
  pb.SendIf(out1, pb.Eq(sel_val, one), in_val);
  pb.SendIf(out2, pb.Eq(sel_val, one), in_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(bool mutually_exclusive,
                           AreStreamingOutputsMutuallyExclusive(proc));
  EXPECT_EQ(mutually_exclusive, false);
}

}  // namespace
}  // namespace xls
