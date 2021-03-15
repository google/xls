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

#include "xls/codegen/function_to_proc.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace verilog {
namespace {

class FunctionToProcTest : public IrTestBase {};

TEST_F(FunctionToProcTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           FunctionToProc(f, "SimpleFunctionProc"));

  EXPECT_EQ(proc->name(), "SimpleFunctionProc");
  EXPECT_EQ(proc->StateType(), p->GetTupleType({}));
  EXPECT_EQ(p->channels().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_ch, p->GetChannel("out"));
  EXPECT_TRUE(out_ch->IsPort());
  EXPECT_EQ(out_ch->supported_ops(), ChannelOps::kSendOnly);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * x_ch, p->GetChannel("x"));
  EXPECT_TRUE(x_ch->IsPort());
  EXPECT_EQ(x_ch->supported_ops(), ChannelOps::kReceiveOnly);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * y_ch, p->GetChannel("y"));
  EXPECT_TRUE(y_ch->IsPort());
  EXPECT_EQ(y_ch->supported_ops(), ChannelOps::kReceiveOnly);
}

TEST_F(FunctionToProcTest, ZeroWidthInputsAndOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetTupleType({}));
  BValue y = fb.Param("y", p->GetBitsType(0));
  fb.Param("z", p->GetBitsType(1234));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({x, y})));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           FunctionToProc(f, "SimpleFunctionProc"));

  EXPECT_EQ(proc->name(), "SimpleFunctionProc");
  EXPECT_EQ(proc->StateType(), p->GetTupleType({}));
  EXPECT_EQ(p->channels().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_ch, p->GetChannel("z"));
  EXPECT_TRUE(out_ch->IsPort());
  EXPECT_EQ(out_ch->supported_ops(), ChannelOps::kReceiveOnly);
}

}  // namespace
}  // namespace verilog
}  // namespace xls
