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

#include "xls/ir/nodes.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "xls/ir/op.h"

namespace xls {
namespace {

// List of all op-classes.
#define FOR_EACH_OP_CLASS(M) \
  M(AfterAll)                \
  M(ArithOp)                 \
  M(Array)                   \
  M(ArrayConcat)             \
  M(ArrayIndex)              \
  M(ArraySlice)              \
  M(ArrayUpdate)             \
  M(Assert)                  \
  M(BinOp)                   \
  M(BitSlice)                \
  M(BitSliceUpdate)          \
  M(BitwiseReductionOp)      \
  M(CompareOp)               \
  M(Concat)                  \
  M(CountedFor)              \
  M(Cover)                   \
  M(Decode)                  \
  M(DynamicBitSlice)         \
  M(DynamicCountedFor)       \
  M(Encode)                  \
  M(ExtendOp)                \
  M(Gate)                    \
  M(InputPort)               \
  M(InstantiationInput)      \
  M(InstantiationOutput)     \
  M(Invoke)                  \
  M(Literal)                 \
  M(Map)                     \
  M(MinDelay)                \
  M(NaryOp)                  \
  M(NewChannel)              \
  M(Next)                    \
  M(OneHot)                  \
  M(OneHotSelect)            \
  M(OutputPort)              \
  M(Param)                   \
  M(PartialProductOp)        \
  M(PrioritySelect)          \
  M(Receive)                 \
  M(RecvChannelEnd)          \
  M(RegisterRead)            \
  M(RegisterWrite)           \
  M(Select)                  \
  M(Send)                    \
  M(SendChannelEnd)          \
  M(StateRead)               \
  M(Trace)                   \
  M(Tuple)                   \
  M(TupleIndex)              \
  M(UnOp)

class IrNodesTest : public testing::TestWithParam<Op> {};
TEST_P(IrNodesTest, OpIsInOneNodeType) {
  std::vector<std::string> found;
#define CNT_OPS_IN(op)                            \
  if (absl::c_count(op::kOps, GetParam()) != 0) { \
    found.push_back(#op);                         \
  }
  FOR_EACH_OP_CLASS(CNT_OPS_IN);
  EXPECT_THAT(found, testing::SizeIs(1))
      << "Bad kOps values for the listed nodes. Some claim to handle the same "
         "ir-opcode. If list is empty possibly unlisted node type.";
}

INSTANTIATE_TEST_SUITE_P(IrNodesTest, IrNodesTest, testing::ValuesIn(kAllOps),
                         testing::PrintToStringParamName());
}  // namespace
}  // namespace xls
