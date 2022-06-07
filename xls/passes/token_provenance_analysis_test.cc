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

#include "xls/passes/token_provenance_analysis.h"

#include <stdint.h>

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

class TokenProvenanceAnalysisTest : public IrTestBase {};

TEST_F(TokenProvenanceAnalysisTest, Simple) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendReceive,
                                p->GetBitsType(32)));

  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 0)));
  BValue recv = pb.Receive(channel, pb.GetTokenParam());
  BValue t1 = pb.TupleIndex(recv, 0);
  BValue t2 = pb.Send(channel, t1, pb.Literal(UBits(50, 32)));
  BValue tuple = pb.Tuple(
      {t1, pb.Literal(UBits(50, 32)),
       pb.Tuple({pb.Literal(UBits(50, 32)), pb.Literal(UBits(50, 32))}),
       pb.Tuple({t2})});
  BValue t3 = pb.Assert(pb.TupleIndex(pb.TupleIndex(tuple, 3), 0),
                        pb.Literal(UBits(1, 1)), "assertion failed");
  BValue t4 = pb.Trace(t3, pb.Literal(UBits(1, 1)), {}, "");
  BValue t5 = pb.Cover(t4, pb.Literal(UBits(1, 1)), "trace");
  BValue t6 = pb.AfterAll({t3, t4, t5});

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(t6, {pb.Literal(UBits(0, 0))}));
  XLS_ASSERT_OK_AND_ASSIGN(TokenProvenance provenance,
                           TokenProvenanceAnalysis(proc));

  EXPECT_EQ(provenance.at(pb.GetTokenParam().node()).Get({}),
            pb.GetTokenParam().node());
  EXPECT_EQ(provenance.at(recv.node()).Get({0}), recv.node());
  EXPECT_EQ(provenance.at(recv.node()).Get({1}), nullptr);
  EXPECT_EQ(provenance.at(tuple.node()).Get({0}), recv.node());
  EXPECT_EQ(provenance.at(tuple.node()).Get({1}), nullptr);
  EXPECT_EQ(provenance.at(tuple.node()).Get({2, 0}), nullptr);
  EXPECT_EQ(provenance.at(tuple.node()).Get({2, 1}), nullptr);
  EXPECT_EQ(provenance.at(tuple.node()).Get({3, 0}), t2.node());
  EXPECT_EQ(provenance.at(t3.node()).Get({}), t3.node());
  EXPECT_EQ(provenance.at(t4.node()).Get({}), t4.node());
  EXPECT_EQ(provenance.at(t5.node()).Get({}), t5.node());
  EXPECT_EQ(provenance.at(t6.node()).Get({}), t6.node());
}

TEST_F(TokenProvenanceAnalysisTest, VeryLongChain) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), "token", p.get());
  BValue token = pb.GetTokenParam();
  for (int i = 0; i < 1000; ++i) {
    token = pb.Identity(token);
  }
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, std::vector<BValue>()));
  XLS_ASSERT_OK_AND_ASSIGN(TokenProvenance provenance,
                           TokenProvenanceAnalysis(proc));

  // The proc only consists of a token param and token-typed identity
  // operations.
  for (Node* node : proc->nodes()) {
    EXPECT_EQ(provenance.at(node).Get({}), proc->TokenParam());
  }
}


}  // namespace
}  // namespace xls
