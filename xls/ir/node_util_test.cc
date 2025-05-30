// Copyright 2020 The XLS Authors
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

#include "xls/ir/node_util.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Key;
using ::testing::Not;
using ::testing::UnorderedElementsAre;

class Result {
 public:
  Result(int64_t leading_zero_count, int64_t set_bit_count,
         int64_t trailing_zero_count)
      : leading_zero_count_(leading_zero_count),
        set_bit_count_(set_bit_count),
        trailing_zero_count_(trailing_zero_count) {}

  bool operator==(const Result& other) const {
    return leading_zero_count_ == other.leading_zero_count_ &&
           set_bit_count_ == other.set_bit_count_ &&
           trailing_zero_count_ == other.trailing_zero_count_;
  }

 private:
  friend std::ostream& operator<<(std::ostream&, const Result&);

  int64_t leading_zero_count_;
  int64_t set_bit_count_;
  int64_t trailing_zero_count_;
};

std::ostream& operator<<(std::ostream& os, const Result& result) {
  os << absl::StreamFormat("{%d, %d, %d}", result.leading_zero_count_,
                           result.set_bit_count_, result.trailing_zero_count_);
  return os;
}

class NodeUtilTest : public IrTestBase {
 protected:
  absl::StatusOr<Result> RunOn(const Bits& bits) {
    auto p = CreatePackage();
    FunctionBuilder fb("f", p.get());
    fb.Literal(bits);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    Node* n = f->return_value();
    int64_t leading_zero_count, set_bit_count, trailing_zero_count;
    XLS_RET_CHECK(IsLiteralWithRunOfSetBits(
        n, &leading_zero_count, &set_bit_count, &trailing_zero_count));
    return Result{leading_zero_count, set_bit_count, trailing_zero_count};
  }

  void ExpectIr(std::string_view got, std::string_view test_name) {
    ExpectEqualToGoldenFile(
        absl::StrFormat("xls/ir/testdata/node_util_test_%s.ir", test_name),
        got);
  }
};

TEST_F(NodeUtilTest, RunOfSetBits) {
  Bits bits = UBits(0x0ff0, /*bit_count=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(Result t, RunOn(bits));
  EXPECT_EQ(Result(4, 8, 4), t);

  bits = UBits(0x00ff, /*bit_count=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(t, RunOn(bits));
  EXPECT_EQ(Result(8, 8, 0), t);

  bits = UBits(0x0500, /*bit_count=*/16);
  EXPECT_THAT(RunOn(bits), StatusIs(absl::StatusCode::kInternal));

  bits = UBits(0x0010, /*bit_count=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(t, RunOn(bits));
  EXPECT_EQ(Result(11, 1, 4), t);
}

TEST_F(NodeUtilTest, GatherBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Node * gathered,
                           GatherBits(f->return_value(), {0, 2, 3, 4, 6}));
  XLS_ASSERT_OK(f->set_return_value(gathered));
  EXPECT_THAT(f->return_value(), m::Concat(m::BitSlice(6, 1), m::BitSlice(2, 3),
                                           m::BitSlice(0, 1)));
}

TEST_F(NodeUtilTest, GatherNoBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * gathered,
      GatherBits(f->return_value(), absl::Span<const int64_t>{}));
  XLS_ASSERT_OK(f->set_return_value(gathered));
  EXPECT_THAT(f->return_value(), m::Literal(0));
}

TEST_F(NodeUtilTest, GatherAllTheBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * gathered, GatherBits(f->return_value(), {0, 1, 2, 3, 4, 5, 6, 7}));
  XLS_ASSERT_OK(f->set_return_value(gathered));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(NodeUtilTest, GatherBitsIndicesNotUnique) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(GatherBits(f->return_value(), {0, 2, 2}),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("Gather indices not unique.")));
}

TEST_F(NodeUtilTest, FillPattern) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(6));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector pattern,
                           StringToTernaryVector("0b01XX1X0XXX"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * filled,
                           FillPattern(pattern, f->return_value()));
  XLS_ASSERT_OK(f->set_return_value(filled));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal("bits[2]:0b01"),
                        m::BitSlice(m::Param("x"), /*start=*/4, /*width=*/2),
                        m::Literal("bits[1]:0b1"),
                        m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1),
                        m::Literal("bits[1]:0b0"),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/3)));
}

TEST_F(NodeUtilTest, FillPatternWithAllTheBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  TernaryVector pattern(8, TernaryValue::kUnknown);
  XLS_ASSERT_OK_AND_ASSIGN(Node * filled,
                           FillPattern(pattern, f->return_value()));
  XLS_ASSERT_OK(f->set_return_value(filled));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(NodeUtilTest, FillPatternWithNoBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  TernaryVector pattern(8, TernaryValue::kKnownZero);
  XLS_ASSERT_OK_AND_ASSIGN(Node * filled,
                           FillPattern(pattern, f->return_value()));
  XLS_ASSERT_OK(f->set_return_value(filled));
  EXPECT_THAT(f->return_value(), m::Literal(0, 8));
}

TEST_F(NodeUtilTest, IsLiteralMask) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto seven_3b = fb.Literal(UBits(0b111, 3));
  auto two_3b = fb.Literal(UBits(0b011, 3));
  auto one_1b = fb.Literal(UBits(0b1, 1));
  auto zero_1b = fb.Literal(UBits(0b0, 1));
  auto zero_0b = fb.Literal(UBits(0b0, 0));

  int64_t leading_zeros, trailing_ones;
  EXPECT_TRUE(IsLiteralMask(seven_3b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(0, leading_zeros);
  EXPECT_EQ(3, trailing_ones);

  EXPECT_TRUE(IsLiteralMask(two_3b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(1, leading_zeros);
  EXPECT_EQ(2, trailing_ones);

  EXPECT_TRUE(IsLiteralMask(one_1b.node(), &leading_zeros, &trailing_ones));
  EXPECT_EQ(0, leading_zeros);
  EXPECT_EQ(1, trailing_ones);

  EXPECT_FALSE(IsLiteralMask(zero_1b.node(), &leading_zeros, &trailing_ones));
  EXPECT_FALSE(IsLiteralMask(zero_0b.node(), &leading_zeros, &trailing_ones));
}

TEST_F(NodeUtilTest, NonReductiveEquivalents) {
  XLS_ASSERT_OK_AND_ASSIGN(Op and_op, OpToNonReductionOp(Op::kAndReduce));
  EXPECT_EQ(and_op, Op::kAnd);
  XLS_ASSERT_OK_AND_ASSIGN(Op or_op, OpToNonReductionOp(Op::kOrReduce));
  EXPECT_EQ(or_op, Op::kOr);
  XLS_ASSERT_OK_AND_ASSIGN(Op xor_op, OpToNonReductionOp(Op::kXorReduce));
  EXPECT_EQ(xor_op, Op::kXor);
  EXPECT_FALSE(OpToNonReductionOp(Op::kBitSlice).ok());
}

TEST_F(NodeUtilTest, ChannelNodes) {
  Package p("my_package");

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));
  ProcBuilder b(TestName(), &p);
  BValue tkn = b.StateElement("tkn", Value::Token());
  BValue state = b.StateElement("st", Value(UBits(0, 0)));
  BValue rcv = b.Receive(ch0, tkn);
  BValue send = b.Send(ch1, tkn, b.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, b.Build({b.AfterAll({b.TupleIndex(rcv, 0), send}), state}));

  EXPECT_THAT(GetChannelUsedByNode(rcv.node()), IsOkAndHolds(ch0));
  EXPECT_THAT(GetChannelUsedByNode(send.node()), IsOkAndHolds(ch1));
  EXPECT_THAT(GetChannelUsedByNode(proc->GetStateRead(int64_t{0})),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No channel associated with node")));
}

TEST_F(NodeUtilTest, ReplaceTupleIndicesWorksWithFunction) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue in = b.Param("in", p.GetTupleType({p.GetBitsType(8), p.GetBitsType(16),
                                            p.GetBitsType(32)}));
  BValue arg0 = b.SignExtend(b.TupleIndex(in, 0), 32);
  BValue arg1 = b.SignExtend(b.TupleIndex(in, 1), 32);
  BValue arg2 = b.TupleIndex(in, 2);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, b.BuildWithReturnValue(b.Add(b.Add(arg0, arg1), arg2)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * in_param, f->GetParamByName("in"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * lit0, f->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 8))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * lit1, f->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 32))));
  XLS_EXPECT_OK(ReplaceTupleElementsWith(in_param, {{0, lit0}, {2, lit1}}));

  ExpectIr(f->DumpIr(), TestName());
}

TEST_F(NodeUtilTest, ReplaceTupleIndicesWorksWithToken) {
  Package p("my_package");
  ProcBuilder b(TestName(), &p);
  BValue tkn = b.StateElement("tkn", Value::Token());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));

  BValue receive = b.Receive(ch0, tkn, SourceInfo(), "receive");
  BValue rcv_token = b.TupleIndex(receive, 0);
  BValue rcv_data = b.TupleIndex(receive, 1);
  BValue send = b.Send(ch1, rcv_token, rcv_data);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({send}));

  XLS_ASSERT_OK_AND_ASSIGN(Node * receive_node, proc->GetNode("receive"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * lit0,
      proc->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 32))));
  // Replace the receive (which is used by the send) with a literal and the
  // token parameter. Note that this example won't verify because we remove the
  // receive from the token chain of the next token. To make something that
  // works, we'd need to make an after_all and add the receive's output token to
  // it after calling ReplaceTupleElementsWith().
  XLS_EXPECT_OK(ReplaceTupleElementsWith(
      receive_node, {{0, proc->GetStateRead(int64_t{0})}, {1, lit0}}));

  ExpectIr(proc->DumpIr(), TestName());
}

TEST_F(NodeUtilTest, ReplaceTupleIndicesFailsWithDependentReplacement) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue in =
      b.Param("in", p.GetTupleType({p.GetBitsType(8), p.GetBitsType(32)}));
  BValue lhs =
      b.SignExtend(b.TupleIndex(in, 0), 32, SourceInfo(), /*name=*/"lhs");
  BValue rhs = b.TupleIndex(in, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           b.BuildWithReturnValue(b.Add(lhs, rhs)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * in_param, f->GetParamByName("in"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Literal * lit0, f->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 8))));
  XLS_ASSERT_OK_AND_ASSIGN(Node * lhs_node, f->GetNode("lhs"));
  EXPECT_THAT(ReplaceTupleElementsWith(in_param, {{0, lit0}, {1, lhs_node}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Replacement index 1 (lhs) depends on")));
}

TEST_F(NodeUtilTest, AndReduceTrailing) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(AndReduceTrailing(a, 2),
              IsOkAndHolds(m::AndReduce(m::BitSlice(a, 0, 2))));
}

TEST_F(NodeUtilTest, AndReduceTrailingEmpty) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(AndReduceTrailing(a, 0), IsOkAndHolds(m::Literal(UBits(1, 1))));
}

TEST_F(NodeUtilTest, OrReduceLeading) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(OrReduceLeading(a, 2),
              IsOkAndHolds(m::OrReduce(m::BitSlice(a, 6, 2))));
}

TEST_F(NodeUtilTest, OrReduceLeadingEmpty) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(OrReduceLeading(a, 0), IsOkAndHolds(m::Literal(UBits(0, 1))));
}

TEST_F(NodeUtilTest, NorReduceLeading) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(NorReduceLeading(a, 2),
              IsOkAndHolds(m::Not(m::OrReduce(m::BitSlice(a, 6, 2)))));
}

TEST_F(NodeUtilTest, NorReduceLeadingEmpty) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Param * a, f->GetParamByName("a"));

  EXPECT_THAT(NorReduceLeading(a, 0), IsOkAndHolds(m::Literal(UBits(1, 1))));
}

TEST_F(NodeUtilTest, NaryAndWithNoInputs) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(NaryAndIfNeeded(f, {}), IsOkAndHolds(m::Literal(UBits(1, 1))));
}

TEST_F(NodeUtilTest, NaryAndWithOneInput) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue a = b.Param("a", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(NaryAndIfNeeded(f, std::vector<Node*>{a.node(), a.node()}),
              IsOkAndHolds(m::Param("a")));
}

TEST_F(NodeUtilTest, NaryAndWithMultipleInputs) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue a0 = b.Param("a0", p.GetBitsType(1));
  BValue a1 = b.Param("a1", p.GetBitsType(1));
  BValue a2 = b.Param("a2", p.GetBitsType(1));
  BValue a3 = b.Param("a3", p.GetBitsType(1));
  BValue a4 = b.Param("a4", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(
      NaryAndIfNeeded(
          f, std::vector<Node*>{a0.node(), a1.node(), a2.node(), a3.node(),
                                a4.node(), a1.node(), a3.node(), a1.node()}),
      IsOkAndHolds(m::And(m::Param("a0"), m::Param("a1"), m::Param("a2"),
                          m::Param("a3"), m::Param("a4"))));
}

TEST_F(NodeUtilTest, NaryNorWithNoInputs) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  b.Param("a", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(NaryNorIfNeeded(f, {}), Not(IsOk()));
}

TEST_F(NodeUtilTest, NaryNorWithOneInput) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue a = b.Param("a", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(NaryNorIfNeeded(f, std::vector<Node*>{a.node(), a.node()}),
              IsOkAndHolds(m::Not(m::Param("a"))));
}

TEST_F(NodeUtilTest, NaryNorWithMultipleInputs) {
  Package p("my_package");
  FunctionBuilder b(TestName(), &p);
  BValue a0 = b.Param("a0", p.GetBitsType(1));
  BValue a1 = b.Param("a1", p.GetBitsType(1));
  BValue a2 = b.Param("a2", p.GetBitsType(1));
  BValue a3 = b.Param("a3", p.GetBitsType(1));
  BValue a4 = b.Param("a4", p.GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  EXPECT_THAT(
      NaryNorIfNeeded(
          f, std::vector<Node*>{a0.node(), a3.node(), a2.node(), a1.node(),
                                a4.node(), a1.node(), a3.node(), a1.node()}),
      IsOkAndHolds(m::Nor(m::Param("a0"), m::Param("a3"), m::Param("a2"),
                          m::Param("a1"), m::Param("a4"))));
}

TEST_F(NodeUtilTest, ChannelUsers) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch2, p.CreateStreamingChannel("ch2", ChannelOps::kSendReceive,
                                              p.GetBitsType(32)));
  TokenlessProcBuilder pb(TestName(), "tok", &p);
  BValue recv0 = pb.Receive(ch0);
  BValue recv2 = pb.Receive(ch2);
  BValue sum = pb.Add(recv0, recv2);
  BValue send1_0 = pb.Send(ch1, sum);
  BValue send1_1 = pb.Send(ch1, sum);
  BValue send2 = pb.Send(ch2, sum);

  XLS_ASSERT_OK(pb.Build({}));

  absl::flat_hash_map<Channel*, ChannelUsers> channel_users;
  XLS_ASSERT_OK_AND_ASSIGN(channel_users, GetChannelUsers(&p));

  EXPECT_THAT(channel_users,
              UnorderedElementsAre(Key(ch0), Key(ch1), Key(ch2)));

  // TokenlessProcBuilder returns the tuple_index() of the receive, so go
  // backwards to get the original receive node.
  ASSERT_TRUE(recv0.node()->op() == Op::kTupleIndex);
  Node* recv0_node = recv0.node()->As<TupleIndex>()->operand(0);
  ASSERT_TRUE(recv2.node()->op() == Op::kTupleIndex);
  Node* recv2_node = recv2.node()->As<TupleIndex>()->operand(0);

  EXPECT_THAT(channel_users[ch0].receives, UnorderedElementsAre(recv0_node));
  EXPECT_THAT(channel_users[ch1].sends,
              UnorderedElementsAre(send1_0.node(), send1_1.node()));
  EXPECT_THAT(channel_users[ch2].sends, UnorderedElementsAre(send2.node()));
  EXPECT_THAT(channel_users[ch2].receives, UnorderedElementsAre(recv2_node));
}

TEST_F(NodeUtilTest, GetNodeAtIndex) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Value val, ValueBuilder::Tuple(
                     {ValueBuilder::Token(), ValueBuilder::Bits(UBits(32, 32)),
                      ValueBuilder::Bits(UBits(33, 33)),
                      ValueBuilder::Tuple({ValueBuilder::Bits(UBits(1, 1)),
                                           ValueBuilder::UBits2DArray(
                                               {{1, 2, 3}, {4, 5, 6}}, 5)})})
                     .Build());
  BValue nd = fb.Literal(val);

  EXPECT_THAT(GetNodeAtIndex(nd.node(), {}), nd.node());
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {0}),
              IsOkAndHolds(m::TupleIndex(nd.node(), 0)));
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {2}),
              IsOkAndHolds(m::TupleIndex(nd.node(), 2)));
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {2, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {3, 1, 1, 2}),
              IsOkAndHolds(m::ArrayIndex(
                  m::TupleIndex(m::TupleIndex(nd.node(), 3), 1),
                  {m::Literal(UBits(1, 64)), m::Literal(UBits(2, 64))})));
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {3, 1, 1, 1, 1, 1, 1, 1}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetNodeAtIndex(nd.node(), {44}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  RecordProperty("ir", fb.Build().value()->DumpIr());
}

TEST_F(NodeUtilTest, CompareLiteral) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * big, CompareLiteral(fb.Param("foo", p->GetBitsType(300)).node(),
                                 42, Op::kULt));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * little, CompareLiteral(fb.Param("foo2", p->GetBitsType(16)).node(),
                                    42, Op::kULt));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * same, CompareLiteral(fb.Param("foo3", p->GetBitsType(64)).node(),
                                  42, Op::kULt));
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({BValue(big, &fb), BValue(little, &fb),
                                        BValue(same, &fb)}))
          .status());
  EXPECT_THAT(big, m::ULt(m::Param("foo"), m::Literal(UBits(42, 300))));
  EXPECT_THAT(little, m::ULt(m::Param("foo2"), m::Literal(UBits(42, 16))));
  EXPECT_THAT(same, m::ULt(m::Param("foo3"), m::Literal(UBits(42, 64))));
}

TEST_F(NodeUtilTest, CompareNumeric) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * big,
      CompareNumeric(fb.Param("foo", p->GetBitsType(100)).node(),
                     fb.Param("bar", p->GetBitsType(16)).node(), Op::kULt));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * little,
      CompareNumeric(fb.Param("foo2", p->GetBitsType(16)).node(),
                     fb.Param("bar2", p->GetBitsType(100)).node(), Op::kULt));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * same,
      CompareNumeric(fb.Param("foo3", p->GetBitsType(16)).node(),
                     fb.Param("bar3", p->GetBitsType(16)).node(), Op::kULt));
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({BValue(big, &fb), BValue(little, &fb),
                                        BValue(same, &fb)}))
          .status());
  EXPECT_THAT(big, m::ULt(m::Param("foo"), m::ZeroExt(m::Param("bar"))));
  EXPECT_THAT(little, m::ULt(m::ZeroExt(m::Param("foo2")), m::Param("bar2")));
  EXPECT_THAT(same, m::ULt(m::Param("foo3"), m::Param("bar3")));
}

TEST_F(NodeUtilTest, UnsignedUpperBoundLiteral) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * big, UnsignedUpperBoundLiteral(
                      fb.Param("foo", p->GetBitsType(100)).node(), 42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * small, UnsignedUpperBoundLiteral(
                        fb.Param("foo2", p->GetBitsType(10)).node(), 42));
  // NB 0x2000 needs 14 bits at least.
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * impossible,
      UnsignedUpperBoundLiteral(fb.Param("foo3", p->GetBitsType(10)).node(),
                                0x2000));
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({BValue(big, &fb), BValue(small, &fb),
                                        BValue(impossible, &fb)}))
          .status());
  EXPECT_THAT(big, m::PrioritySelect(
                       m::UGt(m::Param("foo"), m::Literal(UBits(42, 100))),
                       {m::Literal(UBits(42, 100))}, m::Param("foo")));
  EXPECT_THAT(small, m::PrioritySelect(
                         m::UGt(m::Param("foo2"), m::Literal(UBits(42, 10))),
                         {m::Literal(UBits(42, 10))}, m::Param("foo2")));
  EXPECT_THAT(impossible, m::Param("foo3"));
}

TEST_F(NodeUtilTest, UnsignedBoundByLiterals) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * standard,
      UnsignedBoundByLiterals(fb.Param("foo", p->GetBitsType(100)).node(), 10,
                              42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * no_high,
      UnsignedBoundByLiterals(fb.Param("foo2", p->GetBitsType(10)).node(), 10,
                              0x1000));
  XLS_ASSERT_OK(fb.BuildWithReturnValue(
                      fb.Tuple({BValue(standard, &fb), BValue(no_high, &fb)}))
                    .status());
  RecordProperty("ir", p->DumpIr());
  EXPECT_THAT(
      standard,
      m::PrioritySelect(
          m::Concat(m::UGt(m::Param("foo"), m::Literal(UBits(42, 100))),
                    m::ULt(m::Param("foo"), m::Literal(UBits(10, 100)))),
          {m::Literal(UBits(10, 100)), m::Literal(UBits(42, 100))},
          m::Param("foo")));
  EXPECT_THAT(no_high, m::PrioritySelect(
                           m::ULt(m::Param("foo2"), m::Literal(UBits(10, 10))),
                           {m::Literal(UBits(10, 10))}, m::Param("foo2")));
}

TEST_F(NodeUtilTest, IsAncestorOf) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(100));
  BValue b = fb.Param("b", p->GetBitsType(100));
  BValue c = fb.Param("c", p->GetBitsType(100));
  BValue d = fb.Param("d", p->GetBitsType(100));
  BValue a_or_b = fb.Or({a, b});
  BValue a_or_b_or_d = fb.Or({a_or_b, d});
  BValue c_plus_d = fb.Add(c, d);
  BValue and_a_b_c = fb.And({a, b, c});
  BValue big_or = fb.Or({and_a_b_c, c_plus_d});

  EXPECT_FALSE(IsAncestorOf(a.node(), b.node()));
  EXPECT_FALSE(IsAncestorOf(c.node(), c.node()));
  EXPECT_FALSE(IsAncestorOf(d.node(), c.node()));

  EXPECT_TRUE(IsAncestorOf(a.node(), a_or_b.node()));
  EXPECT_TRUE(IsAncestorOf(b.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(c.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(d.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b_or_d.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(and_a_b_c.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(c_plus_d.node(), a_or_b.node()));
  EXPECT_FALSE(IsAncestorOf(big_or.node(), a_or_b.node()));

  EXPECT_TRUE(IsAncestorOf(a.node(), a_or_b_or_d.node()));
  EXPECT_TRUE(IsAncestorOf(b.node(), a_or_b_or_d.node()));
  EXPECT_FALSE(IsAncestorOf(c.node(), a_or_b_or_d.node()));
  EXPECT_TRUE(IsAncestorOf(d.node(), a_or_b_or_d.node()));
  EXPECT_TRUE(IsAncestorOf(a_or_b.node(), a_or_b_or_d.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b_or_d.node(), a_or_b_or_d.node()));
  EXPECT_FALSE(IsAncestorOf(and_a_b_c.node(), a_or_b_or_d.node()));
  EXPECT_FALSE(IsAncestorOf(c_plus_d.node(), a_or_b_or_d.node()));
  EXPECT_FALSE(IsAncestorOf(big_or.node(), a_or_b_or_d.node()));

  EXPECT_FALSE(IsAncestorOf(a.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(b.node(), c_plus_d.node()));
  EXPECT_TRUE(IsAncestorOf(c.node(), c_plus_d.node()));
  EXPECT_TRUE(IsAncestorOf(d.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b_or_d.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(and_a_b_c.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(c_plus_d.node(), c_plus_d.node()));
  EXPECT_FALSE(IsAncestorOf(big_or.node(), c_plus_d.node()));

  EXPECT_TRUE(IsAncestorOf(a.node(), and_a_b_c.node()));
  EXPECT_TRUE(IsAncestorOf(b.node(), and_a_b_c.node()));
  EXPECT_TRUE(IsAncestorOf(c.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(d.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b_or_d.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(and_a_b_c.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(c_plus_d.node(), and_a_b_c.node()));
  EXPECT_FALSE(IsAncestorOf(big_or.node(), and_a_b_c.node()));

  EXPECT_TRUE(IsAncestorOf(a.node(), big_or.node()));
  EXPECT_TRUE(IsAncestorOf(b.node(), big_or.node()));
  EXPECT_TRUE(IsAncestorOf(c.node(), big_or.node()));
  EXPECT_TRUE(IsAncestorOf(d.node(), big_or.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b.node(), big_or.node()));
  EXPECT_FALSE(IsAncestorOf(a_or_b_or_d.node(), big_or.node()));
  EXPECT_TRUE(IsAncestorOf(and_a_b_c.node(), big_or.node()));
  EXPECT_TRUE(IsAncestorOf(c_plus_d.node(), big_or.node()));
  EXPECT_FALSE(IsAncestorOf(big_or.node(), big_or.node()));
}

TEST_F(NodeUtilTest, RemoveNodeFromBooleanExpression) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue a_or_b = fb.Or({a, b});
  BValue a_or_b_or_d = fb.Or({a_or_b, d});
  BValue c_plus_d = fb.Add(c, d);
  BValue and_a_b_c = fb.And({a, b, c});
  BValue big_nor = fb.Nor({and_a_b_c, c_plus_d});

  EXPECT_THAT(RemoveNodeFromBooleanExpression(a.node(), a_or_b.node(), true),
              IsOkAndHolds(m::Or(m::Literal(1), m::Param("b"))));
  EXPECT_THAT(RemoveNodeFromBooleanExpression(a.node(), a_or_b.node(), false),
              IsOkAndHolds(m::Or(m::Literal(0), m::Param("b"))));
  EXPECT_THAT(RemoveNodeFromBooleanExpression(c.node(), a_or_b.node(), true),
              IsOkAndHolds(a_or_b.node()));
  EXPECT_THAT(RemoveNodeFromBooleanExpression(c.node(), a_or_b.node(), false),
              IsOkAndHolds(a_or_b.node()));

  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(b.node(), a_or_b_or_d.node(), true),
      IsOkAndHolds(m::Or(m::Or(m::Param("a"), m::Literal(1)), m::Param("d"))));
  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(b.node(), a_or_b_or_d.node(), false),
      IsOkAndHolds(m::Or(m::Or(m::Param("a"), m::Literal(0)), m::Param("d"))));

  EXPECT_THAT(RemoveNodeFromBooleanExpression(c.node(), c_plus_d.node(), true),
              IsOkAndHolds(m::Literal(1)));
  EXPECT_THAT(RemoveNodeFromBooleanExpression(c.node(), c_plus_d.node(), false),
              IsOkAndHolds(m::Literal(0)));

  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(a.node(), big_nor.node(), true),
      IsOkAndHolds(m::Nor(m::And(m::Literal(0), m::Param("b"), m::Param("c")),
                          m::Add(m::Param("c"), m::Param("d")))));
  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(a.node(), big_nor.node(), false),
      IsOkAndHolds(m::Nor(m::And(m::Literal(1), m::Param("b"), m::Param("c")),
                          m::Add(m::Param("c"), m::Param("d")))));

  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(b.node(), big_nor.node(), true),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Literal(0), m::Param("c")),
                          m::Add(m::Param("c"), m::Param("d")))));
  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(b.node(), big_nor.node(), false),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Literal(1), m::Param("c")),
                          m::Add(m::Param("c"), m::Param("d")))));

  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(c.node(), big_nor.node(), true),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Param("b"), m::Literal(0)),
                          m::Literal(0))));
  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(c.node(), big_nor.node(), false),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Param("b"), m::Literal(1)),
                          m::Literal(1))));

  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(d.node(), big_nor.node(), true),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Param("b"), m::Param("c")),
                          m::Literal(0))));
  EXPECT_THAT(
      RemoveNodeFromBooleanExpression(d.node(), big_nor.node(), false),
      IsOkAndHolds(m::Nor(m::And(m::Param("a"), m::Param("b"), m::Param("c")),
                          m::Literal(1))));
}

}  // namespace
}  // namespace xls
