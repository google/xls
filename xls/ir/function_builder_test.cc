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

#include "xls/ir/function_builder.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/node_util.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace m = ::xls::op_matchers;

namespace xls {

using status_testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(FunctionBuilderTest, SimpleSourceLocation) {
  // Lineno/Colno are faked out here.
  const Lineno lineno(7);
  const Colno colno(11);

  Package p("p");
  SourceLocation loc = p.AddSourceLocation(__FILE__, lineno, colno);
  FunctionBuilder b("f", &p);

  Type* bits_32 = p.GetBitsType(32);
  auto x = b.Param("x", bits_32, SourceInfo(loc));

  ASSERT_EQ(x.node()->loc().locations.size(), 1);
  EXPECT_EQ(__FILE__ ":7",
            p.SourceLocationToString(x.node()->loc().locations[0]));
}

TEST(FunctionBuilderTest, CheckFilenameToFilenoLookup) {
  const std::string filename("fake_file.cc");
  Package p("p");

  // Verify two AddSourceLocation calls to the same filename result in correct
  // filename lookups.

  SourceLocation loc0 = p.AddSourceLocation(filename, Lineno(7), Colno(11));
  EXPECT_EQ("fake_file.cc:7", p.SourceLocationToString(loc0));

  SourceLocation loc1 = p.AddSourceLocation(filename, Lineno(8), Colno(12));
  EXPECT_EQ("fake_file.cc:8", p.SourceLocationToString(loc1));
}

TEST(FunctionBuilderTest, CheckUnknownFileToString) {
  Package p("p");
  SourceLocation loc(Fileno(1), Lineno(7), Colno(11));
  EXPECT_EQ("UNKNOWN:7", p.SourceLocationToString(loc));
}

TEST(FunctionBuilderTest, EmptyFunctionTest) {
  Package p("p");
  FunctionBuilder b("empty", &p);
  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Function cannot be empty")));
}

TEST(FunctionBuilderTest, LessThanTest) {
  Package p("p");
  FunctionBuilder b("lt", &p);
  BitsType* type = p.GetBitsType(33);
  b.ULt(b.Param("a", type), b.Param("b", type));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  Node* lt = func->return_value();
  EXPECT_EQ(lt->op(), Op::kULt);
  EXPECT_EQ(lt->GetType(), p.GetBitsType(1));
}

TEST(FunctionBuilderTest, NonRootReturnValue) {
  Package p("p");
  FunctionBuilder b("lt", &p);
  BitsType* type = p.GetBitsType(7);
  BValue and_node = b.And(b.Param("a", type), b.Param("b", type));
  b.Negate(and_node);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.BuildWithReturnValue(and_node));
  Node* return_value = func->return_value();
  EXPECT_EQ(return_value->op(), Op::kAnd);
}

TEST(FunctionBuilderTest, LiteralTupleTest) {
  Package p("p");
  FunctionBuilder b("literal_tuple", &p);
  BValue literal_node =
      b.Literal(Value::Tuple({Value(UBits(1, 2)), Value(UBits(3, 3))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func,
                           b.BuildWithReturnValue(literal_node));
  EXPECT_TRUE(func->GetType()->return_type()->IsTuple());
}

TEST(FunctionBuilderTest, NonTupleValueToTupleIndex) {
  Package p("p");
  FunctionBuilder b("tuple_index_test", &p);
  b.TupleIndex(b.Param("x", p.GetBitsType(32)), 0);
  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Operand of tuple-index must be tuple-typed")));
}

TEST(FunctionBuilderTest, LiteralArrayTest) {
  Package p("p");
  FunctionBuilder b("literal_array", &p);
  BValue literal_node =
      b.Literal(Value::ArrayOrDie({Value(UBits(1, 2)), Value(UBits(3, 2))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func,
                           b.BuildWithReturnValue(literal_node));
  EXPECT_TRUE(func->GetType()->return_type()->IsArray());
}

TEST(FunctionBuilderTest, MapTest) {
  Package p("p");
  const int kElementCount = 123;
  BitsType* element_type = p.GetBitsType(42);
  ArrayType* array_type = p.GetArrayType(kElementCount, element_type);
  Function* to_apply;
  {
    FunctionBuilder b("to_apply", &p);
    b.ULt(b.Param("element", element_type),
          b.Literal(Value(UBits(10, element_type->bit_count()))));
    XLS_ASSERT_OK_AND_ASSIGN(to_apply, b.Build());
  }
  Function* top;
  {
    FunctionBuilder b("top_f", &p);
    b.Map(b.Param("input", array_type), to_apply);
    XLS_ASSERT_OK_AND_ASSIGN(top, b.Build());
  }
  Node* map = top->return_value();
  EXPECT_EQ(map->op(), Op::kMap);
  EXPECT_EQ(to_apply->return_value()->GetType(), p.GetBitsType(1));
  EXPECT_EQ(map->GetType(),
            p.GetArrayType(kElementCount, to_apply->return_value()->GetType()));
}

TEST(FunctionBuilderTest, Match) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* cond_type = p.GetBitsType(8);
  BitsType* value_type = p.GetBitsType(32);
  BValue cond = b.Param("cond", cond_type);
  BValue x = b.Param("x", value_type);
  BValue y = b.Param("y", value_type);
  BValue z = b.Param("z", value_type);
  BValue the_default = b.Param("default", value_type);
  b.Match(cond,
          {{b.Literal(UBits(42, 8)), x},
           {b.Literal(UBits(123, 8)), y},
           {b.Literal(UBits(8, 8)), z}},
          the_default);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(
                  m::OneHot(m::Concat(m::Eq(m::Param("cond"), m::Literal(8)),
                                      m::Eq(m::Param("cond"), m::Literal(123)),
                                      m::Eq(m::Param("cond"), m::Literal(42)))),
                  /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z"),
                             m::Param("default")}));
}

TEST(FunctionBuilderTest, MatchTrue) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* pred_type = p.GetBitsType(1);
  BitsType* value_type = p.GetBitsType(32);
  BValue p0 = b.Param("p0", pred_type);
  BValue p1 = b.Param("p1", pred_type);
  BValue p2 = b.Param("p2", pred_type);
  BValue x = b.Param("x", value_type);
  BValue y = b.Param("y", value_type);
  BValue z = b.Param("z", value_type);
  BValue the_default = b.Param("default", value_type);
  b.MatchTrue({{p0, x}, {p1, y}, {p2, z}}, the_default);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(
          m::OneHot(m::Concat(m::Param("p2"), m::Param("p1"), m::Param("p0"))),
          /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z"),
                     m::Param("default")}));
}

// Note: for API consistency we allow the definition of MatchTrue to work when
// there are zero cases given and only a default.
TEST(FunctionBuilderTest, MatchTrueNoCaseArmsOnlyDefault) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* value_type = p.GetBitsType(32);
  BValue the_default = b.Param("default", value_type);
  b.MatchTrue({}, the_default);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(m::OneHot(),
                              /*cases=*/{m::Param("default")}));
}

TEST(FunctionBuilderTest, PrioritySelectOp) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* sel_type = p.GetBitsType(3);
  BitsType* value_type = p.GetBitsType(32);
  BValue sel = b.Param("sel", sel_type);
  BValue x = b.Param("x", value_type);
  BValue y = b.Param("y", value_type);
  BValue z = b.Param("z", value_type);
  b.PrioritySelect(sel, {x, y, z});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Param("sel"),
                  /*cases=*/{m::Param("x"), m::Param("y"), m::Param("z")}));
}

TEST(FunctionBuilderTest, ConcatTuples) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* value_type = p.GetBitsType(32);
  BValue x = b.Param("x", value_type);
  BValue t = b.Tuple({x});
  b.Concat({t, t});
  EXPECT_THAT(b.Build(), status_testing::StatusIs(
                             absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("it has non-bits type")));
}

TEST(FunctionBuilderTest, AfterAll) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BValue token_a = fb.AfterAll({});
  BValue token_b = fb.AfterAll({});
  fb.AfterAll({token_a, token_b});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), m::AfterAll(m::AfterAll(), m::AfterAll()));
}

TEST(FunctionBuilderTest, AfterAllNonTokenArg) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BValue token_a = fb.AfterAll({});
  BitsType* value_type = p.GetBitsType(32);
  BValue x = fb.Param("x", value_type);
  fb.AfterAll({token_a, x});

  EXPECT_THAT(
      fb.Build(),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Dependency type bits[32] is not a token.")));
}

TEST(FunctionBuilderTest, MinDelay) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BValue token = fb.AfterAll({});
  fb.MinDelay(token, /*delay=*/3);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), m::MinDelay(m::AfterAll(), /*delay=*/3));
}

TEST(FunctionBuilderTest, MinDelayZero) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BValue token = fb.AfterAll({});
  fb.MinDelay(token, /*delay=*/0);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), m::MinDelay(m::AfterAll(), /*delay=*/0));
}

TEST(FunctionBuilderTest, MinDelayNonTokenArg) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BitsType* value_type = p.GetBitsType(32);
  BValue x = fb.Param("x", value_type);
  fb.MinDelay(x, /*delay=*/1);

  EXPECT_THAT(fb.Build(),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Input type bits[32] is not a token.")));
}

TEST(FunctionBuilderTest, MinDelayNegative) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  BValue token = fb.AfterAll({});
  fb.MinDelay(token, /*delay=*/-5);

  EXPECT_THAT(fb.Build(), status_testing::StatusIs(
                              absl::StatusCode::kInvalidArgument,
                              testing::HasSubstr("Delay cannot be negative")));
}

TEST(FunctionBuilderTest, MinDelayNegativeWithGetError) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  XLS_EXPECT_OK(fb.GetError());

  BValue token = fb.AfterAll({});
  fb.MinDelay(token, /*delay=*/-5);

  EXPECT_THAT(fb.GetError(),
      status_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               testing::HasSubstr("Delay cannot be negative")));
}

TEST(FunctionBuilderTest, ArrayIndexBits) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* value_type = p.GetBitsType(32);
  BValue x = b.Param("x", value_type);
  b.ArrayIndex(x, {x});
  EXPECT_THAT(
      b.Build(),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Too many indices (1) to index into array of type bits[32]")));
}

TEST(FunctionBuilderTest, ArrayUpdateLiterals) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.ArrayUpdate(b.Literal(Value::UBitsArray({1, 2}, 32).value()),
                b.Literal(Value(UBits(99, 32))),
                {b.Literal(Value(UBits(0, 32)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(), m::Type("bits[32][2]"));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Literal("[bits[32]: 1, bits[32]: 2]"),
                             m::Literal("bits[32]: 99"),
                             /*indices=*/{m::Literal("bits[32]: 0")}));
}

TEST(FunctionBuilderTest, ArrayUpdateOnNonArray) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.ArrayUpdate(b.Literal(Value(UBits(1, 32))), b.Literal(Value(UBits(99, 32))),
                {b.Literal(Value(UBits(1, 32)))});
  EXPECT_THAT(
      b.Build(),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Too many indices (1) to index into array of type bits[32]")));
}

TEST(FunctionBuilderTest, ArrayUpdateIncompatibleUpdateValue) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.ArrayUpdate(
      b.Literal(Value::ArrayOrDie({Value(UBits(1, 32)), Value(UBits(2, 32))})),
      b.Literal(Value(UBits(99, 64))), {b.Literal(Value(UBits(1, 32)))});
  EXPECT_THAT(b.Build(),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Expected update value to have type "
                                     "bits[32]; has type bits[64]")));
}

TEST(FunctionBuilderTest, DynamicBitSlice) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* value_type = p.GetBitsType(32);
  BValue x = b.Param("x", value_type);
  BValue start = b.Param("start", value_type);
  b.DynamicBitSlice(x, start, 4);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              AllOf(m::DynamicBitSlice(), m::Type("bits[4]")));
}

TEST(FunctionBuilderTest, FullWidthDecode) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Decode(b.Param("x", p.GetBitsType(7)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Param()), m::Type("bits[128]")));
}

TEST(FunctionBuilderTest, NarrowedDecode) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Decode(b.Param("x", p.GetBitsType(7)), /*width=*/42);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Param()), m::Type("bits[42]")));
}

TEST(FunctionBuilderTest, OneBitDecode) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Decode(b.Param("x", p.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Decode(m::Param()), m::Type("bits[2]")));
}

TEST(FunctionBuilderTest, EncodePowerOfTwo) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Encode(b.Param("x", p.GetBitsType(256)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Encode(m::Param()), m::Type("bits[8]")));
}

TEST(FunctionBuilderTest, EncodeLessThanPowerOfTwo) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Encode(b.Param("x", p.GetBitsType(255)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Encode(m::Param()), m::Type("bits[8]")));
}

TEST(FunctionBuilderTest, EncodeGreaterThanPowerOfTwo) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Encode(b.Param("x", p.GetBitsType(257)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Encode(m::Param()), m::Type("bits[9]")));
}

TEST(FunctionBuilderTest, OneBitEncode) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Encode(b.Param("x", p.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(),
              AllOf(m::Encode(m::Param()), m::Type("bits[0]")));
}

TEST(FunctionBuilderTest, BuildTwiceFails) {
  Package p("p");
  FunctionBuilder b("lt", &p);
  BitsType* type = p.GetBitsType(33);
  b.ULt(b.Param("a", type), b.Param("b", type));

  XLS_EXPECT_OK(b.Build());
  absl::StatusOr<Function*> result = b.Build();

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kFailedPrecondition,
                               HasSubstr("multiple times")));
}

TEST(FunctionBuilderTest, SendAndReceive) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kSendReceive,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kSendReceive,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch2, p.CreateStreamingChannel("ch2", ChannelOps::kSendReceive,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch3, p.CreateStreamingChannel("ch3", ChannelOps::kSendReceive,
                                              p.GetBitsType(32)));

  ProcBuilder b("sending_receiving", /*token_name=*/"my_token", &p);
  BValue state = b.StateElement("my_state", Value(UBits(42, 32)));
  BValue send = b.Send(ch0, b.GetTokenParam(), state);
  BValue receive = b.Receive(ch1, b.GetTokenParam());
  BValue pred = b.Literal(UBits(1, 1));
  BValue send_if = b.SendIf(ch2, b.GetTokenParam(), pred, state);
  BValue receive_if = b.ReceiveIf(ch3, b.GetTokenParam(), pred);
  BValue after_all = b.AfterAll(
      {send, b.TupleIndex(receive, 0), send_if, b.TupleIndex(receive_if, 0)});
  BValue next_state =
      b.Add(b.TupleIndex(receive, 1), b.TupleIndex(receive_if, 1));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build(after_all, {next_state}));

  EXPECT_THAT(proc->NextToken(),
              m::AfterAll(m::Send(), m::TupleIndex(m::Receive()),
                          m::Send(m::Param(), m::Param(), m::Literal(1)),
                          m::TupleIndex(m::Receive())));
  EXPECT_THAT(proc->GetNextStateElement(0),
              m::Add(m::TupleIndex(m::Receive()), m::TupleIndex(m::Receive())));

  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "my_state");
  EXPECT_EQ(proc->TokenParam()->GetName(), "my_token");
  EXPECT_EQ(proc->GetStateElementType(0), p.GetBitsType(32));

  EXPECT_EQ(send.node()->GetType(), p.GetTokenType());
  EXPECT_EQ(send_if.node()->GetType(), p.GetTokenType());
  EXPECT_EQ(receive.node()->GetType(),
            p.GetTupleType({p.GetTokenType(), ch1->type()}));
  EXPECT_EQ(receive_if.node()->GetType(),
            p.GetTupleType({p.GetTokenType(), ch3->type()}));
}

TEST(FunctionBuilderTest, NonBlockingAndBlockingReceives) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0, p.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1, p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in2, p.CreateSingleValueChannel("in2", ChannelOps::kReceiveOnly,
                                                p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in3, p.CreateSingleValueChannel("in3", ChannelOps::kReceiveOnly,
                                                p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out0, p.CreateStreamingChannel("out0", ChannelOps::kSendOnly,
                                               p.GetBitsType(32)));

  TokenlessProcBuilder b("nb_and_b_receives", /*token_name=*/"tkn", &p);

  // Streaming blocking receive.
  BValue in0_data = b.Receive(in0);

  // Streaming non-blocking receive.
  auto [in1_data, in1_valid] = b.ReceiveNonBlocking(in1);

  // Single-value blocking receive (which will not block).
  BValue in2_data = b.Receive(in2);

  // Single-value non-blocking receive which will always return a valid.
  auto [in3_data, in3_valid] = b.ReceiveNonBlocking(in3);

  BValue sum = b.Add(in0_data, b.Add(in1_data, b.Add(in2_data, in3_data)));
  b.Send(out0, sum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build({}));

  // Check that the receive nodes are of the expected type.
  // Non-blocking nodes will have an extra bit for valid.
  EXPECT_EQ(in0_data.node()->GetType(), in0->type());
  EXPECT_EQ(in1_valid.node()->GetType(), p.GetBitsType(1));
  EXPECT_EQ(in2_data.node()->GetType(), in2->type());
  EXPECT_EQ(in3_data.node()->GetType(), in3->type());
  EXPECT_EQ(in3_valid.node()->GetType(), p.GetBitsType(1));

  // Check that the receive nodes are correctly built as either
  // blocking or non-blocking.
  for (Node* node : proc->nodes()) {
    if (!node->Is<Receive>()) {
      continue;
    }

    Receive* receive = node->As<Receive>();
    XLS_ASSERT_OK_AND_ASSIGN(Channel * channel, GetChannelUsedByNode(receive));

    if (channel == in0 || channel == in2) {
      EXPECT_TRUE(receive->is_blocking());
    } else {
      EXPECT_FALSE(receive->is_blocking());
    }
  }
}

TEST(FunctionBuilderTest, WrongAddOpMethodBinOp) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.AddBinOp(Op::kEq, b.Param("x", p.GetBitsType(32)),
             b.Param("y", p.GetBitsType(32)));
  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Op eq is not a operation of class BinOp")));
}

TEST(FunctionBuilderTest, WrongAddOpMethodUnaryOp) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.AddUnOp(Op::kAdd, b.Param("x", p.GetBitsType(32)));
  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Op add is not a operation of class UnOp")));
}

TEST(FunctionBuilderTest, WrongAddOpMethodCompareOp) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.AddCompareOp(Op::kNeg, b.Param("x", p.GetBitsType(32)),
                 b.Param("y", p.GetBitsType(32)));
  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Op neg is not a operation of class CompareOp")));
}

TEST(FunctionBuilderTest, WrongAddOpMethodNaryOp) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.AddNaryOp(Op::kNe, {b.Param("x", p.GetBitsType(32)),
                        b.Param("y", p.GetBitsType(32))});
  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Op ne is not a operation of class NaryOp")));
}

TEST(FunctionBuilderTest, NamedOps) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BitsType* type = p.GetBitsType(7);
  BValue and_node = b.And(b.Param("a", type), b.Param("b", type), SourceInfo(),
                          /*name=*/"foo");
  b.Negate(and_node, SourceInfo(), /*name=*/"bar");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  Node* return_value = func->return_value();
  EXPECT_EQ(return_value->op(), Op::kNeg);
  EXPECT_TRUE(return_value->HasAssignedName());
  EXPECT_EQ(return_value->GetName(), "bar");
  EXPECT_TRUE(return_value->operand(0)->HasAssignedName());
  EXPECT_EQ(return_value->operand(0)->GetName(), "foo");
}

TEST(FunctionBuilderTest, ArrayIndex) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetArrayType(42, p.GetBitsType(123)));
  BValue idx = b.Param("idx", p.GetBitsType(123));
  b.ArrayIndex(a, {idx});

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              m::ArrayIndex(m::Param("a"), /*indices=*/{m::Param("idx")}));
  EXPECT_EQ(func->return_value()->GetType(), p.GetBitsType(123));
}

TEST(FunctionBuilderTest, ArrayIndexMultipleDimensions) {
  Package p("p");
  FunctionBuilder b("f", &p);
  Type* element_type = p.GetArrayType(42, p.GetBitsType(123));
  BValue a =
      b.Param("a", p.GetArrayType(3333, p.GetArrayType(123, element_type)));
  BValue idx0 = b.Param("idx0", p.GetBitsType(123));
  BValue idx1 = b.Param("idx1", p.GetBitsType(1));
  b.ArrayIndex(a, {idx0, idx1});

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              m::ArrayIndex(m::Param("a"),
                            /*indices=*/{m::Param("idx0"), m::Param("idx1")}));
  EXPECT_EQ(func->return_value()->GetType(), element_type);
}

TEST(FunctionBuilderTest, ArrayIndexNilIndex) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetBitsType(123));
  b.ArrayIndex(a, {});

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              m::ArrayIndex(m::Param("a"), /*indices=*/{}));
  EXPECT_EQ(func->return_value()->GetType(), p.GetBitsType(123));
}

TEST(FunctionBuilderTest, ArrayIndexWrongIndexType) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetBitsType(123));
  BValue idx = b.Param("idx", p.GetTupleType({p.GetBitsType(123)}));
  b.ArrayIndex(a, {idx});

  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Indices to multi-array index operation must all be "
                         "bits types, index 0 is: (bits[123])")));
}

TEST(FunctionBuilderTest, ArrayIndexTooManyElementsInIndex) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetArrayType(42, p.GetBitsType(123)));
  BValue idx = b.Param("idx", p.GetBitsType(123));
  b.ArrayIndex(a, {idx, idx});

  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Too many indices (2) to index into array of "
                                 "type bits[123][42]")));
}

TEST(FunctionBuilderTest, ArrayUpdate) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetArrayType(42, p.GetBitsType(123)));
  BValue idx = b.Param("idx", p.GetBitsType(123));
  BValue value = b.Param("value", p.GetBitsType(123));
  b.ArrayUpdate(a, value, {idx});

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("value"),
                             /*indices=*/{m::Param("idx")}));
  EXPECT_EQ(func->return_value()->GetType(),
            p.GetArrayType(42, p.GetBitsType(123)));
}

TEST(FunctionBuilderTest, ArrayUpdateNilIndex) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetBitsType(123));
  BValue value = b.Param("value", p.GetBitsType(123));
  b.ArrayUpdate(a, value, {});

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, b.Build());
  EXPECT_THAT(func->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("value"), /*indices=*/{}));
  EXPECT_EQ(func->return_value()->GetType(), p.GetBitsType(123));
}

TEST(FunctionBuilderTest, ArrayUpdateTooManyElementsInIndex) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetArrayType(42, p.GetBitsType(123)));
  BValue idx = b.Param("idx", p.GetBitsType(123));
  BValue value = b.Param("value", p.GetBitsType(123));
  b.ArrayUpdate(a, value, {idx, idx});

  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Too many indices (2) to index into array of "
                                 "type bits[123][42]")));
}

TEST(FunctionBuilderTest, ArrayUpdateInvalidIndexType) {
  Package p("p");
  FunctionBuilder b("f", &p);
  BValue a = b.Param("a", p.GetArrayType(42, p.GetBitsType(123)));
  BValue value = b.Param("value", p.GetBitsType(123));
  BValue idx = b.Param("idx", p.GetTupleType({p.GetBitsType(123)}));
  b.ArrayUpdate(a, value, {idx});

  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Indices to multi-array update operation must all be "
                         "bits types, index 0 is: (bits[123])")));
}

TEST(FunctionBuilderTest, MultipleParametersWithSameName) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Param("idx", p.GetBitsType(123));
  b.Param("idx", p.GetBitsType(123));

  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parameter named \"idx\" already exists")));
}

TEST(FunctionBuilderTest, DynamicCountedForTest) {
  Package p("p");
  BitsType* int32_type = p.GetBitsType(32);
  BitsType* int16_type = p.GetBitsType(16);
  Function* body;
  {
    FunctionBuilder b("body", &p);
    auto index = b.Param("index", int32_type);
    auto accumulator = b.Param("accumulator", int32_type);
    b.Param("invariant_1", int16_type);
    b.Param("invariant_2", int16_type);
    b.Add(index, accumulator);
    XLS_ASSERT_OK_AND_ASSIGN(body, b.Build());
  }
  Function* top;
  {
    FunctionBuilder b("top_f", &p);
    b.DynamicCountedFor(b.Param("init", int32_type),
                        b.Param("trip_count", int16_type),
                        b.Param("stride", int16_type), body,
                        {b.Param("invariant_1", int16_type),
                         b.Param("invariant_2", int16_type)});
    XLS_ASSERT_OK_AND_ASSIGN(top, b.Build());
  }
  Node* dynamic_for = top->return_value();
  EXPECT_EQ(dynamic_for->op(), Op::kDynamicCountedFor);
  EXPECT_EQ(body->return_value()->GetType(), p.GetBitsType(32));
  EXPECT_THAT(dynamic_for,
              m::DynamicCountedFor(
                  m::Param("init"), m::Param("trip_count"), m::Param("stride"),
                  body, {m::Param("invariant_1"), m::Param("invariant_2")}));
}

TEST(FunctionBuilderTest, AddParamToProc) {
  Package p("p");
  ProcBuilder b("param_proc", /*token_name=*/"my_token", &p);
  BValue state = b.StateElement("my_state", Value(UBits(42, 32)));
  b.Param("x", p.GetBitsType(32));
  EXPECT_THAT(
      b.Build(b.GetTokenParam(), {state}).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Use StateElement to add state parameters to procs")));
}

TEST(FunctionBuilderTest, TokenlessProcBuilder) {
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      p.CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      p.CreateStreamingChannel("b", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue state = pb.StateElement("st", Value(UBits(42, 16)));
  BValue a_plus_b = pb.Add(pb.Receive(a_ch), pb.Receive(b_ch));
  pb.MinDelay(5);
  pb.Send(out_ch, pb.Add(state, a_plus_b));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({a_plus_b}));

  EXPECT_THAT(proc->NextToken(), m::Send(m::MinDelay(m::TupleIndex(m::Receive(
                                             m::TupleIndex(m::Receive())))),
                                         m::Add()));

  EXPECT_THAT(proc->GetNextStateElement(0),
              m::Add(m::TupleIndex(m::Receive(m::Channel("a")), 1),
                     m::TupleIndex(m::Receive(m::Channel("b")), 1)));
}

TEST(FunctionBuilderTest, StatelessProcBuilder) {
  Package p("p");
  ProcBuilder pb("the_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {}));
  EXPECT_TRUE(proc->StateParams().empty());
}

TEST(FunctionBuilderTest, ProcWithMultipleStateElements) {
  Package p("p");
  ProcBuilder pb("the_proc", "tkn", &p);
  BValue x = pb.StateElement("x", Value(UBits(1, 32)));
  BValue y = pb.StateElement("y", Value(UBits(2, 32)));
  BValue z = pb.StateElement("z", Value(UBits(3, 32)));

  EXPECT_EQ(pb.GetStateParam(0).node()->GetName(), "x");
  EXPECT_EQ(pb.GetStateParam(1).node()->GetName(), "y");
  EXPECT_EQ(pb.GetStateParam(2).node()->GetName(), "z");

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, pb.Build(pb.GetTokenParam(), /*next_state=*/{
                                x, pb.Add(x, y, SourceInfo(), "x_plus_y"), z}));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "y");
  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "z");
  EXPECT_THAT(proc->DumpIr(),
              HasSubstr("proc the_proc(tkn: token, x: bits[32], y: bits[32], "
                        "z: bits[32], init={1, 2, 3})"));
  EXPECT_EQ(proc->GetNextStateElement(0)->GetName(), "x");
  EXPECT_EQ(proc->GetNextStateElement(1)->GetName(), "x_plus_y");
  EXPECT_EQ(proc->GetNextStateElement(2)->GetName(), "z");
}

TEST(FunctionBuilderTest, ProcWithNextStateElement) {
  Package p("p");
  ProcBuilder pb("the_proc", "tkn", &p);
  BValue x = pb.StateElement("x", Value(UBits(1, 1)));
  BValue y = pb.StateElement("y", Value(UBits(2, 32)));
  BValue z = pb.StateElement("z", Value(UBits(3, 32)));
  BValue next = pb.Next(/*param=*/y, /*value=*/z, /*pred=*/x);

  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), /*next_state=*/{x, y, z}));
  EXPECT_THAT(next.node(), m::Next(m::Param("y"), /*value=*/m::Param("z"),
                                   /*predicate=*/m::Param("x")));
}

TEST(FunctionBuilderTest, ProcWithNextStateElementBadPredicate) {
  Package p("p");
  ProcBuilder pb("the_proc", "tkn", &p);
  BValue x = pb.StateElement("x", Value(UBits(1, 32)));
  BValue y = pb.StateElement("y", Value(UBits(2, 32)));
  BValue z = pb.StateElement("z", Value(UBits(3, 32)));
  pb.Next(/*param=*/y, /*value=*/z, /*pred=*/x);

  EXPECT_THAT(pb.Build(pb.GetTokenParam(),
                       /*next_state=*/{x, y, z}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Predicate operand"),
                             HasSubstr("must be of bits type of width 1"),
                             HasSubstr("is: bits[32]"))));
}

TEST(FunctionBuilderTest, TokenlessProcBuilderNoChannelOps) {
  Package p("p");
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue state = pb.StateElement("st", Value(UBits(42, 16)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({state}));

  EXPECT_THAT(proc->NextToken(), m::Param("tkn"));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("st"));
}

TEST(FunctionBuilderTest, Assert) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Assert(b.Param("tkn", p.GetTokenType()), b.Param("cond", p.GetBitsType(1)),
           /*message=*/"It's about sending a message");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(f->return_value(), m::Assert(m::Param("tkn"), m::Param("cond")));
  EXPECT_EQ(f->return_value()->As<Assert>()->message(),
            "It's about sending a message");
}

TEST(FunctionBuilderTest, AssertWrongTypeOperand0) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Assert(b.Param("blah", p.GetBitsType(42)),
           b.Param("cond", p.GetBitsType(1)),
           /*message=*/"It's about sending a message");
  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("First operand of assert must be of token type")));
}

TEST(FunctionBuilderTest, AssertWrongTypeOperand1) {
  Package p("p");
  FunctionBuilder b("f", &p);
  b.Assert(b.Param("blah", p.GetTokenType()), b.Param("cond", p.GetBitsType(2)),
           /*message=*/"It's about sending a message");
  EXPECT_THAT(
      b.Build().status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Condition operand of assert must be of bits type of width 1")));
}

TEST(FunctionBuilderTest, Trace) {
  Package p("p");
  FunctionBuilder b("f", &p);

  std::vector<FormatStep> format = {"x is ", FormatPreference::kPlainHex,
                                    " in hex"};
  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetTokenType()), b.Param("cond", p.GetBitsType(1)),
          {x}, "x is {:x} in hex");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  // TODO(amfv): 2021-08-02 Implement Trace matcher and add more tests of the
  // return value.
  EXPECT_EQ(OperandsExpectedByFormat(f->return_value()->As<Trace>()->format()),
            1);
  EXPECT_EQ(f->return_value()->As<Trace>()->format(), format);
}

TEST(FunctionBuilderTest, TraceWithVerbosity) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetTokenType()), b.Param("cond", p.GetBitsType(1)),
          {x}, "x is {}", /*verbosity=*/1);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_EQ(f->return_value()->As<Trace>()->verbosity(), 1);
}


TEST(FunctionBuilderTest, TraceWrongTypeOperand0) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetBitsType(23)), b.Param("cond", p.GetBitsType(1)),
          {x}, "x is {}");

  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("First operand of trace must be of token type")));
}

TEST(FunctionBuilderTest, TraceWrongTypeOperand1) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetTokenType()), x, {x}, "x is {}");

  EXPECT_THAT(
      b.Build().status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Condition operand of trace must be of bits type of width 1")));
}

TEST(FunctionBuilderTest, TraceWrongTypeTraceArg) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto tkn = b.Param("tkn", p.GetTokenType());

  b.Trace(tkn, b.Param("cond", p.GetBitsType(1)), {tkn}, "x is {}");

  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Trace arguments must be of bits type")));
}

TEST(FunctionBuilderTest, TraceWrongNumberOfArgs) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetTokenType()), b.Param("cond", p.GetBitsType(1)),
          {x, x}, "x is {}");

  EXPECT_THAT(
      b.Build().status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Trace node expects 1 data operands, but 2 were supplied")));
}

TEST(FunctionBuilderTest, TraceNegativeVerbosity) {
  Package p("p");
  FunctionBuilder b("f", &p);

  auto x = b.Param("x", p.GetBitsType(17));

  b.Trace(b.Param("tkn", p.GetTokenType()), b.Param("cond", p.GetBitsType(1)),
          {x}, "x is {}", /*verbosity=*/-1);

  EXPECT_THAT(b.Build().status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Trace verbosity must be >= 0, got -1")));
}

TEST(FunctionBuilderTest, NaryBitwiseXor) {
  Package p("p");
  BitsType* u1 = p.GetBitsType(1);
  FunctionBuilder fb("f", &p);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue c = fb.Param("c", u1);
  BValue nary_node = fb.Xor({a, b, c});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(nary_node));
  EXPECT_THAT(f->return_value(),
              m::Xor(m::Param("a"), m::Param("b"), m::Param("c")));
}

TEST(FunctionBuilderTest, NaryBitwiseOr) {
  Package p("p");
  BitsType* u1 = p.GetBitsType(1);
  FunctionBuilder fb("f", &p);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue c = fb.Param("c", u1);
  BValue nary_node = fb.Or({a, b, c});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(nary_node));
  EXPECT_THAT(f->return_value(),
              m::Or(m::Param("a"), m::Param("b"), m::Param("c")));
}

TEST(FunctionBuilderTest, NaryBitwiseAnd) {
  Package p("p");
  BitsType* u1 = p.GetBitsType(1);
  FunctionBuilder fb("f", &p);
  BValue a = fb.Param("a", u1);
  BValue b = fb.Param("b", u1);
  BValue c = fb.Param("c", u1);
  BValue nary_node = fb.And({a, b, c});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(nary_node));
  EXPECT_THAT(f->return_value(),
              m::And(m::Param("a"), m::Param("b"), m::Param("c")));
}

TEST(FunctionBuilderTest, Registers) {
  Package p("p");
  BlockBuilder b("b", &p);
  XLS_ASSERT_OK(b.block()->AddClockPort("clk"));

  BValue x = b.InputPort("x", p.GetBitsType(32));
  BValue rst = b.InputPort("rst", p.GetBitsType(1));
  BValue le = b.InputPort("le", p.GetBitsType(1));

  BValue x_1 = b.InsertRegister("x_1", x);
  BValue x_2 =
      b.InsertRegister("x_2", x, rst,
                       Reset{Value(UBits(42, 32)), /*asynchronous=*/false,
                             /*active_low=*/false});
  BValue x_3 =
      b.InsertRegister("x_3", x, rst,
                       Reset{Value(UBits(123, 32)), /*asynchronous=*/false,
                             /*active_low=*/true},
                       le);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  auto get_reg_write = [&](BValue reg_read) {
    return block
        ->GetRegisterWrite(reg_read.node()->As<RegisterRead>()->GetRegister())
        .value();
  };
  EXPECT_FALSE(get_reg_write(x_1)->reset().has_value());
  EXPECT_FALSE(get_reg_write(x_1)->load_enable().has_value());

  EXPECT_TRUE(get_reg_write(x_2)->reset().has_value());
  EXPECT_THAT(get_reg_write(x_2)->reset().value(), m::InputPort("rst"));
  EXPECT_FALSE(get_reg_write(x_2)->load_enable().has_value());

  EXPECT_TRUE(get_reg_write(x_3)->reset().has_value());
  EXPECT_THAT(get_reg_write(x_3)->reset().value(), m::InputPort("rst"));
  EXPECT_THAT(get_reg_write(x_3)->load_enable().value(), m::InputPort("le"));
}

}  // namespace xls
