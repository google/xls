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

#include "xls/ir/ir_matcher.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::HasSubstr;

template <typename M, typename T>
std::string Explain(const T& t, const M& m) {
  ::testing::StringMatchResultListener listener;
  EXPECT_THAT(t, ::testing::Not(m));  // For the error message.
  EXPECT_FALSE(m.MatchAndExplain(t, &listener));
  return listener.str();
}

TEST(IrMatchersTest, Basic) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  // Ensure we get the expected order of node evaluation by breaking out
  // statements, so that the ordinals are stable across compilers (C++ does not
  // define argument evaluation order).
  auto lhs = fb.Subtract(x, y);
  auto rhs = fb.Not(x);
  fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(x.node(), m::Param());
  EXPECT_THAT(x.node(), m::Name("x"));
  EXPECT_THAT(x.node(), m::Type("bits[32]"));
  EXPECT_THAT(x.node(), AllOf(m::Name("x"), m::Type("bits[32]")));

  EXPECT_THAT(y.node(), m::Param());
  EXPECT_THAT(y.node(), m::Name("y"));
  EXPECT_THAT(y.node(), m::Type("bits[32]"));

  EXPECT_THAT(f->return_value(), m::Add());
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(), m::Not()));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Sub(m::Param(), m::Param()), m::Not(m::Param())));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Sub(m::Name("x"), m::Name("y")), m::Not(m::Name("x"))));
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(_, m::Name("y")), _));

  EXPECT_THAT(Explain(x.node(), m::Name("z")),
              HasSubstr("has incorrect name, expected: z"));
  EXPECT_THAT(Explain(y.node(), m::Type("bits[123]")),
              HasSubstr("has incorrect type, expected: bits[123]"));
  EXPECT_THAT(Explain(y.node(), m::Add()),
              HasSubstr("has incorrect op (param), expected: add"));

  EXPECT_THAT(Explain(f->return_value(), m::Add(m::Param())),
              HasSubstr("has too many operands (got 2, want 1)"));
  EXPECT_THAT(Explain(f->return_value(), m::Add(m::Add(), _)),
              HasSubstr("has incorrect op (sub), expected: add"));
}

TEST(IrMatchersTest, BitSlice) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetBitsType(32));
  fb.BitSlice(x, /*start=*/7, /*width=*/9);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::BitSlice());
  EXPECT_THAT(f->return_value(), m::BitSlice(/*start=*/7, /*width=*/9));
  EXPECT_THAT(f->return_value(),
              m::BitSlice(m::Param(), /*start=*/7, /*width=*/9));
  EXPECT_THAT(f->return_value(),
              m::BitSlice(m::Name("x"), /*start=*/7, /*width=*/9));

  EXPECT_THAT(
      Explain(f->return_value(), m::BitSlice(/*start=*/7, /*width=*/42)),
      HasSubstr("has incorrect width, expected: 42"));
  EXPECT_THAT(
      Explain(f->return_value(), m::BitSlice(/*start=*/123, /*width=*/9)),
      HasSubstr("has incorrect start, expected: 123"));
}

TEST(IrMatchersTest, DynamicBitSlice) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetBitsType(32));
  auto start = fb.Param("y", p.GetBitsType(32));
  fb.DynamicBitSlice(x, start, /*width=*/5);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::DynamicBitSlice());
  EXPECT_THAT(f->return_value(),
              m::DynamicBitSlice(m::Param(), m::Param(), /*width=*/5));
  EXPECT_THAT(f->return_value(),
              m::DynamicBitSlice(m::Name("x"), m::Name("y"), /*width=*/5));

  EXPECT_THAT(Explain(f->return_value(),
                      m::DynamicBitSlice(m::Param(), m::Param(), /*width=*/42)),
              HasSubstr("has incorrect width, expected: 42"));
}

TEST(IrMatchersTest, Literal) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Literal(Value(UBits(0b1001, 4)));
  auto y = fb.Literal(Value(UBits(12345678, 32)));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(x.node(), m::Literal());
  EXPECT_THAT(x.node(), m::Literal("bits[4]: 0b1001"));
  EXPECT_THAT(x.node(), m::Literal(UBits(9, 4)));
  EXPECT_THAT(x.node(), m::Literal(Value(UBits(9, 4))));
  EXPECT_THAT(x.node(), m::Literal(9));

  EXPECT_THAT(y.node(), m::Literal());
  EXPECT_THAT(y.node(), m::Literal("bits[32]: 12345678"));
  EXPECT_THAT(y.node(), m::Literal(UBits(12345678, 32)));
  EXPECT_THAT(y.node(), m::Literal(Value(UBits(12345678, 32))));
  EXPECT_THAT(y.node(), m::Literal(12345678));

  EXPECT_THAT(Explain(x.node(), m::Literal(Value(UBits(9, 123)))),
              HasSubstr("has value bits[4]:9, expected: bits[123]:0x9"));
  EXPECT_THAT(Explain(x.node(), m::Literal(42)),
              HasSubstr("has value bits[4]:9, expected: bits[64]:42"));

  // When passing in a string for value matching, verify that the number format
  // of the input string is used in the error message.
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 0b1111")),
              HasSubstr("has value bits[4]:0b1001, expected: bits[4]:0b1111"));
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 0xf")),
              HasSubstr("has value bits[4]:0x9, expected: bits[4]:0xf"));
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 12")),
              HasSubstr("has value bits[4]:9, expected: bits[4]:12"));
}

TEST(IrMatchersTest, OneHotSelect) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(2));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.OneHotSelect(pred, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::OneHotSelect());
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(m::Name("pred"), {m::Name("x"), m::Name("y")}));
}

TEST(IrMatchersTest, Select) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(1));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.Select(pred, {x, y}, /*default_value=*/absl::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::Select());
  EXPECT_THAT(f->return_value(),
              m::Select(m::Name("pred"), {m::Name("x"), m::Name("y")}));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Name("pred"), {m::Name("x"), m::Name("y")},
                        /*default_value=*/absl::nullopt));

  EXPECT_THAT(
      Explain(f->return_value(), m::Select(m::Name("pred"), {m::Name("x")},
                                           /*default_value=*/m::Name("y"))),
      HasSubstr("has no default value, expected: y"));
}

TEST(IrMatchersTest, SelectWithDefault) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(2));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  auto z = fb.Param("z", p.GetBitsType(32));
  fb.Select(pred, {x, y}, /*default_value=*/z);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::Select());
  EXPECT_THAT(f->return_value(),
              m::Select(m::Name("pred"), {m::Name("x"), m::Name("y")},
                        /*default_value=*/m::Name("z")));

  EXPECT_THAT(Explain(f->return_value(),
                      m::Select(m::Name("pred"),
                                {m::Name("x"), m::Name("y"), m::Name("z")})),
              HasSubstr("has default value, expected no default value"));
}

TEST(IrMatchersTest, OneHot) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetBitsType(32));
  auto oh0 = fb.OneHot(x, LsbOrMsb::kLsb);
  auto oh1 = fb.OneHot(oh0, LsbOrMsb::kMsb);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(oh0.node(), m::OneHot());
  EXPECT_THAT(oh0.node(), m::OneHot(LsbOrMsb::kLsb));
  EXPECT_THAT(oh1.node(), m::OneHot(m::OneHot(), LsbOrMsb::kMsb));

  EXPECT_THAT(Explain(oh0.node(), m::OneHot(LsbOrMsb::kMsb)),
              HasSubstr("has incorrect priority, expected: lsb_prio=false"));
  EXPECT_THAT(Explain(oh1.node(), m::OneHot(LsbOrMsb::kLsb)),
              HasSubstr("has incorrect priority, expected: lsb_prio=true"));
}

TEST(IrMatchersTest, TupleIndex) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetTupleType({p.GetBitsType(32), p.GetBitsType(7)}));
  auto elem0 = fb.TupleIndex(x, 0);
  auto elem1 = fb.TupleIndex(x, 1);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(elem0.node(), m::TupleIndex());
  EXPECT_THAT(elem0.node(), m::TupleIndex(absl::optional<int64_t>(0)));
  EXPECT_THAT(elem0.node(), m::TupleIndex(m::Param(), 0));

  EXPECT_THAT(elem1.node(), m::TupleIndex());
  EXPECT_THAT(elem1.node(), m::TupleIndex(1));
  EXPECT_THAT(elem1.node(), m::TupleIndex(m::Param(), 1));

  EXPECT_THAT(Explain(elem0.node(), m::TupleIndex(1)),
              HasSubstr("has incorrect index, expected: 1"));
  EXPECT_THAT(Explain(elem1.node(), m::TupleIndex(400)),
              HasSubstr("has incorrect index, expected: 400"));
}

TEST(IrMatchersTest, ReductionOps) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto param = fb.Param("param", p.GetBitsType(8));
  auto or_reduce = fb.OrReduce(param);
  auto and_reduce = fb.AndReduce(param);
  auto xor_reduce = fb.XorReduce(param);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(or_reduce.node(), m::OrReduce(m::Name("param")));
  EXPECT_THAT(xor_reduce.node(), m::XorReduce(m::Name("param")));
  EXPECT_THAT(and_reduce.node(), m::AndReduce(m::Name("param")));
}

TEST(IrMatchersTest, SendOps) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch42,
      p.CreateStreamingChannel("ch42", ChannelOps ::kSendReceive,
                               p.GetBitsType(32), {}, FlowControl::kReadyValid,
                               ChannelMetadataProto(), 42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch123,
      p.CreateStreamingChannel("ch123", ChannelOps::kSendReceive,
                               p.GetBitsType(32), {}, FlowControl::kReadyValid,
                               ChannelMetadataProto(), 123));

  ProcBuilder b("proc", Value(UBits(333, 32)), "my_token", "my_state", &p);
  auto send = b.Send(ch42, b.GetTokenParam(), {b.GetStateParam()});
  auto send_if = b.SendIf(ch123, b.GetTokenParam(), b.Literal(UBits(1, 1)),
                          {b.GetStateParam()});
  XLS_ASSERT_OK(
      b.Build(b.AfterAll({send, send_if}), b.GetStateParam()).status());

  EXPECT_THAT(send.node(), m::Send());
  EXPECT_THAT(send.node(), m::Send(m::Channel(42)));
  EXPECT_THAT(send.node(), m::Send(m::Name("my_token"), {m::Name("my_state")},
                                   m::Channel(42)));

  EXPECT_THAT(send_if.node(), m::Send());
  EXPECT_THAT(send_if.node(), m::Send(m::Channel(123)));
  EXPECT_THAT(send_if.node(),
              m::Send(m::Name("my_token"), {m::Name("my_state")}, m::Literal(),
                      m::Channel(123)));
}

TEST(IrMatchersTest, ReceiveOps) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch42,
      p.CreateStreamingChannel("ch42", ChannelOps ::kSendReceive,
                               p.GetBitsType(32), {}, FlowControl::kReadyValid,
                               ChannelMetadataProto(), 42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch123,
      p.CreateStreamingChannel("ch123", ChannelOps::kSendReceive,
                               p.GetBitsType(32), {}, FlowControl::kReadyValid,
                               ChannelMetadataProto(), 123));

  ProcBuilder b("proc", Value(UBits(333, 32)), "my_token", "my_state", &p);
  auto receive = b.Receive(ch42, b.GetTokenParam());
  auto receive_if =
      b.ReceiveIf(ch123, b.GetTokenParam(), b.Literal(UBits(1, 1)));
  XLS_ASSERT_OK(b.Build(b.AfterAll({b.TupleIndex(receive, 0),
                                    b.TupleIndex(receive_if, 0)}),
                        b.GetStateParam())
                    .status());

  EXPECT_THAT(receive.node(), m::Receive());
  EXPECT_THAT(receive.node(), m::Receive(m::Channel(42)));
  EXPECT_THAT(receive.node(),
              m::Receive(m::Name("my_token"), m::Channel("ch42")));
  EXPECT_THAT(receive.node(), m::Receive(m::Name("my_token"),
                                         m::Channel(ChannelKind::kStreaming)));

  EXPECT_THAT(receive_if.node(), m::Receive());
  EXPECT_THAT(receive_if.node(), m::Receive(m::Channel(123)));
  EXPECT_THAT(receive_if.node(), m::Receive(m::Name("my_token"), m::Literal(),
                                            m::Channel("ch123")));
  EXPECT_THAT(receive_if.node(),
              m::Receive(m::Channel(ChannelKind::kStreaming)));

  // Mismatch conditions.
  EXPECT_THAT(Explain(receive.node(), m::Receive(m::Channel(444))),
              HasSubstr("has incorrect id (42), expected: 444"));
  EXPECT_THAT(Explain(receive.node(), m::Receive(m::Channel("foobar"))),
              HasSubstr("has incorrect name (ch42), expected: foobar"));
  EXPECT_THAT(
      Explain(receive.node(),
              m::Receive(m::Channel(ChannelKind::kSingleValue))),
      HasSubstr(" has incorrect kind (streaming), expected: single_value"));
}

TEST(IrMatchersTest, PortMatcher) {
  Package p("p");
  BlockBuilder bb("my_block", &p);
  Type* u32 = p.GetBitsType(32);
  BValue x = bb.InputPort("x", u32);
  BValue y = bb.InputPort("y", u32);
  BValue out = bb.OutputPort("out", bb.Add(x, y));

  EXPECT_THAT(x.node(), m::InputPort());
  EXPECT_THAT(x.node(), m::InputPort("x"));

  EXPECT_THAT(y.node(), m::InputPort());
  EXPECT_THAT(y.node(), m::InputPort("y"));

  EXPECT_THAT(out.node(), m::OutputPort());
  EXPECT_THAT(out.node(), m::OutputPort("out"));
  EXPECT_THAT(out.node(), m::OutputPort("out", m::Add()));

  // Check mismatch conditions.
  EXPECT_THAT(
      Explain(x.node(), m::OutputPort()),
      HasSubstr("has incorrect op (input_port), expected: output_port"));
  EXPECT_THAT(Explain(x.node(), m::InputPort("foobar")),
              HasSubstr("has incorrect name, expected: foobar"));
}

TEST(IrMatchersTest, RegisterMatcher) {
  Package p("p");
  BlockBuilder bb("my_block", &p);
  Type* u32 = p.GetBitsType(32);
  BValue in = bb.InputPort("in", u32);
  BValue in_d = bb.InsertRegister("reg", in);
  BValue out = bb.OutputPort("out", in_d);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  XLS_ASSERT_OK(bb.Build());

  EXPECT_THAT(in_d.node(), m::Register());
  EXPECT_THAT(in_d.node(), m::Register(m::InputPort()));
  EXPECT_THAT(in_d.node(), m::Register("reg"));
  EXPECT_THAT(in_d.node(), m::Register("reg", m::InputPort()));
  EXPECT_THAT(out.node(),
              m::OutputPort("out", m::Register(m::InputPort("in"))));

  // Check mismatch conditions.
  EXPECT_THAT(Explain(in_d.node(), m::Register(m::Add())),
              HasSubstr("has incorrect op (input_port), expected: add"));
  EXPECT_THAT(Explain(in_d.node(), m::Register("wrong-reg")),
              HasSubstr("has incorrect register (reg), expected: wrong-reg"));
  EXPECT_THAT(Explain(in_d.node(), m::Register("reg", m::Add())),
              HasSubstr("has incorrect op (input_port), expected: add"));
  EXPECT_THAT(Explain(in_d.node(), m::Register("wrong-reg", m::InputPort())),
              HasSubstr("has incorrect register (reg), expected: wrong-reg"));
}

}  // namespace
}  // namespace xls
