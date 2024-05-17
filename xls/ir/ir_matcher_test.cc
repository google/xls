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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

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
  EXPECT_THAT(x.node(), m::Name(HasSubstr("x")));
  EXPECT_THAT(x.node(), m::Type("bits[32]"));
  EXPECT_THAT(x.node(), AllOf(m::Name("x"), m::Type("bits[32]")));

  EXPECT_THAT(y.node(), m::Param());
  EXPECT_THAT(y.node(), m::Name("y"));
  EXPECT_THAT(y.node(), m::Param("y"));
  EXPECT_THAT(y.node(), m::Param(HasSubstr("y")));
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
      HasSubstr("has incorrect width, expected width is equal to 42"));
  EXPECT_THAT(
      Explain(f->return_value(), m::BitSlice(/*start=*/123, /*width=*/9)),
      HasSubstr("has incorrect start, expected start is equal to 123"));
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

TEST(IrMatchersTest, PrioritySelect) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(2));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.PrioritySelect(pred, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::PrioritySelect());
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Name("pred"), {m::Name("x"), m::Name("y")}));
}

TEST(IrMatchersTest, OneHotSelectDoesNotMatchPrioritySelect) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(2));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.OneHotSelect(pred, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), ::testing::Not(m::PrioritySelect()));
}

TEST(IrMatchersTest, PrioritySelectDoesNotMatchOneHotSelect) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(2));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.PrioritySelect(pred, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), ::testing::Not(m::OneHotSelect()));
}

TEST(IrMatchersTest, Select) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto pred = fb.Param("pred", p.GetBitsType(1));
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  fb.Select(pred, {x, y}, /*default_value=*/std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(f->return_value(), m::Select());
  EXPECT_THAT(f->return_value(),
              m::Select(m::Name("pred"), {m::Name("x"), m::Name("y")}));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Name("pred"), {m::Name("x"), m::Name("y")},
                        /*default_value=*/std::nullopt));

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
  EXPECT_THAT(elem0.node(), m::TupleIndex(std::optional<int64_t>(0)));
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
      p.CreateStreamingChannel(
          "ch42", ChannelOps ::kSendReceive, p.GetBitsType(32), {},
          /*fifo_config=*/std::nullopt, FlowControl::kReadyValid,
          ChannelStrictness::kProvenMutuallyExclusive, ChannelMetadataProto(),
          42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch123,
      p.CreateStreamingChannel(
          "ch123", ChannelOps::kSendReceive, p.GetBitsType(32), {},
          /*fifo_config=*/std::nullopt, FlowControl::kReadyValid,
          ChannelStrictness::kProvenMutuallyExclusive, ChannelMetadataProto(),
          123));

  ProcBuilder b("test_proc", &p);
  auto my_token = b.StateElement("my_token", Value::Token());
  auto state = b.StateElement("my_state", Value(UBits(333, 32)));
  auto send = b.Send(ch42, my_token, state);
  auto send_if = b.SendIf(ch123, my_token, b.Literal(UBits(1, 1)), {state});
  XLS_ASSERT_OK(b.Build({b.AfterAll({send, send_if}), state}).status());

  EXPECT_THAT(send.node(), m::Send());
  EXPECT_THAT(send.node(), m::Send(m::Channel(42)));
  EXPECT_THAT(send.node(), m::Send(m::Name("my_token"), m::Name("my_state"),
                                   m::Channel(42)));
  EXPECT_THAT(send.node(), m::Send(m::ChannelWithType("bits[32]")));

  EXPECT_THAT(send_if.node(), m::Send());
  EXPECT_THAT(send_if.node(), m::Send(m::Channel(123)));
  EXPECT_THAT(send_if.node(), m::Send(m::Name("my_token"), m::Name("my_state"),
                                      m::Literal(), m::Channel(123)));
  EXPECT_THAT(send_if.node(), m::Send(m::ChannelWithType("bits[32]")));
}

TEST(IrMatchersTest, ReceiveOps) {
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch42,
      p.CreateStreamingChannel(
          "ch42", ChannelOps ::kSendReceive, p.GetBitsType(32), {},
          /*fifo_config=*/std::nullopt, FlowControl::kReadyValid,
          ChannelStrictness::kProvenMutuallyExclusive, ChannelMetadataProto(),
          42));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch123,
      p.CreateStreamingChannel(
          "ch123", ChannelOps::kSendReceive, p.GetBitsType(32), {},
          /*fifo_config=*/std::nullopt, FlowControl::kReadyValid,
          ChannelStrictness::kProvenMutuallyExclusive, ChannelMetadataProto(),
          123));

  ProcBuilder b("test_proc", &p);
  auto my_token = b.StateElement("my_token", Value::Token());
  auto state = b.StateElement("my_state", Value(UBits(333, 32)));
  auto receive = b.Receive(ch42, my_token);
  auto receive_if = b.ReceiveIf(ch123, my_token, b.Literal(UBits(1, 1)));
  XLS_ASSERT_OK(b.Build({b.AfterAll({b.TupleIndex(receive, 0),
                                     b.TupleIndex(receive_if, 0)}),
                         state})
                    .status());

  EXPECT_THAT(receive.node(), m::Receive());
  EXPECT_THAT(receive.node(), m::Receive(m::Channel(42)));
  EXPECT_THAT(receive.node(),
              m::Receive(m::Name("my_token"), m::Channel("ch42")));
  EXPECT_THAT(receive.node(), m::Receive(m::Name("my_token"),
                                         m::Channel(ChannelKind::kStreaming)));
  EXPECT_THAT(receive.node(), m::Receive(m::ChannelWithType("bits[32]")));

  EXPECT_THAT(receive_if.node(), m::Receive());
  EXPECT_THAT(receive_if.node(), m::Receive(m::Channel(123)));
  EXPECT_THAT(receive_if.node(), m::Receive(m::Name("my_token"), m::Literal(),
                                            m::Channel("ch123")));
  EXPECT_THAT(receive_if.node(),
              m::Receive(m::Channel(ChannelKind::kStreaming)));
  EXPECT_THAT(receive_if.node(), m::Receive(m::ChannelWithType("bits[32]")));

  // Mismatch conditions.
  EXPECT_THAT(Explain(receive.node(), m::Receive(m::Channel(444))),
              HasSubstr("has incorrect id (42), expected: 444"));
  EXPECT_THAT(Explain(receive.node(), m::Receive(m::Channel("foobar"))),
              HasSubstr("ch42 has incorrect name, expected: foobar."));
  EXPECT_THAT(
      Explain(receive.node(),
              m::Receive(m::Channel(ChannelKind::kSingleValue))),
      HasSubstr(" has incorrect kind (streaming), expected: single_value"));
  EXPECT_THAT(
      Explain(receive.node(), m::Receive(m::ChannelWithType("bits[64]"))),
      HasSubstr(" has incorrect type (bits[32]), expected: bits[64]"));
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
  EXPECT_THAT(x.node(), m::InputPort(HasSubstr("x")));

  EXPECT_THAT(y.node(), m::InputPort());
  EXPECT_THAT(y.node(), m::InputPort("y"));

  EXPECT_THAT(out.node(), m::OutputPort());
  EXPECT_THAT(out.node(), m::OutputPort("out"));
  EXPECT_THAT(out.node(), m::OutputPort("out", m::Add()));
  EXPECT_THAT(out.node(), m::OutputPortWithName(HasSubstr("out")));
  EXPECT_THAT(out.node(), m::OutputPort(HasSubstr("out"), m::Add()));

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
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg,
                           bb.block()->AddRegister("other_reg", u32));
  BValue x = bb.RegisterRead(reg);
  BValue y = bb.RegisterWrite(reg, x);
  BValue out = bb.OutputPort("out", in_d);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));

  XLS_ASSERT_OK(bb.Build());

  EXPECT_THAT(in_d.node(), m::Register());
  EXPECT_THAT(in_d.node(), m::Register(m::InputPort()));
  EXPECT_THAT(in_d.node(), m::Register("reg"));
  EXPECT_THAT(in_d.node(), m::RegisterWithName(HasSubstr("reg")));
  EXPECT_THAT(in_d.node(), m::Register("reg", m::InputPort()));
  EXPECT_THAT(in_d.node(), m::Register(HasSubstr("reg"), m::InputPort()));
  EXPECT_THAT(x.node(), m::RegisterRead());
  EXPECT_THAT(x.node(), m::RegisterRead("other_reg"));
  EXPECT_THAT(x.node(), m::RegisterRead(HasSubstr("other")));
  EXPECT_THAT(y.node(), m::RegisterWrite());
  EXPECT_THAT(y.node(), m::RegisterWrite("other_reg"));
  EXPECT_THAT(y.node(), m::RegisterWrite(HasSubstr("other")));
  EXPECT_THAT(y.node(), m::RegisterWrite("other_reg", x.node()));
  EXPECT_THAT(y.node(), m::RegisterWrite(HasSubstr("other"), x.node()));
  EXPECT_THAT(y.node(), m::RegisterWrite("other_reg", m::RegisterRead()));
  EXPECT_THAT(y.node(),
              m::RegisterWrite(HasSubstr("other"), m::RegisterRead()));
  EXPECT_THAT(out.node(),
              m::OutputPort("out", m::Register(m::InputPort("in"))));

  // Check mismatch conditions.
  EXPECT_THAT(Explain(in_d.node(), m::Register(m::Add())),
              HasSubstr("has incorrect op (input_port), expected: add"));
  EXPECT_THAT(Explain(in_d.node(), m::Register("wrong-reg")),
              HasSubstr("has incorrect register (reg), expected: wrong-reg"));
  EXPECT_THAT(Explain(in_d.node(), m::RegisterWithName(HasSubstr("wrong-reg"))),
              HasSubstr("has incorrect register (reg), expected: has substring "
                        "\"wrong-reg\""));
  EXPECT_THAT(Explain(in_d.node(), m::Register("reg", m::Add())),
              HasSubstr("has incorrect op (input_port), expected: add"));
  EXPECT_THAT(Explain(in_d.node(), m::Register("wrong-reg", m::InputPort())),
              HasSubstr("has incorrect register (reg), expected: wrong-reg"));
  EXPECT_THAT(
      Explain(in_d.node(), m::Register(HasSubstr("wrong-reg"), m::InputPort())),
      HasSubstr("has incorrect register (reg), expected: has substring "
                "\"wrong-reg\""));
}

TEST(IrMatchersTest, FunctionBaseMatcher) {
  Package p("p");
  FunctionBuilder fb("f", &p);
  auto x = fb.Param("x", p.GetBitsType(32));
  auto y = fb.Param("y", p.GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Add(x, y)));

  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p.CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly,
                               p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p.CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                               p.GetBitsType(32)));
  ProcBuilder pb("test_proc", &p);
  BValue tok = pb.StateElement("tok", Value::Token());
  BValue rcv = pb.Receive(ch0, tok);
  BValue rcv_token = pb.TupleIndex(rcv, 0);
  BValue rcv_data = pb.TupleIndex(rcv, 1);
  BValue f_of_data = pb.Invoke({rcv_data, rcv_data}, f);
  BValue send_token = pb.Send(ch1, rcv_token, f_of_data);
  XLS_ASSERT_OK(pb.Build({send_token}).status());

  BlockBuilder bb("test_block", &p);
  BValue a = bb.InputPort("x", p.GetBitsType(32));
  BValue b = bb.InputPort("y", p.GetBitsType(32));
  bb.OutputPort("out", bb.Add(a, b));
  XLS_ASSERT_OK(bb.Build());

  // Match FunctionBases.
  EXPECT_THAT(
      p.GetFunctionBases(),
      UnorderedElementsAre(m::FunctionBase("f"), m::FunctionBase("test_proc"),
                           m::FunctionBase("test_block")));
  EXPECT_THAT(p.GetFunctionBases(),
              UnorderedElementsAre(m::FunctionBase(HasSubstr("f")),
                                   m::FunctionBase(HasSubstr("test_pr")),
                                   m::FunctionBase(HasSubstr("test_b"))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::FunctionBase("foobar"))));

  // Match Function, Proc and Block.
  EXPECT_THAT(p.GetFunctionBases(),
              UnorderedElementsAre(m::Function("f"), m::Proc("test_proc"),
                                   m::Block("test_block")));
  EXPECT_THAT(p.GetFunctionBases(),
              UnorderedElementsAre(m::Function(HasSubstr("f")),
                                   m::Proc(HasSubstr("test_p")),
                                   m::Block(HasSubstr("test_b"))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::Function("test_proc"))));
  EXPECT_THAT(p.GetFunctionBases(),
              Not(Contains(m::Function(HasSubstr("proc")))));
  EXPECT_THAT(p.GetFunctionBases(),
              Not(Contains(m::Function(HasSubstr("block")))));
  EXPECT_THAT(p.GetFunctionBases(), ::testing::Not(Contains(m::Proc("f"))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::Proc(HasSubstr("f")))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::Proc(HasSubstr("block")))));
  EXPECT_THAT(p.GetFunctionBases(), ::testing::Not(Contains(m::Block("f"))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::Block(HasSubstr("f")))));
  EXPECT_THAT(p.GetFunctionBases(),
              ::testing::Not(Contains(m::Block(HasSubstr("proc")))));

  EXPECT_THAT(p.procs(), UnorderedElementsAre(m::Proc("test_proc")));
  EXPECT_THAT(p.functions(), UnorderedElementsAre(m::Function("f")));
  EXPECT_THAT(p.blocks(), UnorderedElementsAre(m::Block("test_block")));
}

TEST(IrMatchersTest, MinDelayMatcher) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto token = fb.AfterAll({});
  auto x = fb.MinDelay(token, 1);
  auto y = fb.MinDelay(token, 3);
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_THAT(x.node(), m::MinDelay());
  EXPECT_THAT(x.node(), m::MinDelay(token.node()));
  EXPECT_THAT(x.node(), m::MinDelay(token.node(), 1));
  EXPECT_THAT(x.node(), m::MinDelay(1));

  EXPECT_THAT(y.node(), m::MinDelay());
  EXPECT_THAT(y.node(), m::MinDelay(token.node()));
  EXPECT_THAT(y.node(), m::MinDelay(token.node(), 3));
  EXPECT_THAT(y.node(), m::MinDelay(3));

  EXPECT_THAT(Explain(x.node(), m::MinDelay(3)),
              HasSubstr("delay 1 isn't equal to 3"));
  EXPECT_THAT(Explain(x.node(), m::MinDelay(token.node(), 3)),
              HasSubstr("delay 1 isn't equal to 3"));

  EXPECT_THAT(Explain(y.node(), m::MinDelay(0)),
              HasSubstr("delay 3 isn't equal to 0"));
  EXPECT_THAT(Explain(y.node(), m::MinDelay(token.node(), 0)),
              HasSubstr("delay 3 isn't equal to 0"));
}

TEST(IrMatchersTest, NameMatcher) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto xy = fb.Param("xy____", p.GetBitsType(32));
  auto yx = fb.Param("____yx", p.GetBitsType(32));
  auto sum = fb.Add(xy, yx);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sum));

  // Tests that matchers can be passed to m::Name().
  EXPECT_THAT(f->return_value(),
              m::Add(m::Name(HasSubstr("xy")), m::Name(HasSubstr("yx"))));
}

// Make and return a block which adds two u32 numbers.
absl::StatusOr<Block*> MakeAddBlock(std::string_view name, Package* package) {
  Type* u32 = package->GetBitsType(32);
  BlockBuilder bb(name, package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("result", bb.Add(a, b));
  return bb.Build();
}

TEST(IrMatchersTest, InstantiationMatcher) {
  Package p("p");
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(Block * add_block, MakeAddBlock("adder", &p));
  BlockBuilder bb("my_block", &p);
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * add0,
      bb.block()->AddBlockInstantiation("add0", add_block));
  BValue x = bb.InputPort("x", u32);
  BValue y = bb.InputPort("y", u32);

  BValue a = bb.InstantiationInput(add0, "a", x);
  bb.InstantiationInput(add0, "b", y);
  BValue x_plus_y = bb.InstantiationOutput(add0, "result");
  bb.OutputPort("x_plus_y", x_plus_y);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(m::Instantiation("add0")));
  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(
                  m::Instantiation("add0", InstantiationKind::kBlock)));
  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(m::Instantiation(InstantiationKind::kBlock)));

  EXPECT_THAT(
      Explain(block->GetInstantiations().at(0), m::Instantiation("sub0")),
      HasSubstr("add0"));
  EXPECT_THAT(Explain(block->GetInstantiations().at(0),
                      m::Instantiation("sub0", InstantiationKind::kExtern)),
              HasSubstr("add0 has incorrect name, expected: sub0"));
  EXPECT_THAT(Explain(block->GetInstantiations().at(0),
                      m::Instantiation(InstantiationKind::kExtern)),
              HasSubstr("add0 has incorrect kind, expected: extern"));

  EXPECT_THAT(
      block->nodes(),
      AllOf(
          Contains(m::InstantiationOutput()),
          Contains(m::InstantiationOutput("result")),
          Contains(m::InstantiationOutput(HasSubstr("res"))),
          Contains(m::InstantiationOutput("result", m::Instantiation("add0"))),
          Contains(m::InstantiationInput(m::InputPort("x"))),
          Contains(m::InstantiationInput(m::InputPort(HasSubstr("x")))),
          Contains(m::InstantiationInput(m::InputPort("x"), "a")),
          Contains(m::InstantiationInput(m::InputPort("x"), "a",
                                         m::Instantiation("add0")))));
  EXPECT_THAT(a.node(), ::testing::Not(m::InstantiationInput(
                            m::InputPort("x"), HasSubstr("b"),
                            m::Instantiation("add0"))));

  EXPECT_THAT(Explain(a.node(), m::InstantiationInput(m::InputPort("y"))),
              HasSubstr("x has incorrect name, expected: y."));
  EXPECT_THAT(Explain(a.node(), m::InstantiationInput(m::InputPort("x"), "b")),
              HasSubstr("a has incorrect name, expected: b."));
  EXPECT_THAT(
      Explain(a.node(), m::InstantiationInput(m::InputPort("x"), "a",
                                              m::Instantiation("add1"))),
      HasSubstr("add0 has incorrect name, expected: add1."));
}

}  // namespace
}  // namespace xls
