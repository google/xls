// Copyright 2020 Google LLC
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
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Eq;

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
  EXPECT_THAT(x.node(), m::Param("x"));
  EXPECT_THAT(x.node(), m::Type("bits[32]"));
  EXPECT_THAT(x.node(), AllOf(m::Param("x"), m::Type("bits[32]")));

  EXPECT_THAT(y.node(), m::Param());
  EXPECT_THAT(y.node(), m::Param("y"));
  EXPECT_THAT(y.node(), m::Type("bits[32]"));

  EXPECT_THAT(f->return_value(), m::Add());
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(), m::Not()));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Sub(m::Param(), m::Param()), m::Not(m::Param())));
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(m::Param("x"), m::Param("y")),
                                        m::Not(m::Param("x"))));
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(_, m::Param("y")), _));

  EXPECT_THAT(Explain(x.node(), m::Param("z")),
              Eq("x: bits[32] = param(x) has incorrect name, expected: z"));
  EXPECT_THAT(
      Explain(y.node(), m::Type("bits[123]")),
      Eq("y: bits[32] = param(y) has incorrect type, expected: bits[123]"));
  EXPECT_THAT(Explain(y.node(), m::Add()),
              Eq("y: bits[32] = param(y) has incorrect op, expected: add"));

  EXPECT_THAT(Explain(f->return_value(), m::Add(m::Param())),
              Eq("add.5: bits[32] = add(sub.3, not.4) has too many operands "
                 "(got 2, want 1)"));
  EXPECT_THAT(Explain(f->return_value(), m::Add(m::Add(), _)),
              Eq("add.5: bits[32] = add(sub.3, not.4)\noperand 0:\n\tsub.3: "
                 "bits[32] = sub(x, y)\ndoesn't match expected:\n\tadd, sub.3: "
                 "bits[32] = sub(x, y) has incorrect op, expected: add"));
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
              m::BitSlice(m::Param("x"), /*start=*/7, /*width=*/9));

  EXPECT_THAT(
      Explain(f->return_value(), m::BitSlice(/*start=*/7, /*width=*/42)),
      Eq("bit_slice.2: bits[9] = bit_slice(x, start=7, width=9) has incorrect "
         "width, expected: 42"));
  EXPECT_THAT(
      Explain(f->return_value(), m::BitSlice(/*start=*/123, /*width=*/9)),
      Eq("bit_slice.2: bits[9] = bit_slice(x, start=7, width=9) has incorrect "
         "start, expected: 123"));
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
              m::DynamicBitSlice(m::Param("x"), m::Param("y"), /*width=*/5));

  EXPECT_THAT(Explain(f->return_value(),
                      m::DynamicBitSlice(m::Param(), m::Param(), /*width=*/42)),
              Eq("dynamic_bit_slice.3: bits[5] = dynamic_bit_slice(x, y, "
                 "width=5) has incorrect "
                 "width, expected: 42"));
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
              Eq("literal.1: bits[4] = literal(value=9) has value bits[4]:9, "
                 "expected: bits[123]:0x9"));
  EXPECT_THAT(Explain(x.node(), m::Literal(42)),
              Eq("literal.1: bits[4] = literal(value=9) has value bits[4]:9, "
                 "expected: bits[64]:42"));

  // When passing in a string for value matching, verify that the number format
  // of the input string is used in the error message.
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 0b1111")),
              Eq("literal.1: bits[4] = literal(value=9) has value "
                 "bits[4]:0b1001, expected: bits[4]:0b1111"));
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 0xf")),
              Eq("literal.1: bits[4] = literal(value=9) has value bits[4]:0x9, "
                 "expected: bits[4]:0xf"));
  EXPECT_THAT(Explain(x.node(), m::Literal("bits[4]: 12")),
              Eq("literal.1: bits[4] = literal(value=9) has value bits[4]:9, "
                 "expected: bits[4]:12"));
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
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Param("pred"), {m::Param("x"), m::Param("y")}));
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
              m::Select(m::Param("pred"), {m::Param("x"), m::Param("y")}));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("pred"), {m::Param("x"), m::Param("y")},
                        /*default_value=*/absl::nullopt));

  EXPECT_THAT(
      Explain(f->return_value(), m::Select(m::Param("pred"), {m::Param("x")},
                                           /*default_value=*/m::Param("y"))),
      Eq("sel.4: bits[32] = sel(pred, cases=[x, y]) has no default value, "
         "expected: param"));
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
              m::Select(m::Param("pred"), {m::Param("x"), m::Param("y")},
                        /*default_value=*/m::Param("z")));

  EXPECT_THAT(
      Explain(f->return_value(),
              m::Select(m::Param("pred"),
                        {m::Param("x"), m::Param("y"), m::Param("z")})),
      Eq("sel.5: bits[32] = sel(pred, cases=[x, y], default=z) has default "
         "value, expected no default value"));
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
              Eq("one_hot.2: bits[33] = one_hot(x, lsb_prio=true) has "
                 "incorrect priority, expected: lsb_prio=false"));
  EXPECT_THAT(Explain(oh1.node(), m::OneHot(LsbOrMsb::kLsb)),
              Eq("one_hot.3: bits[34] = one_hot(one_hot.2, lsb_prio=false) has "
                 "incorrect priority, expected: lsb_prio=true"));
}

TEST(IrMatchersTest, TupleIndex) {
  Package p("p");
  FunctionBuilder fb("f", &p);

  auto x = fb.Param("x", p.GetTupleType({p.GetBitsType(32), p.GetBitsType(7)}));
  auto elem0 = fb.TupleIndex(x, 0);
  auto elem1 = fb.TupleIndex(x, 1);
  XLS_ASSERT_OK(fb.Build().status());
  EXPECT_THAT(elem0.node(), m::TupleIndex());
  EXPECT_THAT(elem0.node(), m::TupleIndex(absl::optional<int64>(0)));
  EXPECT_THAT(elem0.node(), m::TupleIndex(m::Param(), 0));

  EXPECT_THAT(elem1.node(), m::TupleIndex());
  EXPECT_THAT(elem1.node(), m::TupleIndex(1));
  EXPECT_THAT(elem1.node(), m::TupleIndex(m::Param(), 1));

  EXPECT_THAT(Explain(elem0.node(), m::TupleIndex(1)),
              Eq("tuple_index.2: bits[32] = tuple_index(x, index=0) has "
                 "incorrect index, expected: 1"));
  EXPECT_THAT(Explain(elem1.node(), m::TupleIndex(400)),
              Eq("tuple_index.3: bits[7] = tuple_index(x, index=1) has "
                 "incorrect index, expected: 400"));
}

}  // namespace
}  // namespace xls
