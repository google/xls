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

#ifndef XLS_IR_IR_MATCHER_H_
#define XLS_IR_IR_MATCHER_H_

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace op_matchers {

// Implements matching over XLS IR. Enables easy pattern matching of XLS
// expressions in tests.
//
// Example usage which EXPECTs the return value of the function 'f' to be a
// BitSlice of a parameter with the given start and width values:
//
//    EXPECT_THAT(f->return_value(),
//                m::BitSlice(m::Param(), /*start=*/3, /*width=*/8));
//
// Matchers can be combined with ::testing::AllOf like in the example below
// which expects the bit-slice operand to be a param node of the given type:
//
//    using ::testing::AllOf;
//    ...
//    EXPECT_THAT(f->return_value(),
//                m::BitSlice(AllOf(m::Param(), m::Type("bits[7]")),
//                            /*start=*/3, /*width=*/8));
//
// Nodes can also be matched by name. Here, "foo", "bar", and "baz" are
// names of nodes in the function.
//
//    EXPECT_THAT(f->return_value(),
//                m::Or(m::Name("foo"), m::Name("bar"), m::Name("baz")));

// Base class for matchers. Has two constructions. The first checks the op and
// then recursively checks the operands. The second simply matches the name.
class NodeMatcher : public ::testing::MatcherInterface<const Node*> {
 public:
  NodeMatcher(Op op, std::vector<::testing::Matcher<const Node*>> operands)
      : op_(op), operands_(operands) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 private:
  Op op_;
  std::vector<::testing::Matcher<const Node*>> operands_;
};

// Class for matching XLS Types. Example usage:
//
//   EXPECT_THAT(foo, m::Type("bits[32]"));
//   EXPECT_THAT(foo, m::Type(package->GetBitsType(32)));
class TypeMatcher : public ::testing::MatcherInterface<const Node*> {
 public:
  explicit TypeMatcher(absl::string_view type_str) : type_str_(type_str) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::string type_str_;
};

inline ::testing::Matcher<const ::xls::Node*> Type(const Type* type) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::TypeMatcher(type->ToString()));
}

inline ::testing::Matcher<const ::xls::Node*> Type(const char* type_str) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::TypeMatcher(type_str));
}

// Class for matching node names. Example usage:
//
//   EXPECT_THAT(baz, m::Or(m::Name("foo"), m::Name("bar"))
//
// TODO(meheff): Through template wizardry it'd probably be possible to elide
// the m::Name. For example: EXPECT_THAT(baz, m::Or("foo", "bar")).
class NameMatcher : public ::testing::MatcherInterface<const Node*> {
 public:
  explicit NameMatcher(absl::string_view name) : name_(name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(std::ostream* os) const override;

 private:
  std::string name_;
};

inline ::testing::Matcher<const ::xls::Node*> Name(absl::string_view name_str) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::NameMatcher(name_str));
}

// Node* matchers for ops which have no metadata beyond Op, type, and operands.
#define NODE_MATCHER(op)                                                       \
  template <typename... M>                                                     \
  ::testing::Matcher<const ::xls::Node*> op(M... operands) {                   \
    return ::testing::MakeMatcher(                                             \
        new ::xls::op_matchers::NodeMatcher(::xls::Op::k##op, {operands...})); \
  }
NODE_MATCHER(Add);
NODE_MATCHER(AfterAll);
NODE_MATCHER(And);
NODE_MATCHER(AndReduce);
NODE_MATCHER(Array);
NODE_MATCHER(ArrayIndex);
NODE_MATCHER(ArrayUpdate);
NODE_MATCHER(Concat);
NODE_MATCHER(Decode);
NODE_MATCHER(Encode);
NODE_MATCHER(Eq);
NODE_MATCHER(Identity);
NODE_MATCHER(Nand);
NODE_MATCHER(Ne);
NODE_MATCHER(Neg);
NODE_MATCHER(Nor);
NODE_MATCHER(Not);
NODE_MATCHER(Or);
NODE_MATCHER(OrReduce);
NODE_MATCHER(Reverse);
NODE_MATCHER(SDiv);
NODE_MATCHER(SGe);
NODE_MATCHER(SGt);
NODE_MATCHER(SLe);
NODE_MATCHER(SLt);
NODE_MATCHER(SMod);
NODE_MATCHER(SMul);
NODE_MATCHER(Sel);
NODE_MATCHER(Shll);
NODE_MATCHER(Shra);
NODE_MATCHER(Shrl);
NODE_MATCHER(SignExt);
NODE_MATCHER(Sub);
NODE_MATCHER(Tuple);
NODE_MATCHER(UDiv);
NODE_MATCHER(UGe);
NODE_MATCHER(UGt);
NODE_MATCHER(ULe);
NODE_MATCHER(ULt);
NODE_MATCHER(UMod);
NODE_MATCHER(UMul);
NODE_MATCHER(Xor);
NODE_MATCHER(XorReduce);
NODE_MATCHER(ZeroExt);

// TODO(meheff): The following ops should have custom matchers defined as they
// have additional metadata.
NODE_MATCHER(CountedFor);
NODE_MATCHER(Invoke);
NODE_MATCHER(Map);
#undef NODE_MATCHER

// Ops which have metadata beyond the Op, type, and the operands (e.g., Literals
// which have values) require their own subclass of NodeMatcher. Below are the
// definitions of these classes.

// Param matcher. Matches parameter name only. Supported forms:
//
//   EXPECT_THAT(x, m::Param());
//   EXPECT_THAT(x, m::Param("x"));
class ParamMatcher : public NodeMatcher {
 public:
  explicit ParamMatcher(absl::optional<std::string> name)
      : NodeMatcher(Op::kParam, /*operands=*/{}), name_(name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<std::string> name_;
};

inline ::testing::Matcher<const ::xls::Node*> Param(
    absl::optional<std::string> name) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ParamMatcher(name));
}

inline ::testing::Matcher<const ::xls::Node*> Param() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kParam, {}));
}

// BitSlice matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::BitSlice());
//   EXPECT_THAT(foo, op::BitSlice(op::Param()));
//   EXPECT_THAT(foo, op::BitSlice(/*start=*/7, /*width=*/8));
//   EXPECT_THAT(foo, op::BitSlice(/*operand=*/op::Param(), /*start=*/7,
//                                 /*width=*/8));
class BitSliceMatcher : public NodeMatcher {
 public:
  BitSliceMatcher(::testing::Matcher<const Node*> operand,
                  absl::optional<int64> start, absl::optional<int64> width)
      : NodeMatcher(Op::kBitSlice, {operand}), start_(start), width_(width) {}
  BitSliceMatcher(absl::optional<int64> start, absl::optional<int64> width)
      : NodeMatcher(Op::kBitSlice, {}), start_(start), width_(width) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> start_;
  absl::optional<int64> width_;
};

inline ::testing::Matcher<const ::xls::Node*> BitSlice() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::BitSliceMatcher(absl::nullopt, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(
    ::testing::Matcher<const Node*> operand) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::BitSliceMatcher(
      operand, absl::nullopt, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(
    ::testing::Matcher<const Node*> operand, int64 start, int64 width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::BitSliceMatcher(operand, start, width));
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(int64 start,
                                                       int64 width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::BitSliceMatcher(start, width));
}

// DynamicBitSlice matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::DynamicBitSlice());
//   EXPECT_THAT(foo, op::DynamicBitSlice(op::Param(), op::Param()));
//   EXPECT_THAT(foo, op::DynamicBitSlice(/*operand=*/op::Param(),
//                                        /*start=*/op::Param(), /*width=*/8));
class DynamicBitSliceMatcher : public NodeMatcher {
 public:
  DynamicBitSliceMatcher(::testing::Matcher<const Node*> operand,
                         ::testing::Matcher<const Node*> start,
                         absl::optional<int64> width)
      : NodeMatcher(Op::kDynamicBitSlice, {operand, start}), width_(width) {}
  DynamicBitSliceMatcher() : NodeMatcher(Op::kDynamicBitSlice, {}) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> width_;
};

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::DynamicBitSliceMatcher());
}

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice(
    ::testing::Matcher<const Node*> operand,
    ::testing::Matcher<const Node*> start) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::DynamicBitSliceMatcher(
      operand, start, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice(
    ::testing::Matcher<const Node*> operand,
    ::testing::Matcher<const Node*> start, int64 width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::DynamicBitSliceMatcher(operand, start, width));
}

// Literal matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::Literal());
//   EXPECT_THAT(foo, op::Literal(Value(UBits(7, 8))));
//   EXPECT_THAT(foo, op::Literal(UBits(7, 8)));
//   EXPECT_THAT(foo, op::Literal(42));
//   EXPECT_THAT(foo, op::Literal("bits[8]:7"));
//   EXPECT_THAT(foo, op::Literal("bits[8]:0x7"));
//   EXPECT_THAT(foo, op::Literal("bits[8]:0b111"));
class LiteralMatcher : public NodeMatcher {
 public:
  explicit LiteralMatcher(FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), format_(format) {}
  explicit LiteralMatcher(absl::optional<Value> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), value_(value), format_(format) {}
  explicit LiteralMatcher(absl::optional<int64> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), uint64_value_(value), format_(format) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  // At most one of the optional data members has a value.
  absl::optional<Value> value_;
  absl::optional<uint64> uint64_value_;
  FormatPreference format_;
};

inline ::testing::Matcher<const ::xls::Node*> Literal() {
  return ::testing::MakeMatcher(new ::xls::op_matchers::LiteralMatcher());
}

inline ::testing::Matcher<const ::xls::Node*> Literal(const Value& value) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::LiteralMatcher(value));
}

inline ::testing::Matcher<const ::xls::Node*> Literal(const Bits& bits) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::LiteralMatcher(Value(bits)));
}
inline ::testing::Matcher<const ::xls::Node*> Literal(uint64 value) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::LiteralMatcher(value));
}

inline ::testing::Matcher<const ::xls::Node*> Literal(
    absl::string_view value_str) {
  Value value = Parser::ParseTypedValue(value_str).value();
  FormatPreference format = FormatPreference::kDefault;
  if (absl::StrContains(value_str, "0b")) {
    format = FormatPreference::kBinary;
  } else if (absl::StrContains(value_str, "0x")) {
    format = FormatPreference::kHex;
  }
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::LiteralMatcher(value, format));
}

// OneHot matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::OneHot());
//   EXPECT_THAT(foo, op::OneHot(op::Param()));
//   EXPECT_THAT(foo, op::OneHot(/*priority=*/LsbOrMsb::kLsb));
//   EXPECT_THAT(foo, op::OneHot(op::Param(), /*priority=*/LsbOrMsb::kLsb));
class OneHotMatcher : public NodeMatcher {
 public:
  explicit OneHotMatcher(::testing::Matcher<const Node*> operand,
                         absl::optional<LsbOrMsb> priority = absl::nullopt)
      : NodeMatcher(Op::kOneHot, {operand}), priority_(priority) {}
  explicit OneHotMatcher(absl::optional<LsbOrMsb> priority = absl::nullopt)
      : NodeMatcher(Op::kOneHot, {}), priority_(priority) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<LsbOrMsb> priority_;
};

inline ::testing::Matcher<const ::xls::Node*> OneHot(
    absl::optional<LsbOrMsb> priority = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::OneHotMatcher(priority));
}

inline ::testing::Matcher<const ::xls::Node*> OneHot(
    ::testing::Matcher<const ::xls::Node*> operand,
    absl::optional<LsbOrMsb> priority = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::OneHotMatcher(operand, priority));
}

// Select matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::Select());
//   EXPECT_THAT(foo, op::Select(op::Param(), /*cases=*/{op::Xor(), op::And});
//   EXPECT_THAT(foo, op::Select(op::Param(),
//                               /*cases=*/{op::Xor(), op::And},
//                               /*default_value=*/op::Literal()));
class SelectMatcher : public NodeMatcher {
 public:
  SelectMatcher(::testing::Matcher<const Node*> selector,
                std::vector<::testing::Matcher<const Node*>> cases,
                absl::optional<::testing::Matcher<const Node*>> default_value)
      : NodeMatcher(Op::kSel,
                    [&]() {
                      std::vector<::testing::Matcher<const Node*>> operands;
                      operands.push_back(selector);
                      operands.insert(operands.end(), cases.begin(),
                                      cases.end());
                      if (default_value.has_value()) {
                        operands.push_back(*default_value);
                      }
                      return operands;
                    }()),
        default_value_(default_value) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<::testing::Matcher<const Node*>> default_value_;
};

inline ::testing::Matcher<const ::xls::Node*> Select(
    ::testing::Matcher<const Node*> selector,
    std::vector<::testing::Matcher<const Node*>> cases,
    absl::optional<::testing::Matcher<const Node*>> default_value =
        absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SelectMatcher(selector, cases, default_value));
}

inline ::testing::Matcher<const ::xls::Node*> Select() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kSel, {}));
}

// OneHotSelect matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::OneHotSelect());
//   EXPECT_THAT(foo, op::OneHotSelect(op::Param(),
//                                     /*cases=*/{op::Xor(), op::And});
inline ::testing::Matcher<const ::xls::Node*> OneHotSelect(
    ::testing::Matcher<const Node*> selector,
    std::vector<::testing::Matcher<const Node*>> cases) {
  std::vector<::testing::Matcher<const Node*>> operands;
  operands.push_back(selector);
  operands.insert(operands.end(), cases.begin(), cases.end());
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kOneHotSel, operands));
}

inline ::testing::Matcher<const ::xls::Node*> OneHotSelect() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kOneHotSel, {}));
}

// TupleIndex matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::TupleIndex());
//   EXPECT_THAT(foo, op::TupleIndex(/*index=*/42));
//   EXPECT_THAT(foo, op::TupleIndex(op::Param(), /*index=*/42));
class TupleIndexMatcher : public NodeMatcher {
 public:
  explicit TupleIndexMatcher(::testing::Matcher<const Node*> operand,
                             absl::optional<int64> index = absl::nullopt)
      : NodeMatcher(Op::kTupleIndex, {operand}), index_(index) {}
  explicit TupleIndexMatcher(absl::optional<int64> index = absl::nullopt)
      : NodeMatcher(Op::kTupleIndex, {}), index_(index) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> index_;
};

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    absl::optional<int64> index = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::TupleIndexMatcher(index));
}

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    ::testing::Matcher<const ::xls::Node*> operand,
    absl::optional<int64> index = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::TupleIndexMatcher(operand, index));
}

// Send matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::Send());
//   EXPECT_THAT(foo, op::Send(/*channel_id=*/42));
//   EXPECT_THAT(foo, op::Send(/*token=*/op::Param(), /*data=*/{op::Param()},
//                             /*channel_id=*/42));
class SendMatcher : public NodeMatcher {
 public:
  explicit SendMatcher(::testing::Matcher<const Node*> token,
                       std::vector<::testing::Matcher<const Node*>> data,
                       absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kSend,
                    [&]() {
                      std::vector<::testing::Matcher<const Node*>> operands;
                      operands.push_back(token);
                      operands.insert(operands.end(), data.begin(), data.end());
                      return operands;
                    }()),
        channel_id_(channel_id) {}
  explicit SendMatcher(absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kSend, {}), channel_id_(channel_id) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> channel_id_;
};

inline ::testing::Matcher<const ::xls::Node*> Send(
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendMatcher(channel_id));
}

inline ::testing::Matcher<const ::xls::Node*> Send(
    ::testing::Matcher<const ::xls::Node*> token,
    std::vector<::testing::Matcher<const Node*>> data,
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendMatcher(token, data, channel_id));
}

// SendIf matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::SendIf());
//   EXPECT_THAT(foo, op::SendIf(/*channel_id=*/42));
//   EXPECT_THAT(foo, op::SendIf(/*token=*/op::Param(), /*pred=*/op::Param(),
//                               /*data=*/{op::Param()}, /*channel_id=*/42));
class SendIfMatcher : public NodeMatcher {
 public:
  explicit SendIfMatcher(::testing::Matcher<const Node*> token,
                         ::testing::Matcher<const Node*> pred,
                         std::vector<::testing::Matcher<const Node*>> data,
                         absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kSendIf,
                    [&]() {
                      std::vector<::testing::Matcher<const Node*>> operands;
                      operands.push_back(token);
                      operands.push_back(pred);
                      operands.insert(operands.end(), data.begin(), data.end());
                      return operands;
                    }()),
        channel_id_(channel_id) {}
  explicit SendIfMatcher(absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kSendIf, {}), channel_id_(channel_id) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> channel_id_;
};

inline ::testing::Matcher<const ::xls::Node*> SendIf(
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendIfMatcher(channel_id));
}

inline ::testing::Matcher<const ::xls::Node*> SendIf(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const ::xls::Node*> pred,
    std::vector<::testing::Matcher<const Node*>> data,
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendIfMatcher(token, pred, data, channel_id));
}

// Receive matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::Receive());
//   EXPECT_THAT(foo, op::Receive(/*channel_id=*/42));
//   EXPECT_THAT(foo, op::Receive(/*token=*/op::Param(), /*pred=*/op::Param(),
//                                /*channel_id=*/42));
class ReceiveMatcher : public NodeMatcher {
 public:
  explicit ReceiveMatcher(::testing::Matcher<const Node*> token,
                          absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kReceive, {token}), channel_id_(channel_id) {}
  explicit ReceiveMatcher(absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kReceive, {}), channel_id_(channel_id) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> channel_id_;
};

inline ::testing::Matcher<const ::xls::Node*> Receive(
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveMatcher(channel_id));
}

inline ::testing::Matcher<const ::xls::Node*> Receive(
    ::testing::Matcher<const ::xls::Node*> token,
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveMatcher(token, channel_id));
}

// ReceiveIf matcher. Supported forms:
//
//   EXPECT_THAT(foo, op::ReceiveIf());
//   EXPECT_THAT(foo, op::ReceiveIf(/*channel_id=*/42));
//   EXPECT_THAT(foo, op::ReceiveIf(/*token=*/op::Param(),
//                                  /*pred=*/op::Param(), /*channel_id=*/42));
class ReceiveIfMatcher : public NodeMatcher {
 public:
  explicit ReceiveIfMatcher(::testing::Matcher<const Node*> token,
                            ::testing::Matcher<const Node*> pred,
                            absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kReceiveIf, {token, pred}), channel_id_(channel_id) {}
  explicit ReceiveIfMatcher(absl::optional<int64> channel_id = absl::nullopt)
      : NodeMatcher(Op::kReceiveIf, {}), channel_id_(channel_id) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64> channel_id_;
};

inline ::testing::Matcher<const ::xls::Node*> ReceiveIf(
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveIfMatcher(channel_id));
}

inline ::testing::Matcher<const ::xls::Node*> ReceiveIf(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const ::xls::Node*> pred,
    absl::optional<int64> channel_id = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveIfMatcher(token, pred, channel_id));
}

}  // namespace op_matchers
}  // namespace xls

#endif  // XLS_IR_IR_MATCHER_H_
