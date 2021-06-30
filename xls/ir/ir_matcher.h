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

#ifndef XLS_IR_IR_MATCHER_H_
#define XLS_IR_IR_MATCHER_H_

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_base.h"
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
  NodeMatcher(const NodeMatcher&) = default;
  NodeMatcher(Op op, absl::Span<const ::testing::Matcher<const Node*>> operands)
      : op_(op), operands_(operands.begin(), operands.end()) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 protected:
  // Helper for DescribeTo which emits a description of the match with optional
  // extra fields. The resulting string has the form:
  //
  //  op(operand_0, operand_1, ..., operand_n, field_0, field_1, ..., field_n)
  //
  // This enables emission of match descriptions with attribute values. For
  // example:
  //
  //   tuple_index(param(), index=1)
  void DescribeToHelper(::std::ostream* os,
                        absl::Span<const std::string> additional_fields) const;

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
NODE_MATCHER(ArrayConcat);
NODE_MATCHER(Assert);
NODE_MATCHER(BitSliceUpdate);
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
  void DescribeTo(::std::ostream* os) const override;

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
//   EXPECT_THAT(foo, m::BitSlice());
//   EXPECT_THAT(foo, m::BitSlice(m::Param()));
//   EXPECT_THAT(foo, m::BitSlice(/*start=*/7, /*width=*/8));
//   EXPECT_THAT(foo, m::BitSlice(/*operand=*/m::Param(), /*start=*/7,
//                                /*width=*/8));
class BitSliceMatcher : public NodeMatcher {
 public:
  BitSliceMatcher(::testing::Matcher<const Node*> operand,
                  absl::optional<int64_t> start, absl::optional<int64_t> width)
      : NodeMatcher(Op::kBitSlice, {operand}), start_(start), width_(width) {}
  BitSliceMatcher(absl::optional<int64_t> start, absl::optional<int64_t> width)
      : NodeMatcher(Op::kBitSlice, {}), start_(start), width_(width) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64_t> start_;
  absl::optional<int64_t> width_;
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
    ::testing::Matcher<const Node*> operand, int64_t start, int64_t width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::BitSliceMatcher(operand, start, width));
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(int64_t start,
                                                       int64_t width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::BitSliceMatcher(start, width));
}

// DynamicBitSlice matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::DynamicBitSlice());
//   EXPECT_THAT(foo, m::DynamicBitSlice(m::Param(), m::Param()));
//   EXPECT_THAT(foo, m::DynamicBitSlice(/*operand=*/m::Param(),
//                                        /*start=*/m::Param(), /*width=*/8));
class DynamicBitSliceMatcher : public NodeMatcher {
 public:
  DynamicBitSliceMatcher(::testing::Matcher<const Node*> operand,
                         ::testing::Matcher<const Node*> start,
                         absl::optional<int64_t> width)
      : NodeMatcher(Op::kDynamicBitSlice, {operand, start}), width_(width) {}
  DynamicBitSliceMatcher() : NodeMatcher(Op::kDynamicBitSlice, {}) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64_t> width_;
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
    ::testing::Matcher<const Node*> start, int64_t width) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::DynamicBitSliceMatcher(operand, start, width));
}

// DynamicCountedFor mather. Supported forms:
//   EXPECT_THAT(foo,
//               m::DynamicCountedForMatcher(
//                   m::Param(), m::Param(),
//                   m::Param(), {m::Xor(), bar}));
//
// Where bar is a FunctionBase*.
class DynamicCountedForMatcher : public NodeMatcher {
 public:
  explicit DynamicCountedForMatcher(
      ::testing::Matcher<const Node*> init,
      ::testing::Matcher<const Node*> trip_count,
      ::testing::Matcher<const Node*> stride, FunctionBase* body,
      std::vector<::testing::Matcher<const Node*>> invariant_args)
      : NodeMatcher(Op::kDynamicCountedFor,
                    [&]() {
                      std::vector<::testing::Matcher<const Node*>> operands;
                      operands.push_back(init);
                      operands.push_back(trip_count);
                      operands.push_back(stride);
                      operands.insert(operands.end(), invariant_args.begin(),
                                      invariant_args.end());
                      return operands;
                    }()),
        body_(body) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  FunctionBase* body_;
};

inline ::testing::Matcher<const ::xls::Node*> DynamicCountedFor(
    ::testing::Matcher<const Node*> init,
    ::testing::Matcher<const Node*> trip_count,
    ::testing::Matcher<const Node*> stride, FunctionBase* body,
    std::vector<::testing::Matcher<const Node*>> invariant_args) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::DynamicCountedForMatcher(init, trip_count, stride,
                                                       body, invariant_args));
}

// Literal matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::Literal());
//   EXPECT_THAT(foo, m::Literal(Value(UBits(7, 8))));
//   EXPECT_THAT(foo, m::Literal(UBits(7, 8)));
//   EXPECT_THAT(foo, m::Literal(42));
//   EXPECT_THAT(foo, m::Literal("bits[8]:7"));
//   EXPECT_THAT(foo, m::Literal("bits[8]:0x7"));
//   EXPECT_THAT(foo, m::Literal("bits[8]:0b111"));
class LiteralMatcher : public NodeMatcher {
 public:
  explicit LiteralMatcher(FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), format_(format) {}
  explicit LiteralMatcher(absl::optional<Value> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), value_(value), format_(format) {}
  explicit LiteralMatcher(absl::optional<int64_t> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), uint64_value_(value), format_(format) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  // At most one of the optional data members has a value.
  absl::optional<Value> value_;
  absl::optional<uint64_t> uint64_value_;
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
inline ::testing::Matcher<const ::xls::Node*> Literal(uint64_t value) {
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
//   EXPECT_THAT(foo, m::OneHot());
//   EXPECT_THAT(foo, m::OneHot(m::Param()));
//   EXPECT_THAT(foo, m::OneHot(/*priority=*/LsbOrMsb::kLsb));
//   EXPECT_THAT(foo, m::OneHot(m::Param(), /*priority=*/LsbOrMsb::kLsb));
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
//   EXPECT_THAT(foo, m::Select());
//   EXPECT_THAT(foo, m::Select(m::Param(), /*cases=*/{m::Xor(), m::And});
//   EXPECT_THAT(foo, m::Select(m::Param(),
//                              /*cases=*/{m::Xor(), m::And},
//                              /*default_value=*/m::Literal()));
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
//   EXPECT_THAT(foo, m::OneHotSelect());
//   EXPECT_THAT(foo, m::OneHotSelect(m::Param(),
//                                    /*cases=*/{m::Xor(), m::And});
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
//   EXPECT_THAT(foo, m::TupleIndex());
//   EXPECT_THAT(foo, m::TupleIndex(/*index=*/42));
//   EXPECT_THAT(foo, m::TupleIndex(m::Param(), /*index=*/42));
class TupleIndexMatcher : public NodeMatcher {
 public:
  explicit TupleIndexMatcher(::testing::Matcher<const Node*> operand,
                             absl::optional<int64_t> index = absl::nullopt)
      : NodeMatcher(Op::kTupleIndex, {operand}), index_(index) {}
  explicit TupleIndexMatcher(absl::optional<int64_t> index = absl::nullopt)
      : NodeMatcher(Op::kTupleIndex, {}), index_(index) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  absl::optional<int64_t> index_;
};

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    absl::optional<int64_t> index = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::TupleIndexMatcher(index));
}

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    ::testing::Matcher<const ::xls::Node*> operand,
    absl::optional<int64_t> index = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::TupleIndexMatcher(operand, index));
}

// Matcher for various properties of channels. Used within matcher of nodes
// which communicate over channels (e.g., send and receive). Supported forms:
//
//   m::Channel(/*name=*/"foo");
//   m::Channel(/*id=*/42);
//   m::Channel(ChannelKind::kPort);
//
class ChannelMatcher
    : public ::testing::MatcherInterface<const ::xls::Channel*> {
 public:
  ChannelMatcher(absl::optional<int64_t> id, absl::optional<std::string> name,
                 absl::optional<ChannelKind> kind)
      : id_(id), name_(name), kind_(kind) {}

  bool MatchAndExplain(const ::xls::Channel* channel,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 protected:
  absl::optional<int64_t> id_;
  absl::optional<std::string> name_;
  absl::optional<ChannelKind> kind_;
};

inline ::testing::Matcher<const ::xls::Channel*> Channel() {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      absl::nullopt, absl::nullopt, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(
    absl::optional<int64_t> id, absl::optional<std::string> name,
    absl::optional<ChannelKind> kind) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ChannelMatcher(id, name, kind));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(int64_t id) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ChannelMatcher(id, absl::nullopt, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(
    absl::string_view name) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      absl::nullopt, std::string{name}, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(ChannelKind kind) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      absl::nullopt, absl::nullopt, kind));
}

// Abstract base class for matchers of nodes which use channels.
class ChannelNodeMatcher : public NodeMatcher {
 public:
  ChannelNodeMatcher(
      Op op, absl::Span<const ::testing::Matcher<const Node*>> operands,
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : NodeMatcher(op, operands), channel_matcher_(channel_matcher) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher_;
};

// Send matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::Send());
//   EXPECT_THAT(foo, m::Send(m::Channel(42)));
//   EXPECT_THAT(foo, m::Send(/*token=*/m::Param(), /*data=*/m::Param(),
//                            m::Channel(42)));
//   EXPECT_THAT(foo, m::Send(/*token=*/m::Param(), /*data=*/m::Param(),
//                            /*predicate=*/m::Param(),
//                            m::Channel(42)));
class SendMatcher : public ChannelNodeMatcher {
 public:
  explicit SendMatcher(
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kSend, {}, channel_matcher) {}
  explicit SendMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> data,
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kSend, {token, data}, channel_matcher) {}
  explicit SendMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> data,
      ::testing::Matcher<const Node*> predicate,
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kSend, {token, data, predicate},
                           channel_matcher) {}
};

inline ::testing::Matcher<const ::xls::Node*> Send(
    absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendMatcher(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Send(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const Node*> data,
    absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::SendMatcher(token, data, channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Send(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const Node*> data,
    ::testing::Matcher<const Node*> predicate,
    absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        absl::nullopt) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::SendMatcher(
      token, data, predicate, channel_matcher));
}

// Receive matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::Receive());
//   EXPECT_THAT(foo, m::Receive(m::Channel(...))
//   EXPECT_THAT(foo, m::Receive(/*token=*/m::Param(), m::Channel(...)))
//   EXPECT_THAT(foo, m::Receive(/*token=*/m::Param(), /*predicate=*/m::Param(),
//                               m::Channel(...)))
class ReceiveMatcher : public ChannelNodeMatcher {
 public:
  explicit ReceiveMatcher(
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive, {}, channel_matcher) {}
  explicit ReceiveMatcher(
      ::testing::Matcher<const Node*> token,
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive, {token}, channel_matcher) {}
  explicit ReceiveMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> predicate,
      absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive, {token, predicate}, channel_matcher) {}
};

inline ::testing::Matcher<const ::xls::Node*> Receive(
    absl::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveMatcher(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Receive(
    ::testing::Matcher<const Node*> token,
    ::testing::Matcher<const ::xls::Channel*> channel_matcher) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ReceiveMatcher(token, channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Receive(
    ::testing::Matcher<const Node*> token,
    ::testing::Matcher<const Node*> predicate,
    ::testing::Matcher<const ::xls::Channel*> channel_matcher) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ReceiveMatcher(
      token, predicate, channel_matcher));
}

// ArrayIndex matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::ArrayIndex());
//   EXPECT_THAT(foo, m::ArrayIndex(m::Param(),
//                                  /*indices=*/{m::Xor(), m::And});
class ArrayIndexMatcher : public NodeMatcher {
 public:
  ArrayIndexMatcher(::testing::Matcher<const Node*> array,
                    std::vector<::testing::Matcher<const Node*>> indices)
      : NodeMatcher(Op::kArrayIndex, [&]() {
          std::vector<::testing::Matcher<const Node*>> operands;
          operands.push_back(array);
          operands.insert(operands.end(), indices.begin(), indices.end());
          return operands;
        }()) {}
};

inline ::testing::Matcher<const ::xls::Node*> ArrayIndex(
    ::testing::Matcher<const Node*> array,
    std::vector<::testing::Matcher<const Node*>> indices) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ArrayIndexMatcher(array, indices));
}

inline ::testing::Matcher<const ::xls::Node*> ArrayIndex() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kArrayIndex, {}));
}

// ArrayUpdate matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::ArrayUpdate());
//   EXPECT_THAT(foo, m::ArrayUpdate(m::Param(),
//                                        /*indices=*/{m::Xor(), m::And});
class ArrayUpdateMatcher : public NodeMatcher {
 public:
  ArrayUpdateMatcher(::testing::Matcher<const Node*> array,
                     ::testing::Matcher<const Node*> value,
                     std::vector<::testing::Matcher<const Node*>> indices)
      : NodeMatcher(Op::kArrayUpdate, [&]() {
          std::vector<::testing::Matcher<const Node*>> operands;
          operands.push_back(array);
          operands.push_back(value);
          operands.insert(operands.end(), indices.begin(), indices.end());
          return operands;
        }()) {}
};

inline ::testing::Matcher<const ::xls::Node*> ArrayUpdate(
    ::testing::Matcher<const Node*> array,
    ::testing::Matcher<const Node*> value,
    std::vector<::testing::Matcher<const Node*>> indices) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::ArrayUpdateMatcher(array, value, indices));
}

inline ::testing::Matcher<const ::xls::Node*> ArrayUpdate() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kArrayUpdate, {}));
}

// InputPort matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::InputPort());
//   EXPECT_THAT(foo, m::InputPort("foo"));
//
class InputPortMatcher : public NodeMatcher {
 public:
  explicit InputPortMatcher(absl::optional<std::string> name)
      : NodeMatcher(Op::kInputPort, /*operands=*/{}), name_(name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  absl::optional<std::string> name_;
};

inline ::testing::Matcher<const ::xls::Node*> InputPort(
    absl::optional<std::string> name = absl::nullopt) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::InputPortMatcher(name));
}

// OutputPort matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::OutputPort());
//   EXPECT_THAT(foo, m::OutputPort("foo"));
//   EXPECT_THAT(foo, m::OutputPort(m::Add()));
//   EXPECT_THAT(foo, m::OutputPort("foo", m::Add());
//
class OutputPortMatcher : public NodeMatcher {
 public:
  explicit OutputPortMatcher(absl::optional<std::string> name)
      : NodeMatcher(Op::kOutputPort, /*operands=*/{}), name_(name) {}

  OutputPortMatcher(::testing::Matcher<const ::xls::Node*> data,
                    absl::optional<std::string> name)
      : NodeMatcher(Op::kOutputPort, /*operands=*/{data}), name_(name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  absl::optional<std::string> name_;
};

inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    absl::optional<std::string> name = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::OutputPortMatcher(name));
}

inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    ::testing::Matcher<const ::xls::Node*> data) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::OutputPortMatcher(data, /*name=*/absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    absl::optional<std::string> name,
    ::testing::Matcher<const ::xls::Node*> data) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::OutputPortMatcher(data, name));
}

// RegisterRead matcher. Matches register name only. Supported forms:
//
//   EXPECT_THAT(x, m::RegisterRead());
//   EXPECT_THAT(x, m::RegisterRead("x"));
//
class RegisterReadMatcher : public NodeMatcher {
 public:
  explicit RegisterReadMatcher(absl::optional<std::string> register_name)
      : NodeMatcher(Op::kRegisterRead, /*operands=*/{}),
        register_name_(register_name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  absl::optional<std::string> register_name_;
};

inline ::testing::Matcher<const ::xls::Node*> RegisterRead(
    absl::optional<std::string> register_name) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterReadMatcher(register_name));
}

inline ::testing::Matcher<const ::xls::Node*> RegisterRead() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::NodeMatcher(Op::kRegisterRead, {}));
}

// RegisterWrite matcher. Matches register name only. Supported forms:
//
//   EXPECT_THAT(x, m::RegisterWrite());
//   EXPECT_THAT(x, m::RegisterWrite("x"));
//   EXPECT_THAT(x, m::RegisterWrite(data));
//   EXPECT_THAT(x, m::RegisterWrite("x", data));
//
class RegisterWriteMatcher : public NodeMatcher {
 public:
  explicit RegisterWriteMatcher(absl::optional<std::string> register_name)
      : NodeMatcher(Op::kRegisterWrite, /*operands=*/{}),
        register_name_(register_name) {}

  explicit RegisterWriteMatcher(::testing::Matcher<const ::xls::Node*> data,
                                absl::optional<std::string> register_name)
      : NodeMatcher(Op::kRegisterWrite, /*operands=*/{data}),
        register_name_(register_name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  absl::optional<std::string> register_name_;
};

inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    absl::optional<std::string> register_name = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterWriteMatcher(register_name));
}

inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    absl::optional<std::string> register_name, const Node* node) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterWriteMatcher(node, register_name));
}

// Register matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::Register());
//   EXPECT_THAT(foo, m::Register("foo"));
//   EXPECT_THAT(foo, m::Register(m::Add()));
//   EXPECT_THAT(foo, m::Register("foo", m::Add()));
//
class RegisterMatcher : public ::testing::MatcherInterface<const Node*> {
 public:
  explicit RegisterMatcher(absl::optional<std::string> register_name)
      : d_matcher_(register_name) {}
  RegisterMatcher(::testing::Matcher<const Node*> input,
                  absl::optional<std::string> register_name)
      : q_matcher_(RegisterWriteMatcher(input, register_name)),
        d_matcher_(register_name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  // Optional matcher for the send side of the register (the Q input port).
  absl::optional<RegisterWriteMatcher> q_matcher_;

  // Matcher for the receive side of the register (the D output port). This
  // matches a tuple index of a receive operation.
  RegisterReadMatcher d_matcher_;
};

inline ::testing::Matcher<const ::xls::Node*> Register(
    absl::optional<std::string> register_name = absl::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterMatcher(register_name));
}

inline ::testing::Matcher<const ::xls::Node*> Register(
    ::testing::Matcher<const Node*> input) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterMatcher(input, absl::nullopt));
}

inline ::testing::Matcher<const ::xls::Node*> Register(
    absl::optional<std::string> register_name,
    ::testing::Matcher<const Node*> input) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::RegisterMatcher(input, register_name));
}

}  // namespace op_matchers
}  // namespace xls

#endif  // XLS_IR_IR_MATCHER_H_
