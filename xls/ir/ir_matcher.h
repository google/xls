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

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace op_matchers {
namespace internal {
// Internal matcher for IR names. When a matcher uses a name, e.g.
// m::Name("some_string_literal"), a NameMatcherInternal is constructed and
// passed as to the outer matcher, in the above example a NameMatcher.
// We could use the default matcher, but this gives a slightly nicer message
// that lets you know its a name.
struct NameMatcherInternal {
  using is_gtest_matcher = void;

  explicit NameMatcherInternal(std::string_view name) : name_(name) {}
  explicit NameMatcherInternal(std::string&& name) : name_(std::move(name)) {}

  bool MatchAndExplain(std::string_view name,
                       ::testing::MatchResultListener* listener) const;
  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(::std::ostream* os) const;

  std::string name_;
};
}  // namespace internal

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
class NodeMatcher {
 public:
  using is_gtest_matcher = void;

  NodeMatcher(Op op, absl::Span<const ::testing::Matcher<const Node*>> operands)
      : op_(op), operands_(operands.begin(), operands.end()) {}
  virtual ~NodeMatcher() = default;

  virtual bool MatchAndExplain(const Node* node,
                               ::testing::MatchResultListener* listener) const;
  virtual void DescribeTo(::std::ostream* os) const;

  virtual void DescribeNegationTo(::std::ostream* os) const {
    *os << "did not match: ";
    DescribeTo(os);
  }

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
class TypeMatcher {
 public:
  using is_gtest_matcher = void;

  explicit TypeMatcher(std::string_view type_str) : type_str_(type_str) {}

  // Match against Node*.
  bool MatchAndExplain(const ::xls::Node* node,
                       ::testing::MatchResultListener* listener) const;
  // Match against Type*.
  bool MatchAndExplain(const ::xls::Type* type,
                       ::testing::MatchResultListener* listener) const;

  void DescribeTo(std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 private:
  std::string type_str_;
};

inline TypeMatcher Type(const Type* type) {
  return ::xls::op_matchers::TypeMatcher(type->ToString());
}

template <typename T>
inline TypeMatcher Type(T type_str)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::TypeMatcher(std::string_view{type_str});
}

// Class for matching node names. Example usage:
//
//   EXPECT_THAT(baz, m::Or(m::Name("foo"), m::Name("bar"))
//
// TODO(meheff): Through template wizardry it'd probably be possible to elide
// the m::Name. For example: EXPECT_THAT(baz, m::Or("foo", "bar")).
class NameMatcher {
 public:
  using is_gtest_matcher = void;

  explicit NameMatcher(::testing::Matcher<const std::string> inner_matcher)
      : inner_matcher_(std::move(inner_matcher)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const;
  void DescribeTo(std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 private:
  const ::testing::Matcher<const std::string> inner_matcher_;
};

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> Name(T name_str)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::NameMatcher(
      internal::NameMatcherInternal(std::string_view{name_str}));
}

inline ::testing::Matcher<const ::xls::Node*> Name(
    ::testing::Matcher<const std::string> matcher) {
  return ::xls::op_matchers::NameMatcher(std::move(matcher));
}

// Node* matchers for ops which have no metadata beyond Op, type, and operands.
#define NODE_MATCHER(op)                                                     \
  template <typename... M>                                                   \
  ::testing::Matcher<const ::xls::Node*> op(M... operands) {                 \
    return ::xls::op_matchers::NodeMatcher(::xls::Op::k##op, {operands...}); \
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
NODE_MATCHER(Gate);
NODE_MATCHER(Identity);
NODE_MATCHER(Nand);
NODE_MATCHER(Ne);
NODE_MATCHER(Neg);
NODE_MATCHER(Next);
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
NODE_MATCHER(SMulp);
NODE_MATCHER(Shll);
NODE_MATCHER(Shra);
NODE_MATCHER(Shrl);
NODE_MATCHER(SignExt);
NODE_MATCHER(Sub);
NODE_MATCHER(Trace);
NODE_MATCHER(Tuple);
NODE_MATCHER(UDiv);
NODE_MATCHER(UGe);
NODE_MATCHER(UGt);
NODE_MATCHER(ULe);
NODE_MATCHER(ULt);
NODE_MATCHER(UMod);
NODE_MATCHER(UMul);
NODE_MATCHER(UMulp);
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
//   EXPECT_THAT(x, m::Param(HasSubstr("substr")));
class ParamMatcher : public NodeMatcher {
 public:
  explicit ParamMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : NodeMatcher(Op::kParam, /*operands=*/{}),
        name_matcher_(std::move(name)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const std::string>> name_matcher_;
};

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> Param(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::ParamMatcher(
      std::make_optional(internal::NameMatcherInternal(std::string{name})));
}

inline ::testing::Matcher<const ::xls::Node*> Param(
    ::testing::Matcher<const std::string> matcher) {
  return ::xls::op_matchers::ParamMatcher(std::move(matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Param() {
  return ::xls::op_matchers::ParamMatcher(std::nullopt);
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
                  ::testing::Matcher<int64_t> start,
                  ::testing::Matcher<int64_t> width)
      : NodeMatcher(Op::kBitSlice, {std::move(operand)}),
        start_(std::move(start)),
        width_(std::move(width)) {}
  BitSliceMatcher(::testing::Matcher<int64_t> start,
                  ::testing::Matcher<int64_t> width)
      : NodeMatcher(Op::kBitSlice, {}),
        start_(std::move(start)),
        width_(std::move(width)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  ::testing::Matcher<int64_t> start_;
  ::testing::Matcher<int64_t> width_;
};

inline ::testing::Matcher<const ::xls::Node*> BitSlice() {
  return ::xls::op_matchers::BitSliceMatcher(::testing::_, ::testing::_);
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(
    ::testing::Matcher<const Node*> operand) {
  return ::xls::op_matchers::BitSliceMatcher(std::move(operand), ::testing::_,
                                             ::testing::_);
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(
    ::testing::Matcher<const Node*> operand, ::testing::Matcher<int64_t> start,
    ::testing::Matcher<int64_t> width) {
  return ::xls::op_matchers::BitSliceMatcher(
      std::move(operand), std::move(start), std::move(width));
}

inline ::testing::Matcher<const ::xls::Node*> BitSlice(
    ::testing::Matcher<int64_t> start, ::testing::Matcher<int64_t> width) {
  return ::xls::op_matchers::BitSliceMatcher(std::move(start),
                                             std::move(width));
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
                         std::optional<int64_t> width)
      : NodeMatcher(Op::kDynamicBitSlice,
                    {std::move(operand), std::move(start)}),
        width_(width) {}
  DynamicBitSliceMatcher() : NodeMatcher(Op::kDynamicBitSlice, {}) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  std::optional<int64_t> width_;
};

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice() {
  return ::xls::op_matchers::DynamicBitSliceMatcher();
}

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice(
    ::testing::Matcher<const Node*> operand,
    ::testing::Matcher<const Node*> start) {
  return ::xls::op_matchers::DynamicBitSliceMatcher(
      std::move(operand), std::move(start), std::nullopt);
}

inline ::testing::Matcher<const ::xls::Node*> DynamicBitSlice(
    ::testing::Matcher<const Node*> operand,
    ::testing::Matcher<const Node*> start, int64_t width) {
  return ::xls::op_matchers::DynamicBitSliceMatcher(std::move(operand),
                                                    std::move(start), width);
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
  return ::xls::op_matchers::DynamicCountedForMatcher(
      std::move(init), std::move(trip_count), std::move(stride), body,
      std::move(invariant_args));
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
  explicit LiteralMatcher(std::optional<Value> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}),
        value_(std::move(value)),
        format_(format) {}
  explicit LiteralMatcher(std::optional<int64_t> value,
                          FormatPreference format = FormatPreference::kDefault)
      : NodeMatcher(Op::kLiteral, {}), uint64_value_(value), format_(format) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  // At most one of the optional data members has a value.
  std::optional<Value> value_;
  std::optional<uint64_t> uint64_value_;
  FormatPreference format_;
};

inline ::testing::Matcher<const ::xls::Node*> Literal() {
  return ::xls::op_matchers::LiteralMatcher();
}

inline ::testing::Matcher<const ::xls::Node*> Literal(const Value& value) {
  return ::xls::op_matchers::LiteralMatcher(value);
}

inline ::testing::Matcher<const ::xls::Node*> Literal(const Bits& bits) {
  return ::xls::op_matchers::LiteralMatcher(Value(bits));
}
inline ::testing::Matcher<const ::xls::Node*> Literal(uint64_t value) {
  return ::xls::op_matchers::LiteralMatcher(value);
}
inline ::testing::Matcher<const ::xls::Node*> Literal(uint64_t value,
                                                      int64_t width) {
  return ::xls::op_matchers::LiteralMatcher(Value(UBits(value, width)));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> Literal(T value_str)
  requires(std::is_convertible_v<T, std::string_view>)
{
  std::string_view value_str_view = value_str;
  Value value = Parser::ParseTypedValue(value_str_view).value();
  FormatPreference format = FormatPreference::kDefault;
  if (absl::StrContains(value_str_view, "0b")) {
    format = FormatPreference::kBinary;
  } else if (absl::StrContains(value_str_view, "0x")) {
    format = FormatPreference::kHex;
  }
  return ::xls::op_matchers::LiteralMatcher(value, format);
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
                         std::optional<LsbOrMsb> priority = std::nullopt)
      : NodeMatcher(Op::kOneHot, {std::move(operand)}), priority_(priority) {}
  explicit OneHotMatcher(std::optional<LsbOrMsb> priority = std::nullopt)
      : NodeMatcher(Op::kOneHot, {}), priority_(priority) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  std::optional<LsbOrMsb> priority_;
};

inline ::testing::Matcher<const ::xls::Node*> OneHot(
    std::optional<LsbOrMsb> priority = std::nullopt) {
  return ::xls::op_matchers::OneHotMatcher(priority);
}

inline ::testing::Matcher<const ::xls::Node*> OneHot(
    ::testing::Matcher<const ::xls::Node*> operand,
    std::optional<LsbOrMsb> priority = std::nullopt) {
  return ::xls::op_matchers::OneHotMatcher(std::move(operand), priority);
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
                std::optional<::testing::Matcher<const Node*>> default_value)
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
  std::optional<::testing::Matcher<const Node*>> default_value_;
};

inline ::testing::Matcher<const ::xls::Node*> Select(
    ::testing::Matcher<const Node*> selector,
    std::vector<::testing::Matcher<const Node*>> cases,
    std::optional<::testing::Matcher<const Node*>> default_value =
        std::nullopt) {
  return ::xls::op_matchers::SelectMatcher(
      std::move(selector), std::move(cases), std::move(default_value));
}

inline ::testing::Matcher<const ::xls::Node*> Select() {
  return ::xls::op_matchers::NodeMatcher(Op::kSel, {});
}

// OneHotSelect matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::OneHotSelect());
//   EXPECT_THAT(foo, m::OneHotSelect(m::Param(),
//                                    /*cases=*/{m::Xor(), m::And});
inline ::testing::Matcher<const ::xls::Node*> OneHotSelect(
    const ::testing::Matcher<const Node*>& selector,
    std::vector<::testing::Matcher<const Node*>> cases) {
  std::vector<::testing::Matcher<const Node*>> operands;
  operands.push_back(selector);
  operands.insert(operands.end(), cases.begin(), cases.end());
  return ::xls::op_matchers::NodeMatcher(Op::kOneHotSel, operands);
}

inline ::testing::Matcher<const ::xls::Node*> OneHotSelect() {
  return ::xls::op_matchers::NodeMatcher(Op::kOneHotSel, {});
}

// PrioritySelect matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::PrioritySelect());
//   EXPECT_THAT(foo, m::PrioritySelect(m::Param(),
//                                      /*cases=*/{m::Xor(), m::And},
//                                      /*default_value=*/m::Literal());
inline ::testing::Matcher<const ::xls::Node*> PrioritySelect(
    const ::testing::Matcher<const Node*>& selector,
    std::vector<::testing::Matcher<const Node*>> cases,
    const ::testing::Matcher<const Node*>& default_value) {
  std::vector<::testing::Matcher<const Node*>> operands;
  operands.push_back(selector);
  operands.insert(operands.end(), cases.begin(), cases.end());
  operands.push_back(default_value);
  return ::xls::op_matchers::NodeMatcher(Op::kPrioritySel, operands);
}

inline ::testing::Matcher<const ::xls::Node*> PrioritySelect() {
  return ::xls::op_matchers::NodeMatcher(Op::kPrioritySel, {});
}

// TupleIndex matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::TupleIndex());
//   EXPECT_THAT(foo, m::TupleIndex(/*index=*/42));
//   EXPECT_THAT(foo, m::TupleIndex(m::Param(), /*index=*/42));
class TupleIndexMatcher : public NodeMatcher {
 public:
  explicit TupleIndexMatcher(::testing::Matcher<const Node*> operand,
                             std::optional<int64_t> index = std::nullopt)
      : NodeMatcher(Op::kTupleIndex, {std::move(operand)}), index_(index) {}
  explicit TupleIndexMatcher(std::optional<int64_t> index = std::nullopt)
      : NodeMatcher(Op::kTupleIndex, {}), index_(index) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;

 private:
  std::optional<int64_t> index_;
};

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    std::optional<int64_t> index = std::nullopt) {
  return ::xls::op_matchers::TupleIndexMatcher(index);
}

inline ::testing::Matcher<const ::xls::Node*> TupleIndex(
    ::testing::Matcher<const ::xls::Node*> operand,
    std::optional<int64_t> index = std::nullopt) {
  return ::xls::op_matchers::TupleIndexMatcher(std::move(operand), index);
}

// Matcher for various properties of channels. Used within matcher of nodes
// which communicate over channels (e.g., send and receive). Supported forms:
//
//   m::Channel(/*name=*/"foo");
//   m::Channel(/*id=*/42);
//   m::Channel(ChannelKind::kPort);
//   m::Channel(node->GetType());
//   m::ChannelWithType("bits[32]");
//
class ChannelMatcher
    : public ::testing::MatcherInterface<const ::xls::Channel*> {
 public:
  ChannelMatcher(std::optional<int64_t> id,
                 std::optional<::testing::Matcher<std::string>> name,
                 std::optional<ChannelKind> kind,
                 std::optional<std::string_view> type_string)
      : id_(id),
        name_(std::move(name)),
        kind_(kind),
        type_string_(type_string) {}

  bool MatchAndExplain(const ::xls::Channel* channel,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 protected:
  std::optional<int64_t> id_;
  std::optional<::testing::Matcher<std::string>> name_;
  std::optional<ChannelKind> kind_;
  std::optional<std::string> type_string_;
};

inline ::testing::Matcher<const ::xls::Channel*> Channel() {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, std::nullopt, std::nullopt, std::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(
    std::optional<int64_t> id) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      id, std::nullopt, std::nullopt, std::nullopt));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Channel*> Channel(
    std::optional<int64_t> id, T name,
    std::optional<ChannelKind> kind = std::nullopt,
    std::optional<const ::xls::Type*> type_ = std::nullopt)
  requires(std::is_convertible_v<T, std::string>)
{
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      id, std::string{name}, kind,
      type_.has_value() ? std::optional(type_.value()->ToString())
                        : std::nullopt));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Channel*> Channel(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, internal::NameMatcherInternal(std::string_view{name}),
      std::nullopt, std::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(
    const ::testing::Matcher<std::string>& matcher) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, matcher, std::nullopt, std::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(ChannelKind kind) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, std::nullopt, kind, std::nullopt));
}

inline ::testing::Matcher<const ::xls::Channel*> Channel(
    const ::xls::Type* type_) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, std::nullopt, std::nullopt, type_->ToString()));
}

inline ::testing::Matcher<const ::xls::Channel*> ChannelWithType(
    std::string_view type_string) {
  return ::testing::MakeMatcher(new ::xls::op_matchers::ChannelMatcher(
      std::nullopt, std::nullopt, std::nullopt, type_string));
}

// Abstract base class for matchers of nodes which use channels.
class ChannelNodeMatcher : public NodeMatcher {
 public:
  ChannelNodeMatcher(
      Op op, absl::Span<const ::testing::Matcher<const Node*>> operands,
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : NodeMatcher(op, operands),
        channel_matcher_(std::move(channel_matcher)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher_;
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
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kSend, {}, std::move(channel_matcher)) {}
  explicit SendMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> data,
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kSend, {std::move(token), std::move(data)},
                           std::move(channel_matcher)) {}
  explicit SendMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> data,
      ::testing::Matcher<const Node*> predicate,
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(
            Op::kSend,
            {std::move(token), std::move(data), std::move(predicate)},
            std::move(channel_matcher)) {}
};

inline ::testing::Matcher<const ::xls::Node*> Send(
    std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        std::nullopt) {
  return ::xls::op_matchers::SendMatcher(std::move(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Send(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const Node*> data,
    std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        std::nullopt) {
  return ::xls::op_matchers::SendMatcher(std::move(token), std::move(data),
                                         std::move(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Send(
    ::testing::Matcher<const ::xls::Node*> token,
    ::testing::Matcher<const Node*> data,
    ::testing::Matcher<const Node*> predicate,
    std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        std::nullopt) {
  return ::xls::op_matchers::SendMatcher(std::move(token), std::move(data),
                                         std::move(predicate),
                                         std::move(channel_matcher));
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
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive, {}, std::move(channel_matcher)) {}
  explicit ReceiveMatcher(
      ::testing::Matcher<const Node*> token,
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive, {std::move(token)},
                           std::move(channel_matcher)) {}
  explicit ReceiveMatcher(
      ::testing::Matcher<const Node*> token,
      ::testing::Matcher<const Node*> predicate,
      std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher)
      : ChannelNodeMatcher(Op::kReceive,
                           {std::move(token), std::move(predicate)},
                           std::move(channel_matcher)) {}
};

inline ::testing::Matcher<const ::xls::Node*> Receive(
    std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        std::nullopt) {
  return ::xls::op_matchers::ReceiveMatcher(std::move(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Receive(
    ::testing::Matcher<const Node*> token,
    std::optional<::testing::Matcher<const ::xls::Channel*>> channel_matcher =
        std::nullopt) {
  return ::xls::op_matchers::ReceiveMatcher(std::move(token),
                                            std::move(channel_matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Receive(
    ::testing::Matcher<const Node*> token,
    ::testing::Matcher<const Node*> predicate,
    ::testing::Matcher<const ::xls::Channel*> channel_matcher) {
  return ::xls::op_matchers::ReceiveMatcher(
      std::move(token), std::move(predicate), channel_matcher);
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
  return ::xls::op_matchers::ArrayIndexMatcher(std::move(array),
                                               std::move(indices));
}

inline ::testing::Matcher<const ::xls::Node*> ArrayIndex() {
  return ::xls::op_matchers::NodeMatcher(Op::kArrayIndex, {});
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
  return ::xls::op_matchers::ArrayUpdateMatcher(
      std::move(array), std::move(value), std::move(indices));
}

inline ::testing::Matcher<const ::xls::Node*> ArrayUpdate() {
  return ::xls::op_matchers::NodeMatcher(Op::kArrayUpdate, {});
}

// Trace matcher. Supported forms:
//
// EXPECT_THAT(foo, m::Trace());
// EXPECT_THAT(foo, m::Trace({tok, condition, args}));
// EXPECT_THAT(foo, m::Trace(verbosity))
//
class TraceVerbosityMatcher : public NodeMatcher {
 public:
  explicit TraceVerbosityMatcher(::testing::Matcher<int64_t> verbosity)
      : NodeMatcher(Op::kTrace, /*operands=*/{}),
        verbosity_(std::move(verbosity)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  ::testing::Matcher<int64_t> verbosity_;
};

inline ::testing::Matcher<const ::xls::Node*> TraceWithVerbosity(
    ::testing::Matcher<int64_t> verbosity) {
  return ::xls::op_matchers::TraceVerbosityMatcher(std::move(verbosity));
}

// InputPort matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::InputPort());
//   EXPECT_THAT(foo, m::InputPort("foo"));
//   EXPECT_THAT(foo, m::InputPort(HasSubstr("substr")));
//
class InputPortMatcher : public NodeMatcher {
 public:
  explicit InputPortMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : NodeMatcher(Op::kInputPort, /*operands=*/{}),
        name_matcher_(std::move(name)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const std::string>> name_matcher_;
};

inline ::testing::Matcher<const ::xls::Node*> InputPort() {
  return ::xls::op_matchers::InputPortMatcher(std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> InputPort(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::InputPortMatcher(std::make_optional(
      internal::NameMatcherInternal(std::string_view{name})));
}

inline ::testing::Matcher<const ::xls::Node*> InputPort(
    ::testing::Matcher<const std::string> matcher) {
  return ::xls::op_matchers::InputPortMatcher(std::move(matcher));
}

// OutputPort matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::OutputPort());
//   EXPECT_THAT(foo, m::OutputPort("foo"));
//   EXPECT_THAT(foo, m::OutputPort(HasSubstr("substr")));
//   EXPECT_THAT(foo, m::OutputPort(m::Add()));
//   EXPECT_THAT(foo, m::OutputPort("foo", m::Add());
//   EXPECT_THAT(foo, m::OutputPort(HasSubstr("substr"), m::Add());
//
class OutputPortMatcher : public NodeMatcher {
 public:
  explicit OutputPortMatcher(
      std::optional<::testing::Matcher<const std::string>> name_matcher)
      : NodeMatcher(Op::kOutputPort, /*operands=*/{}),
        name_matcher_(std::move(name_matcher)) {}

  OutputPortMatcher(
      ::testing::Matcher<const ::xls::Node*> data,
      std::optional<::testing::Matcher<const std::string>> name_matcher)
      : NodeMatcher(Op::kOutputPort, /*operands=*/{std::move(data)}),
        name_matcher_(std::move(name_matcher)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const std::string>> name_matcher_;
};

inline ::testing::Matcher<const ::xls::Node*> OutputPort() {
  return ::xls::op_matchers::OutputPortMatcher(std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> OutputPort(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::OutputPortMatcher(std::make_optional(
      internal::NameMatcherInternal(std::string_view{name})));
}

// Disambiguate calls to OutputPort(::testing::Matcher<const ::xls::Node*>) by
// making a distinct name.
inline ::testing::Matcher<const ::xls::Node*> OutputPortWithName(
    ::testing::Matcher<const std::string> name) {
  return ::xls::op_matchers::OutputPortMatcher(std::move(name));
}

inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    ::testing::Matcher<const ::xls::Node*> data) {
  return ::xls::op_matchers::OutputPortMatcher(std::move(data),
                                               /*name_matcher=*/std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    T name, const ::xls::Node* data)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::OutputPortMatcher(
      data, std::make_optional(
                internal::NameMatcherInternal(std::string_view{name})));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    T name, ::testing::Matcher<const ::xls::Node*> data)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::OutputPortMatcher(
      std::move(data), internal::NameMatcherInternal(std::string_view{name}));
}

inline ::testing::Matcher<const ::xls::Node*> OutputPort(
    ::testing::Matcher<const std::string> name,
    ::testing::Matcher<const ::xls::Node*> data) {
  return ::xls::op_matchers::OutputPortMatcher(std::move(data),
                                               std::move(name));
}

// RegisterRead matcher. Matches register name only. Supported forms:
//
//   EXPECT_THAT(x, m::RegisterRead());
//   EXPECT_THAT(x, m::RegisterRead("x"));
//   EXPECT_THAT(x, m::RegisterRead(HasSubstr("substr")));
//
class RegisterReadMatcher : public NodeMatcher {
 public:
  explicit RegisterReadMatcher(
      std::optional<::testing::Matcher<const std::string>> register_name)
      : NodeMatcher(Op::kRegisterRead, /*operands=*/{}),
        register_name_(std::move(register_name)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const std::string>> register_name_;
};

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> RegisterRead(T register_name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::RegisterReadMatcher(
      internal::NameMatcherInternal(std::string_view{register_name}));
}

inline ::testing::Matcher<const ::xls::Node*> RegisterRead(
    ::testing::Matcher<const std::string> name) {
  return ::xls::op_matchers::RegisterReadMatcher(std::move(name));
}

inline ::testing::Matcher<const ::xls::Node*> RegisterRead() {
  return ::xls::op_matchers::NodeMatcher(Op::kRegisterRead, {});
}

// RegisterWrite matcher. Matches register name only. Supported forms:
//
//   EXPECT_THAT(x, m::RegisterWrite());
//   EXPECT_THAT(x, m::RegisterWrite("x"));
//   EXPECT_THAT(x, m::RegisterWrite(HasSubstr("substr")));
//   EXPECT_THAT(x, m::RegisterWrite(data));
//   EXPECT_THAT(x, m::RegisterWrite("x", data));
//   EXPECT_THAT(x, m::RegisterWrite(HasSubstr("substr"), data));
//
class RegisterWriteMatcher : public NodeMatcher {
 public:
  explicit RegisterWriteMatcher(
      std::optional<::testing::Matcher<const std::string>> register_name)
      : NodeMatcher(Op::kRegisterWrite, /*operands=*/{}),
        register_name_(std::move(register_name)) {}

  explicit RegisterWriteMatcher(
      ::testing::Matcher<const ::xls::Node*> data,
      std::optional<::testing::Matcher<const std::string>> register_name)
      : NodeMatcher(Op::kRegisterWrite, /*operands=*/{std::move(data)}),
        register_name_(std::move(register_name)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  std::optional<::testing::Matcher<const std::string>> register_name_;
};

inline ::testing::Matcher<const ::xls::Node*> RegisterWrite() {
  return ::xls::op_matchers::RegisterWriteMatcher(std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(T register_name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::RegisterWriteMatcher(
      internal::NameMatcherInternal(std::string_view{register_name}));
}

inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    ::testing::Matcher<const std::string> matcher) {
  return ::xls::op_matchers::RegisterWriteMatcher(std::move(matcher));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    std::optional<T> register_name, const Node* node)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::RegisterWriteMatcher(
      node, register_name.has_value()
                ? std::make_optional(internal::NameMatcherInternal(
                      std::string_view{*register_name}))
                : std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    T register_name, ::testing::Matcher<const ::xls::Node*> node)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::RegisterWriteMatcher(
      std::move(node),
      internal::NameMatcherInternal(std::string_view{register_name}));
}
inline ::testing::Matcher<const ::xls::Node*> RegisterWrite(
    ::testing::Matcher<const std::string> register_name,
    ::testing::Matcher<const ::xls::Node*> node) {
  return ::xls::op_matchers::RegisterWriteMatcher(std::move(node),
                                                  std::move(register_name));
}

// Register matcher. Supported forms:
//
//   EXPECT_THAT(foo, m::Register());
//   EXPECT_THAT(foo, m::Register("foo"));
//   EXPECT_THAT(foo, m::Register(HasSusbtr("substr")));
//   EXPECT_THAT(foo, m::Register(m::Add()));
//   EXPECT_THAT(foo, m::Register("foo", m::Add()));
//   EXPECT_THAT(foo, m::Register(HasSubstr("substr"), m::Add()));
//
class RegisterMatcher {
 public:
  using is_gtest_matcher = void;
  explicit RegisterMatcher(
      std::optional<::testing::Matcher<const std::string>> register_name)
      : d_matcher_(std::move(register_name)) {}
  RegisterMatcher(
      ::testing::Matcher<const Node*> input,
      const std::optional<::testing::Matcher<const std::string>>& register_name)
      : q_matcher_(RegisterWriteMatcher(std::move(input), register_name)),
        d_matcher_(register_name) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const;
  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 private:
  // Optional matcher for the send side of the register (the Q input port).
  std::optional<RegisterWriteMatcher> q_matcher_;

  // Matcher for the receive side of the register (the D output port). This
  // matches a tuple index of a receive operation.
  RegisterReadMatcher d_matcher_;
};

inline ::testing::Matcher<const ::xls::Node*> Register() {
  return ::xls::op_matchers::RegisterMatcher(std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> Register(
    T register_name = std::nullopt) {
  return ::xls::op_matchers::RegisterMatcher(
      internal::NameMatcherInternal(std::string_view{register_name}));
}

// Disambiguate calls to Register(::testing::Matcher<const ::xls::Node*>) by
// making a distinct name.
inline ::testing::Matcher<const ::xls::Node*> RegisterWithName(
    ::testing::Matcher<const std::string> matcher) {
  return ::xls::op_matchers::RegisterMatcher(std::move(matcher));
}

inline ::testing::Matcher<const ::xls::Node*> Register(
    ::testing::Matcher<const Node*> input) {
  return ::xls::op_matchers::RegisterMatcher(std::move(input), std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> Register(
    T register_name, ::testing::Matcher<const Node*> input)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::RegisterMatcher(
      std::move(input),
      internal::NameMatcherInternal(std::string_view{register_name}));
}

inline ::testing::Matcher<const ::xls::Node*> Register(
    ::testing::Matcher<const std::string> register_name,
    ::testing::Matcher<const ::xls::Node*> input) {
  return ::xls::op_matchers::RegisterMatcher(std::move(input),
                                             std::move(register_name));
}

// MinDelay matcher. Supported forms:
//
//   EXPECT_THAT(x, m::MinDelay());
//   EXPECT_THAT(x, m::MinDelay(/*token=*/_));
//   EXPECT_THAT(x, m::MinDelay(/*delay=*/1));
//   EXPECT_THAT(x, m::MinDelay(/*token=*/_, /*delay=*/1));
class MinDelayMatcher : public NodeMatcher {
 public:
  explicit MinDelayMatcher(::testing::Matcher<const ::xls::Node*> token,
                           ::testing::Matcher<int64_t> delay = ::testing::_)
      : NodeMatcher(Op::kMinDelay, /*operands=*/{std::move(token)}),
        delay_(std::move(delay)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;

 private:
  ::testing::Matcher<int64_t> delay_;
};

inline ::testing::Matcher<const ::xls::Node*> MinDelay(
    ::testing::Matcher<const ::xls::Node*> data = ::testing::_,
    ::testing::Matcher<int64_t> delay = ::testing::_) {
  return ::xls::op_matchers::MinDelayMatcher(std::move(data), std::move(delay));
}

inline ::testing::Matcher<const ::xls::Node*> MinDelay(int64_t delay) {
  return ::xls::op_matchers::MinDelayMatcher(::testing::_, delay);
}

// Matcher for FunctionBase. Supported form:
//
//   m::FunctionBase(/*name=*/"foo");
//   m::FunctionBase(/*name=*/HasSubstr("substr"));
//
class FunctionBaseMatcher
    : public ::testing::MatcherInterface<const ::xls::FunctionBase*> {
 public:
  explicit FunctionBaseMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : name_(std::move(name)) {}

  bool MatchAndExplain(const ::xls::FunctionBase* fb,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(::std::ostream* os) const override;

 protected:
  std::optional<::testing::Matcher<const std::string>> name_;
};

inline ::testing::Matcher<const ::xls::FunctionBase*> FunctionBase() {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::FunctionBaseMatcher(std::nullopt));
}

template <typename T>
inline ::testing::Matcher<const ::xls::FunctionBase*> FunctionBase(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::testing::MakeMatcher(new ::xls::op_matchers::FunctionBaseMatcher(
      internal::NameMatcherInternal(std::string_view{name})));
}

inline ::testing::Matcher<const ::xls::FunctionBase*> FunctionBase(
    std::optional<::testing::Matcher<const std::string>> name = std::nullopt) {
  return ::testing::MakeMatcher(
      new ::xls::op_matchers::FunctionBaseMatcher(std::move(name)));
}

// Matcher for functions. Supported forms:
//
//   m::Function();
//   m::Function(/*name=*/"foo");
//   m::Function(/*name=*/HasSubstr("substr"));
//
class FunctionMatcher {
 public:
  using is_gtest_matcher = void;

  explicit FunctionMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : name_(std::move(name)) {}

  bool MatchAndExplain(const ::xls::FunctionBase* fb,
                       ::testing::MatchResultListener* listener) const {
    if (fb == nullptr) {
      return false;
    }
    *listener << fb->name();
    if (!fb->IsFunction()) {
      *listener << " is not a function.";
      return false;
    }
    // Now, match on FunctionBase.
    if (!FunctionBase(name_).MatchAndExplain(fb, listener)) {
      return false;
    }

    return true;
  }

  template <typename T>
  bool MatchAndExplain(const std::unique_ptr<T>& fb,
                       ::testing::MatchResultListener* listener) const
    requires(std::is_base_of_v<::xls::FunctionBase, T>)
  {
    return MatchAndExplain(fb.get(), listener);
  }

  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 protected:
  std::optional<::testing::Matcher<const std::string>> name_;
};

inline ::testing::PolymorphicMatcher<FunctionMatcher> Function() {
  return testing::MakePolymorphicMatcher(
      ::xls::op_matchers::FunctionMatcher(std::nullopt));
}

template <typename T>
inline ::testing::PolymorphicMatcher<FunctionMatcher> Function(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return testing::MakePolymorphicMatcher(::xls::op_matchers::FunctionMatcher(
      internal::NameMatcherInternal(std::string_view{name})));
}

inline ::testing::PolymorphicMatcher<FunctionMatcher> Function(
    ::testing::Matcher<const std::string> name) {
  return testing::MakePolymorphicMatcher(
      ::xls::op_matchers::FunctionMatcher(std::move(name)));
}

// Matcher for procs. Supported forms:
//
//   m::Proc();
//   m::Proc(/*name=*/"foo");
//   m::Proc(/*name=*/HasSusbtr("substr"));
//
class ProcMatcher {
 public:
  using is_gtest_matcher = void;

  explicit ProcMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : name_(std::move(name)) {}

  bool MatchAndExplain(const ::xls::FunctionBase* fb,
                       ::testing::MatchResultListener* listener) const {
    if (fb == nullptr) {
      return false;
    }
    *listener << fb->name();
    if (!fb->IsProc()) {
      *listener << " is not a proc.";
      return false;
    }
    // Now, match on FunctionBase.
    if (!FunctionBase(name_).MatchAndExplain(fb, listener)) {
      return false;
    }

    return true;
  }

  template <typename T>
  bool MatchAndExplain(const std::unique_ptr<T>& fb,
                       ::testing::MatchResultListener* listener) const
    requires(std::is_base_of_v<::xls::FunctionBase, T>)
  {
    return MatchAndExplain(fb.get(), listener);
  }

  template <typename T>
  bool MatchAndExplain(const std::unique_ptr<T>& fb,
                       ::testing::MatchResultListener* listener) const
    requires(std::is_base_of_v<::xls::FunctionBase, T>())
  {
    return MatchAndExplain(fb.get(), listener);
  }

  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 protected:
  std::optional<::testing::Matcher<const std::string>> name_;
};

inline ::testing::PolymorphicMatcher<ProcMatcher> Proc() {
  return ::testing::MakePolymorphicMatcher(
      ::xls::op_matchers::ProcMatcher(std::nullopt));
}

template <typename T>
inline ::testing::PolymorphicMatcher<ProcMatcher> Proc(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::testing::MakePolymorphicMatcher(::xls::op_matchers::ProcMatcher(
      internal::NameMatcherInternal(std::string_view{name})));
}

inline ::testing::PolymorphicMatcher<ProcMatcher> Proc(
    ::testing::Matcher<const std::string> name) {
  return ::testing::MakePolymorphicMatcher(
      ::xls::op_matchers::ProcMatcher(std::move(name)));
}

// Matcher for blocks. Supported forms:
//
//   m::Block();
//   m::Block(/*name=*/"foo");
//   m::Block(/*name=*/HasSubstr("substr"));
//
class BlockMatcher {
 public:
  using is_gtest_matcher = void;

  explicit BlockMatcher(
      std::optional<::testing::Matcher<const std::string>> name)
      : name_(std::move(name)) {}

  bool MatchAndExplain(const ::xls::FunctionBase* fb,
                       ::testing::MatchResultListener* listener) const {
    if (fb == nullptr) {
      return false;
    }
    *listener << fb->name();
    if (!fb->IsBlock()) {
      *listener << " is not a block.";
      return false;
    }
    // Now, match on FunctionBase.
    if (!FunctionBase(name_).MatchAndExplain(fb, listener)) {
      return false;
    }

    return true;
  }

  template <typename T>
  bool MatchAndExplain(const std::unique_ptr<T>& fb,
                       ::testing::MatchResultListener* listener) const
    requires(std::is_base_of_v<::xls::FunctionBase, T>)
  {
    return MatchAndExplain(fb.get(), listener);
  }

  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 protected:
  std::optional<::testing::Matcher<const std::string>> name_;
};

inline ::testing::PolymorphicMatcher<BlockMatcher> Block() {
  return testing::MakePolymorphicMatcher(
      ::xls::op_matchers::BlockMatcher(std::nullopt));
}

template <typename T>
inline ::testing::PolymorphicMatcher<BlockMatcher> Block(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return testing::MakePolymorphicMatcher(::xls::op_matchers::BlockMatcher(
      internal::NameMatcherInternal(std::string_view{name})));
}

inline ::testing::PolymorphicMatcher<BlockMatcher> Block(
    ::testing::Matcher<const std::string> name) {
  return testing::MakePolymorphicMatcher(
      ::xls::op_matchers::BlockMatcher(std::move(name)));
}

// Matcher for instances. Supported forms:
//
//   m::Instantiation()
//   m::Instantiation(/*instance_name=*/name)
//   m::Instantiation(/*kind=*/kind)
//   m::Instantiation(/*instance_name=*/name, /*kind=*/kind)
//
class InstantiationMatcher {
 public:
  using is_gtest_matcher = void;

  explicit InstantiationMatcher(
      std::optional<::testing::Matcher<const std::string>> name,
      std::optional<InstantiationKind> kind)
      : name_(std::move(name)), kind_(kind) {}

  bool MatchAndExplain(const ::xls::Instantiation* instantiation,
                       ::testing::MatchResultListener* listener) const;

  void DescribeTo(::std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 protected:
  std::optional<::testing::Matcher<const std::string>> name_;
  std::optional<InstantiationKind> kind_;
};

inline ::testing::Matcher<const ::xls::Instantiation*> Instantiation() {
  return ::xls::op_matchers::InstantiationMatcher(std::nullopt, std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Instantiation*> Instantiation(T name)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::InstantiationMatcher(
      internal::NameMatcherInternal(std::string_view{name}), std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Instantiation*> Instantiation(
    T name, InstantiationKind kind)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::InstantiationMatcher(
      internal::NameMatcherInternal(std::string_view{name}), kind);
}

inline ::testing::Matcher<const ::xls::Instantiation*> Instantiation(
    ::testing::Matcher<const std::string> name, InstantiationKind kind) {
  return ::xls::op_matchers::InstantiationMatcher(std::move(name), kind);
}

inline InstantiationMatcher Instantiation(InstantiationKind kind) {
  return ::xls::op_matchers::InstantiationMatcher(std::nullopt, kind);
}

class InstantiationOutputMatcher : public NodeMatcher {
 public:
  using is_gtest_matcher = void;

  explicit InstantiationOutputMatcher(
      std::optional<::testing::Matcher<std::string>> port_name,
      std::optional<::testing::Matcher<const class Instantiation*>>
          instantiation)
      : NodeMatcher(Op::kInstantiationOutput, /*operands=*/{}),
        port_name_(std::move(port_name)),
        instantiation_(std::move(instantiation)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "did not match: ";
    DescribeTo(os);
  }

 private:
  std::optional<::testing::Matcher<std::string>> port_name_;
  std::optional<::testing::Matcher<const class Instantiation*>> instantiation_;
};

class InstantiationInputMatcher : public NodeMatcher {
 public:
  using is_gtest_matcher = void;

  explicit InstantiationInputMatcher(
      ::testing::Matcher<const ::xls::Node*> data,
      std::optional<::testing::Matcher<std::string>> name,
      std::optional<::testing::Matcher<const class Instantiation*>>
          instantiation)
      : NodeMatcher(Op::kInstantiationInput,
                    /*operands=*/{std::move(data)}),
        name_(std::move(name)),
        instantiation_(std::move(instantiation)) {}

  bool MatchAndExplain(const Node* node,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* os) const override;
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "did not match: ";
    DescribeTo(os);
  }

 private:
  std::optional<::testing::Matcher<std::string>> name_;
  std::optional<::testing::Matcher<const class Instantiation*>> instantiation_;
};

inline ::testing::Matcher<const ::xls::Node*> InstantiationOutput() {
  return ::xls::op_matchers::InstantiationOutputMatcher(std::nullopt,
                                                        std::nullopt);
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> InstantiationOutput(
    T port_name, std::optional<::testing::Matcher<const ::xls::Instantiation*>>
                     instantiation = std::nullopt)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::InstantiationOutputMatcher(
      internal::NameMatcherInternal(std::string_view{port_name}),
      std::move(instantiation));
}

inline ::testing::Matcher<const ::xls::Node*> InstantiationOutput(
    ::testing::Matcher<std::string> port_name,
    std::optional<::testing::Matcher<const ::xls::Instantiation*>>
        instantiation = std::nullopt) {
  return ::xls::op_matchers::InstantiationOutputMatcher(
      std::move(port_name), std::move(instantiation));
}

inline ::testing::Matcher<const ::xls::Node*> InstantiationInput(
    ::testing::Matcher<const ::xls::Node*> data =
        ::testing::A<const ::xls::Node*>()) {
  return ::xls::op_matchers::InstantiationInputMatcher(
      std::move(data), std::nullopt, std::nullopt);
}

inline ::testing::Matcher<const ::xls::Node*> InstantiationInput(
    ::testing::Matcher<const ::xls::Node*> data,
    ::testing::Matcher<std::string> name,
    std::optional<::testing::Matcher<const ::xls::Instantiation*>>
        instantiation = std::nullopt) {
  return ::xls::op_matchers::InstantiationInputMatcher(
      std::move(data), std::move(name), std::move(instantiation));
}

template <typename T>
inline ::testing::Matcher<const ::xls::Node*> InstantiationInput(
    ::testing::Matcher<const ::xls::Node*> data, T name,
    std::optional<::testing::Matcher<const ::xls::Instantiation*>>
        instantiation = std::nullopt)
  requires(std::is_convertible_v<T, std::string_view>)
{
  return ::xls::op_matchers::InstantiationInputMatcher(
      std::move(data), internal::NameMatcherInternal(std::string_view{name}),
      std::move(instantiation));
}

}  // namespace op_matchers
}  // namespace xls

#endif  // XLS_IR_IR_MATCHER_H_
