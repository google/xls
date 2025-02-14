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

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace op_matchers {

namespace internal {
bool NameMatcherInternal::MatchAndExplain(
    std::string_view name, ::testing::MatchResultListener* listener) const {
  if (name_ == name) {
    return true;
  }
  *listener << absl::StreamFormat("%s has incorrect name, expected: %s.", name,
                                  name_);
  return false;
}

void NameMatcherInternal::DescribeTo(std::ostream* os) const { *os << name_; }

void NameMatcherInternal::DescribeNegationTo(std::ostream* os) const {
  *os << absl::StreamFormat("name not %s", name_);
}
}  // namespace internal

bool ChannelMatcher::MatchAndExplain(
    ::xls::ChannelRef channel, ::testing::MatchResultListener* listener) const {
  *listener << ChannelRefToString(channel);

  if (name_.has_value() &&
      !name_->MatchAndExplain(std::string{ChannelRefName(channel)}, listener)) {
    return false;
  }

  if (kind_.has_value() && ChannelRefKind(channel) != kind_.value()) {
    *listener << absl::StreamFormat(
        " has incorrect kind (%s), expected: %s",
        ChannelKindToString(ChannelRefKind(channel)),
        ChannelKindToString(kind_.value()));
    return false;
  }
  if (type_string_.has_value() &&
      ChannelRefType(channel)->ToString() != type_string_.value()) {
    *listener << absl::StreamFormat(" has incorrect type (%s), expected: %s",
                                    ChannelRefType(channel)->ToString(),
                                    type_string_.value());
    return false;
  }
  return true;
}

void ChannelMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> pieces;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    name_->DescribeTo(&ss);
    ss << '"';
    pieces.push_back(ss.str());
  }
  if (kind_.has_value()) {
    pieces.push_back(
        absl::StrFormat("kind=%s", ChannelKindToString(kind_.value())));
  }
  *os << absl::StreamFormat("channel(%s)", absl::StrJoin(pieces, ", "));
}

bool NodeMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!node) {
    return false;
  }
  if (node->op() != op_) {
    *listener << " has incorrect op (" << OpToString(node->op())
              << "), expected: " << OpToString(op_);
    return false;
  }

  // If no operands are specified, then matching stops here.
  if (operands_.empty()) {
    return true;
  }
  const auto& operands = node->operands();
  if (operands.size() != operands_.size()) {
    *listener << " has too "
              << (operands.size() > operands_.size() ? "many" : "few")
              << " operands (got " << operands.size() << ", want "
              << operands_.size() << ")";
    return false;
  }
  for (int64_t index = 0; index < operands.size(); index++) {
    ::testing::StringMatchResultListener inner_listener;
    if (!operands_[index].MatchAndExplain(operands[index], &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << "\noperand " << index << ":\n\t"
                  << operands[index]->ToString()
                  << "\ndoesn't match expected:\n\t";
        operands_[index].DescribeTo(listener->stream());
        std::string explanation = inner_listener.str();
        if (!explanation.empty()) {
          *listener << ", " << explanation;
        }
      }
      return false;
    }
  }
  return true;
}

void NodeMatcher::DescribeTo(::std::ostream* os) const {
  DescribeToHelper(os, {});
}

void NodeMatcher::DescribeToHelper(
    ::std::ostream* os, absl::Span<const std::string> additional_fields) const {
  *os << op_;
  std::vector<std::string> all_fields;
  for (int i = 0; i < operands_.size(); i++) {
    std::ostringstream sub_os;
    operands_[i].DescribeTo(&sub_os);
    all_fields.push_back(sub_os.str());
  }
  all_fields.insert(all_fields.end(), additional_fields.begin(),
                    additional_fields.end());
  *os << absl::StreamFormat("(%s)", absl::StrJoin(all_fields, ", "));
}

bool TypeMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (type_str_ == node->GetType()->ToString()) {
    return true;
  }
  *listener << node->ToString()
            << " has incorrect type, expected: " << type_str_;
  return false;
}
bool TypeMatcher::MatchAndExplain(
    const ::xls::Type* type, ::testing::MatchResultListener* listener) const {
  if (type_str_ == type->ToString()) {
    return true;
  }
  *listener << type->ToString()
            << " has incorrect type, expected: " << type_str_;
  return false;
}

void TypeMatcher::DescribeTo(std::ostream* os) const { *os << type_str_; }

void TypeMatcher::DescribeNegationTo(std::ostream* os) const {
  *os << "is not " << type_str_;
}

bool StateElementMatcher::MatchAndExplain(
    const class StateElement* state_element,
    ::testing::MatchResultListener* listener) const {
  if (!state_element) {
    return false;
  }
  bool match = true;
  if (::testing::StringMatchResultListener inner_listener;
      name_matcher_.has_value() &&
      !name_matcher_->MatchAndExplain(state_element->name(), &inner_listener)) {
    if (listener->IsInterested()) {
      if (match) {
        *listener << "State element has";
      }
      *listener << " incorrect name, expected: ";
      name_matcher_->DescribeTo(listener->stream());
      std::string explanation = inner_listener.str();
      if (!explanation.empty()) {
        *listener << ", " << explanation;
      }
    }
    match = false;
  }
  if (::testing::StringMatchResultListener inner_listener;
      type_matcher_.has_value() &&
      !type_matcher_->MatchAndExplain(state_element->type(), &inner_listener)) {
    if (listener->IsInterested()) {
      if (match) {
        *listener << "State element has";
      }
      *listener << " incorrect type, expected: ";
      type_matcher_->DescribeTo(listener->stream());
      std::string explanation = inner_listener.str();
      if (!explanation.empty()) {
        *listener << ", " << explanation;
      }
    }
    match = false;
  }
  if (::testing::StringMatchResultListener inner_listener;
      initial_value_matcher_.has_value() &&
      !initial_value_matcher_->MatchAndExplain(state_element->initial_value(),
                                               &inner_listener)) {
    if (listener->IsInterested()) {
      if (match) {
        *listener << "State element has";
      }
      *listener << " incorrect initial value, expected: ";
      initial_value_matcher_->DescribeTo(listener->stream());
      std::string explanation = inner_listener.str();
      if (!explanation.empty()) {
        *listener << ", " << explanation;
      }
    }
    match = false;
  }
  return match;
}

void StateElementMatcher::DescribeTo(::std::ostream* os) const {
  *os << "state element";
  if (name_matcher_.has_value()) {
    *os << " with name ";
    name_matcher_->DescribeTo(os);
  }
}

bool NameMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  return inner_matcher_.MatchAndExplain(node->GetName(), listener);
}

void NameMatcher::DescribeTo(std::ostream* os) const {
  inner_matcher_.DescribeTo(os);
}

void NameMatcher::DescribeNegationTo(std::ostream* os) const {
  inner_matcher_.DescribeNegationTo(os);
}

bool ParamMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (name_matcher_.has_value()) {
    return name_matcher_->MatchAndExplain(node->GetName(), listener);
  }
  return true;
}

void ParamMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (name_matcher_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    name_matcher_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool ArrayIndexMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (!assumed_in_bounds_.MatchAndExplain(
          node->As<::xls::ArrayIndex>()->assumed_in_bounds(), listener)) {
    *listener << "Unexpected value of assumed_in_bounds for " << node;
    return false;
  }
  return true;
}
bool ArrayUpdateMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (!assumed_in_bounds_.MatchAndExplain(
          node->As<::xls::ArrayUpdate>()->assumed_in_bounds(), listener)) {
    *listener << "Unexpected value of assumed_in_bounds for " << node;
    return false;
  }
  return true;
}

bool BitSliceMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (!start_.Matches(node->As<::xls::BitSlice>()->start())) {
    if (listener->IsInterested()) {
      *listener << "has incorrect start, expected start ";
      start_.DescribeTo(listener->stream());
    }
    return false;
  }
  if (!width_.Matches(node->As<::xls::BitSlice>()->width())) {
    if (listener->IsInterested()) {
      *listener << "has incorrect width, expected width ";
      width_.DescribeTo(listener->stream());
    }
    return false;
  }
  return true;
}

void BitSliceMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  additional_fields.reserve(2);
  {
    std::stringstream ss;
    ss << "start ";
    start_.DescribeTo(&ss);
    additional_fields.push_back(ss.str());
  }
  {
    std::stringstream ss;
    ss << "width ";
    width_.DescribeTo(&ss);
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool DynamicBitSliceMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (width_.has_value() &&
      node->As<::xls::DynamicBitSlice>()->width() != *width_) {
    *listener << " has incorrect width, expected: " << *width_;
    return false;
  }
  return true;
}

bool LiteralMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  const Value& literal_value = node->As<::xls::Literal>()->value();
  if (value_.has_value() || uint64_value_.has_value()) {
    Value expected;
    if (value_.has_value()) {
      expected = *value_;
    } else {
      // The int64_t expected value does not carry width information, so create
      // a value object with width matching the literal.
      if (!literal_value.IsBits() || Bits::MinBitCountUnsigned(*uint64_value_) >
                                         literal_value.bits().bit_count()) {
        // Literal value isn't a Bits value or it is too narrow to hold the
        // expected value. Just create a 64-bit Bits value for the error
        // message.
        expected = Value(UBits(*uint64_value_, 64));
      } else {
        expected =
            Value(UBits(*uint64_value_, literal_value.bits().bit_count()));
      }
    }
    if (literal_value != expected) {
      *listener << " has value " << literal_value.ToString(format_)
                << ", expected: " << expected.ToString(format_);
      return false;
    }
  }

  return true;
}

bool DynamicCountedForMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (node->As<::xls::DynamicCountedFor>()->body() != body_) {
    *listener << " has incorrect body function, expected: " << body_->name();
    return false;
  }
  return true;
}

bool SelectMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (default_value_.has_value() &&
      !node->As<::xls::Select>()->default_value().has_value()) {
    *listener << " has no default value, expected: " << (*default_value_);
    return false;
  }
  if (!default_value_.has_value() &&
      node->As<::xls::Select>()->default_value().has_value()) {
    *listener << " has default value, expected no default value";
    return false;
  }
  return true;
}

bool OneHotMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (priority_.has_value() &&
      *priority_ != node->As<::xls::OneHot>()->priority()) {
    *listener << " has incorrect priority, expected: lsb_prio="
              << (*priority_ == LsbOrMsb::kLsb ? "true" : "false");
    return false;
  }
  return true;
}

bool TupleIndexMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (index_.has_value() && *index_ != node->As<::xls::TupleIndex>()->index()) {
    *listener << " has incorrect index, expected: " << *index_;
    return false;
  }
  return true;
}

static bool MatchChannel(
    std::string_view channel, ::xls::Proc* proc, ChannelDirection direction,
    const ::testing::Matcher<::xls::ChannelRef>& channel_matcher,
    ::testing::MatchResultListener* listener) {
  absl::StatusOr<::xls::ChannelRef> channel_status =
      proc->GetChannelRef(channel, direction);
  if (!channel_status.ok()) {
    *listener << " has an invalid channel name: " << channel;
    return false;
  }
  ::xls::ChannelRef ch_ref = channel_status.value();
  return channel_matcher.MatchAndExplain(ch_ref, listener);
}

static std::string_view GetChannelName(const Node* node) {
  switch (node->op()) {
    case Op::kReceive:
      return node->As<::xls::Receive>()->channel_name();
    case Op::kSend:
      return node->As<::xls::Send>()->channel_name();
    default:
      LOG(FATAL) << "Node is not a channel node: " << node->ToString();
  }
}

bool ChannelNodeMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (!channel_matcher_.has_value()) {
    return true;
  }
  ChannelDirection direction;
  if (node->Is<::xls::Send>()) {
    direction = ChannelDirection::kSend;
  } else if (node->Is<::xls::Receive>()) {
    direction = ChannelDirection::kReceive;
  } else {
    LOG(FATAL) << absl::StrFormat(
        "Expected send or receive node, got node `%s` with op `%s`",
        node->GetName(), OpToString(node->op()));
  }
  return MatchChannel(GetChannelName(node),
                      node->function_base()->AsProcOrDie(), direction,
                      channel_matcher_.value(), listener);
}

void ChannelNodeMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> fields;
  if (channel_matcher_.has_value()) {
    std::ostringstream sub_os;
    channel_matcher_.value().DescribeTo(&sub_os);
    fields.push_back(sub_os.str());
  }
  NodeMatcher::DescribeToHelper(os, fields);
}

bool TraceVerbosityMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (!verbosity_.MatchAndExplain(node->As<::xls::Trace>()->verbosity(),
                                  listener)) {
    return false;
  }
  return true;
}

void TraceVerbosityMatcher::DescribeTo(::std::ostream* os) const {
  std::stringstream ss;
  ss << "trace_verbosity=\"";
  verbosity_.DescribeTo(&ss);
  ss << '"';
  std::array<std::string, 1> additional_fields = {ss.str()};
  DescribeToHelper(os, additional_fields);
}

bool InputPortMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (name_matcher_.has_value() &&
      !name_matcher_->MatchAndExplain(node->GetName(), listener)) {
    return false;
  }
  return true;
}

void InputPortMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (name_matcher_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    name_matcher_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool OutputPortMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (name_matcher_.has_value() &&
      !name_matcher_->MatchAndExplain(node->GetName(), listener)) {
    return false;
  }
  return true;
}

void OutputPortMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (name_matcher_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    name_matcher_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool StateReadMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (state_element_name_.has_value() &&
      !state_element_name_->Matches(
          node->As<xls::StateRead>()->state_element()->name())) {
    *listener << " has incorrect state element ("
              << node->As<xls::StateRead>()->state_element()->name()
              << "), expected: ";
    if (listener->stream() != nullptr) {
      state_element_name_->DescribeTo(listener->stream());
    }
    return false;
  }
  return true;
}

void StateReadMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (state_element_name_.has_value()) {
    std::stringstream ss;
    ss << "state_element=\"";
    state_element_name_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool RegisterReadMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (register_name_.has_value() &&
      !register_name_->Matches(
          node->As<xls::RegisterRead>()->GetRegister()->name())) {
    *listener << " has incorrect register ("
              << node->As<xls::RegisterRead>()->GetRegister()->name()
              << "), expected: ";
    if (listener->stream() != nullptr) {
      register_name_->DescribeTo(listener->stream());
    }
    return false;
  }
  return true;
}

void RegisterReadMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (register_name_.has_value()) {
    std::stringstream ss;
    ss << "register=\"";
    register_name_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool RegisterWriteMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (register_name_.has_value() &&
      !register_name_->Matches(
          node->As<xls::RegisterWrite>()->GetRegister()->name())) {
    *listener << " has incorrect register ("
              << node->As<xls::RegisterWrite>()->GetRegister()->name()
              << "), expected: ";

    if (listener->stream() != nullptr) {
      register_name_->DescribeTo(listener->stream());
    }
    return false;
  }
  return true;
}

void RegisterWriteMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (register_name_.has_value()) {
    std::stringstream ss;
    ss << "register=\"";
    register_name_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(ss.str());
  }
  DescribeToHelper(os, additional_fields);
}

bool RegisterMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  // First match the output of the register. It should be a tuple-index of a
  // receive operation.
  if (!d_matcher_.MatchAndExplain(node, listener)) {
    return false;
  }
  if (q_matcher_.has_value()) {
    const std::string& register_name =
        node->As<::xls::RegisterRead>()->GetRegister()->name();
    xls::RegisterWrite* reg_write = nullptr;
    for (Node* node : node->function_base()->nodes()) {
      if (node->Is<::xls::RegisterWrite>() &&
          node->As<::xls::RegisterWrite>()->GetRegister()->name() ==
              register_name) {
        reg_write = node->As<::xls::RegisterWrite>();
        break;
      }
    }
    if (reg_write == nullptr) {
      *listener << " has no associated register write operation. IR may be "
                   "malformed.";
      return false;
    }

    if (!q_matcher_->MatchAndExplain(reg_write, listener)) {
      return false;
    }
  }
  return true;
}

void RegisterMatcher::DescribeTo(::std::ostream* os) const {
  *os << "register(";
  if (q_matcher_.has_value()) {
    q_matcher_->DescribeTo(os);
  }
  *os << ")";
}

void RegisterMatcher::DescribeNegationTo(::std::ostream* os) const {
  *os << "did not match register(";
  if (q_matcher_.has_value()) {
    q_matcher_->DescribeTo(os);
  }
  *os << ")";
}

bool FunctionBaseMatcher::MatchAndExplain(
    const ::xls::FunctionBase* fb,
    ::testing::MatchResultListener* listener) const {
  if (fb == nullptr) {
    return false;
  }
  *listener << fb->name();
  if (name_.has_value() && !name_->MatchAndExplain(fb->name(), listener)) {
    return false;
  }
  return true;
}

void FunctionBaseMatcher::DescribeTo(::std::ostream* os) const {
  std::string name_str;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << "name=";
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("FunctionBase(%s)", name_str);
}

void ProcMatcher::DescribeTo(::std::ostream* os) const {
  std::optional<std::string> name_str;
  if (name_.has_value()) {
    std::stringstream ss;
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("proc %s { ... }",
                            name_str.value_or("<unspecified>"));
}

void ProcMatcher::DescribeNegationTo(std::ostream* os) const {
  std::string name_str;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << " named ";
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("FunctionBase was not a proc%s.", name_str);
}

void FunctionMatcher::DescribeTo(::std::ostream* os) const {
  std::stringstream ss;
  std::optional<std::string> name_str;
  if (name_.has_value()) {
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("fn %s { ... }",
                            name_str.value_or("<unspecified>"));
}

void FunctionMatcher::DescribeNegationTo(std::ostream* os) const {
  std::string name_str;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << " named ";
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("FunctionBase was not a function%s.", name_str);
}

void BlockMatcher::DescribeTo(::std::ostream* os) const {
  std::stringstream ss;
  std::optional<std::string> name_str;
  if (name_.has_value()) {
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("block %s { ... }",
                            name_str.value_or("<unspecified>"));
}

void BlockMatcher::DescribeNegationTo(std::ostream* os) const {
  std::string name_str;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << " named ";
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("FunctionBase was not a block%s.", name_str);
}

bool MinDelayMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  int64_t delay = node->As<::xls::MinDelay>()->delay();
  if (delay_.Matches(delay)) {
    return true;
  }
  if (listener->IsInterested()) {
    *listener << "delay " << delay << " ";
    delay_.DescribeNegationTo(listener->stream());
  }
  return false;
}

void MinDelayMatcher::DescribeTo(::std::ostream* os) const {
  std::stringstream delay_description;
  delay_.DescribeTo(&delay_description);
  DescribeToHelper(os, {absl::StrCat("delay=", delay_description.str())});
}

bool InstantiationMatcher::MatchAndExplain(
    const ::xls::Instantiation* instantiation,
    ::testing::MatchResultListener* listener) const {
  if (name_.has_value() &&
      !name_->MatchAndExplain(std::string{instantiation->name()}, listener)) {
    return false;
  }

  if (kind_.has_value() && *kind_ != instantiation->kind()) {
    *listener << absl::StreamFormat("%s has incorrect kind, expected: %v",
                                    instantiation->name(), *kind_);
    return false;
  }
  return true;
}

void InstantiationMatcher::DescribeTo(::std::ostream* os) const {
  std::string kind_str;
  if (kind_.has_value()) {
    kind_str = absl::StrFormat("(kind=%s)", InstantiationKindToString(*kind_));
  }
  std::string name_str = "<unspecified>";
  if (name_.has_value()) {
    std::stringstream ss;
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("instantiation %s%s", name_str, kind_str);
}
void InstantiationMatcher::DescribeNegationTo(std::ostream* os) const {
  std::string kind_str;
  if (kind_.has_value()) {
    kind_str = absl::StrFormat("kind=%s", InstantiationKindToString(*kind_));
  }
  std::string name_str = "<unspecified>";
  if (name_.has_value()) {
    std::stringstream ss;
    name_->DescribeTo(&ss);
    name_str = ss.str();
  }
  *os << absl::StreamFormat("Instantiation did not have (name=%s, kind=%s)",
                            name_str, kind_str);
}

bool InstantiationOutputMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (port_name_.has_value() &&
      !port_name_->MatchAndExplain(
          node->As<::xls::InstantiationOutput>()->port_name(), listener)) {
    return false;
  }
  if (instantiation_.has_value() &&
      !instantiation_->MatchAndExplain(
          node->As<::xls::InstantiationOutput>()->instantiation(), listener)) {
    return false;
  }
  return true;
}

void InstantiationOutputMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (port_name_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    port_name_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(std::move(ss).str());
  }
  if (instantiation_.has_value()) {
    std::stringstream ss;
    ss << "instantiation=\"";
    instantiation_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(std::move(ss).str());
  }
  DescribeToHelper(os, additional_fields);
}

bool InstantiationInputMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (name_.has_value() &&
      !name_->MatchAndExplain(node->As<xls::InstantiationInput>()->port_name(),
                              listener)) {
    return false;
  }
  if (instantiation_.has_value() &&
      !instantiation_->MatchAndExplain(
          node->As<::xls::InstantiationInput>()->instantiation(), listener)) {
    return false;
  }
  return true;
}

void InstantiationInputMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> additional_fields;
  if (name_.has_value()) {
    std::stringstream ss;
    ss << "name=\"";
    name_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(std::move(ss).str());
  }
  if (instantiation_.has_value()) {
    std::stringstream ss;
    ss << "instantiation=\"";
    instantiation_->DescribeTo(&ss);
    ss << '"';
    additional_fields.push_back(std::move(ss).str());
  }
  DescribeToHelper(os, additional_fields);
}

}  // namespace op_matchers
}  // namespace xls
