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
#include "xls/ir/nodes.h"

namespace xls {
namespace op_matchers {

bool NodeMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!node) {
    return false;
  }
  *listener << node->ToString();
  if (node->op() != op_) {
    *listener << " has incorrect op, expected: " << OpToString(op_);
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
  for (int64 index = 0; index < operands.size(); index++) {
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
  *os << op_;
  if (!operands_.empty()) {
    *os << "(";
    for (int i = 0; i < operands_.size(); i++) {
      if (i > 0) {
        *os << ", ";
      }
      operands_[i].DescribeTo(os);
    }
    *os << ")";
  }
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

void TypeMatcher::DescribeTo(std::ostream* os) const { *os << type_str_; }

bool NameMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (name_ == node->GetName()) {
    return true;
  }
  *listener << node->ToString() << " has incorrect name, expected: " << name_;
  return false;
}

void NameMatcher::DescribeTo(std::ostream* os) const { *os << name_; }

bool ParamMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (name_.has_value() && node->GetName() != *name_) {
    *listener << " has incorrect name, expected: " << *name_;
    return false;
  }
  return true;
}

bool BitSliceMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (start_.has_value() && node->As<::xls::BitSlice>()->start() != *start_) {
    *listener << " has incorrect start, expected: " << *start_;
    return false;
  }
  if (width_.has_value() && node->As<::xls::BitSlice>()->width() != *width_) {
    *listener << " has incorrect width, expected: " << *width_;
    return false;
  }
  return true;
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
      // The int64 expected value does not carry width information, so create a
      // value object with width matching the literal.
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

bool SendMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (channel_id_.has_value() &&
      *channel_id_ != node->As<::xls::Send>()->channel_id()) {
    *listener << " has incorrect channel id, expected: " << *channel_id_;
    return false;
  }
  return true;
}

bool SendIfMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (channel_id_.has_value() &&
      *channel_id_ != node->As<::xls::SendIf>()->channel_id()) {
    *listener << " has incorrect channel id, expected: " << *channel_id_;
    return false;
  }
  return true;
}

bool ReceiveMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (channel_id_.has_value() &&
      *channel_id_ != node->As<::xls::Receive>()->channel_id()) {
    *listener << " has incorrect channel id, expected: " << *channel_id_;
    return false;
  }
  return true;
}

bool ReceiveIfMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  if (!NodeMatcher::MatchAndExplain(node, listener)) {
    return false;
  }
  if (channel_id_.has_value() &&
      *channel_id_ != node->As<::xls::ReceiveIf>()->channel_id()) {
    *listener << " has incorrect channel id, expected: " << *channel_id_;
    return false;
  }
  return true;
}

}  // namespace op_matchers
}  // namespace xls
