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
#include "absl/strings/str_join.h"
#include "xls/ir/nodes.h"

namespace xls {
namespace op_matchers {

bool ChannelMatcher::MatchAndExplain(
    const ::xls::Channel* channel,
    ::testing::MatchResultListener* listener) const {
  if (!channel) {
    return false;
  }
  *listener << channel->ToString();
  if (id_.has_value() && channel->id() != id_.value()) {
    *listener << absl::StreamFormat(" has incorrect id (%d), expected: %d",
                                    channel->id(), id_.value());
    return false;
  }

  if (name_.has_value() && channel->name() != name_.value()) {
    *listener << absl::StreamFormat(" has incorrect name (%s), expected: %s",
                                    channel->name(), name_.value());
    return false;
  }

  if (kind_.has_value() && channel->kind() != kind_.value()) {
    *listener << absl::StreamFormat(" has incorrect kind (%s), expected: %s",
                                    ChannelKindToString(channel->kind()),
                                    ChannelKindToString(kind_.value()));
    return false;
  }
  return true;
}

void ChannelMatcher::DescribeTo(::std::ostream* os) const {
  std::vector<std::string> pieces;
  if (id_.has_value()) {
    pieces.push_back(absl::StrFormat("id=%d", id_.value()));
  }
  if (name_.has_value()) {
    pieces.push_back(absl::StrFormat("name=%s", name_.value()));
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

void ParamMatcher::DescribeTo(::std::ostream* os) const {
  if (name_.has_value()) {
    DescribeToHelper(os, {absl::StrFormat("name=\"%s\"", name_.value())});
  } else {
    DescribeToHelper(os, {});
  }
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
    int64_t channel_id, Package* package,
    const ::testing::Matcher<const ::xls::Channel*>& channel_matcher,
    ::testing::MatchResultListener* listener) {
  absl::StatusOr<::xls::Channel*> channel_status =
      package->GetChannel(channel_id);
  if (!channel_status.ok()) {
    *listener << " has an invalid channel id: " << channel_id;
    return false;
  }
  ::xls::Channel* ch = channel_status.value();
  return channel_matcher.MatchAndExplain(ch, listener);
}

static int64_t GetChannelId(const Node* node) {
  switch (node->op()) {
    case Op::kReceive:
      return node->As<::xls::Receive>()->channel_id();
    case Op::kReceiveIf:
      return node->As<::xls::ReceiveIf>()->channel_id();
    case Op::kSend:
      return node->As<::xls::Send>()->channel_id();
    case Op::kSendIf:
      return node->As<::xls::SendIf>()->channel_id();
    default:
      XLS_LOG(FATAL) << "Node is not a channel node: " << node->ToString();
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
  return MatchChannel(GetChannelId(node), node->package(),
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

bool InputPortMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  return matcher_.MatchAndExplain(node, listener);
}

void InputPortMatcher::DescribeTo(::std::ostream* os) const {
  matcher_.DescribeTo(os);
}

bool OutputPortMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  return matcher_.MatchAndExplain(node, listener);
}

void OutputPortMatcher::DescribeTo(::std::ostream* os) const {
  matcher_.DescribeTo(os);
}

bool RegisterMatcher::MatchAndExplain(
    const Node* node, ::testing::MatchResultListener* listener) const {
  // First match the output of the register. It should be a tuple-index of a
  // receive operation.
  if (!d_matcher_.MatchAndExplain(node, listener)) {
    return false;
  }
  if (q_matcher_.has_value()) {
    // If d_matching above was successful, then node must be a tuple-index of a
    // receive node. Find matching send node in proc.
    int64_t channel_id = node->As<::xls::TupleIndex>()
                             ->operand(0)
                             ->As<::xls::Receive>()
                             ->channel_id();
    // Find the send operation associated with the receive on the register
    // channel.
    // TODO(meheff): 2021/04/20 Find a better way of doing this.
    ::xls::Send* send = nullptr;
    for (Node* node : node->function_base()->nodes()) {
      if (node->Is<::xls::Send>() &&
          node->As<::xls::Send>()->channel_id() == channel_id) {
        send = node->As<::xls::Send>();
        break;
      }
    }
    if (send == nullptr) {
      *listener
          << " has no associated register send operation. IR may be malformed.";
      return false;
    }
    ::testing::StringMatchResultListener inner_listener;
    if (!q_matcher_->MatchAndExplain(send->data(), &inner_listener)) {
      if (listener->IsInterested()) {
        *listener << absl::StreamFormat(
            "\ndata (%s) of send operation (%s) assocated with receive "
            "(%s):\n\t%s\ndoesn't match expected:\n\t",
            send->data()->GetName(), send->GetName(),
            node->operand(0)->GetName(), send->data()->ToString());
        std::string explanation = inner_listener.str();
        if (!inner_listener.str().empty()) {
          *listener << inner_listener.str();
        }
      }
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

}  // namespace op_matchers
}  // namespace xls
