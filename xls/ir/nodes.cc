// Copyright 2024 The XLS Authors
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

#include "xls/ir/nodes.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

namespace {

Type* GetTupleType(Package* package, absl::Span<Node* const> operands) {
  std::vector<Type*> operand_types;
  for (Node* operand : operands) {
    operand_types.push_back(operand->GetType());
  }
  return package->GetTupleType(operand_types);
}

Type* GetConcatType(Package* package, absl::Span<Node* const> operands) {
  int64_t width = 0;
  for (Node* operand : operands) {
    width += operand->BitCountOrDie();
  }
  return package->GetBitsType(width);
}

Type* GetArrayConcatType(Package* package, absl::Span<Node* const> operands) {
  int64_t size = 0;
  Type* element_type = nullptr;

  for (Node* operand : operands) {
    auto operand_type = operand->GetType()->AsArrayOrDie();

    size += operand_type->size();
    if (element_type == nullptr) {
      // Set element_type to the first operand's element type
      element_type = operand_type->element_type();
    }
  }

  CHECK(element_type);
  return package->GetArrayType(size, element_type);
}

Type* GetMapType(Node* operand, Function* to_apply) {
  return operand->package()->GetArrayType(
      operand->GetType()->AsArrayOrDie()->size(),
      to_apply->return_value()->GetType());
}

Type* GetReceivePayloadType(FunctionBase* function_base,
                            std::string_view channel_name) {
  return function_base->AsProcOrDie()
      ->GetChannelReferenceType(channel_name)
      .value();
}

Type* GetReceiveType(FunctionBase* function_base, std::string_view channel_name,
                     bool is_blocking) {
  if (is_blocking) {
    return function_base->package()->GetTupleType(
        {function_base->package()->GetTokenType(),
         GetReceivePayloadType(function_base, channel_name)});
  }

  return function_base->package()->GetTupleType(
      {function_base->package()->GetTokenType(),
       GetReceivePayloadType(function_base, channel_name),
       function_base->package()->GetBitsType(1)});
}

}  // namespace

AfterAll::AfterAll(const SourceInfo& loc, absl::Span<Node* const> dependencies,
                   std::string_view name, FunctionBase* function)
    : Node(Op::kAfterAll, function->package()->GetTokenType(), loc, name,
           function) {
  CHECK(IsOpClass<AfterAll>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `AfterAll`.";
  AddOperands(dependencies);
}

absl::StatusOr<Node*> AfterAll::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<AfterAll>(loc(), new_operands,
                                                  GetNameView());
}

MinDelay::MinDelay(const SourceInfo& loc, Node* token, int64_t delay,
                   std::string_view name, FunctionBase* function)
    : Node(Op::kMinDelay, function->package()->GetTokenType(), loc, name,
           function),
      delay_(delay) {
  CHECK(IsOpClass<MinDelay>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `MinDelay`.";
  AddOperand(token);
}

absl::StatusOr<Node*> MinDelay::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<MinDelay>(loc(), new_operands[0],
                                                  delay(), GetNameView());
}

bool MinDelay::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return delay_ == other->As<MinDelay>()->delay_;
}

Array::Array(const SourceInfo& loc, absl::Span<Node* const> elements,
             Type* element_type, std::string_view name, FunctionBase* function)
    : Node(Op::kArray,
           function->package()->GetArrayType(elements.size(), element_type),
           loc, name, function),
      element_type_(element_type) {
  CHECK(IsOpClass<Array>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Array`.";
  AddOperands(elements);
}

bool Array::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return element_type_ == other->As<Array>()->element_type_;
}

ArrayIndex::ArrayIndex(const SourceInfo& loc, Node* arg,
                       absl::Span<Node* const> indices, std::string_view name,
                       FunctionBase* function)
    : Node(Op::kArrayIndex,
           GetIndexedElementType(arg->GetType(), indices.size()).value(), loc,
           name, function) {
  CHECK(IsOpClass<ArrayIndex>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ArrayIndex`.";
  AddOperand(arg);
  AddOperands(indices);
}

ArraySlice::ArraySlice(const SourceInfo& loc, Node* array, Node* start,
                       int64_t width, std::string_view name,
                       FunctionBase* function)
    : Node(Op::kArraySlice,
           function->package()->GetArrayType(
               width, array->GetType()->AsArrayOrDie()->element_type()),
           loc, name, function),
      width_(width) {
  CHECK(IsOpClass<ArraySlice>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ArraySlice`.";
  AddOperand(array);
  AddOperand(start);
}

absl::StatusOr<Node*> ArraySlice::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<ArraySlice>(
      loc(), new_operands[0], new_operands[1], width(), GetNameView());
}

bool ArraySlice::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return width_ == other->As<ArraySlice>()->width_;
}

ArrayUpdate::ArrayUpdate(const SourceInfo& loc, Node* arg, Node* update_value,
                         absl::Span<Node* const> indices, std::string_view name,
                         FunctionBase* function)
    : Node(Op::kArrayUpdate, arg->GetType(), loc, name, function) {
  CHECK(IsOpClass<ArrayUpdate>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ArrayUpdate`.";
  AddOperand(arg);
  AddOperand(update_value);
  AddOperands(indices);
}

ArrayConcat::ArrayConcat(const SourceInfo& loc, absl::Span<Node* const> args,
                         std::string_view name, FunctionBase* function)
    : Node(Op::kArrayConcat, GetArrayConcatType(function->package(), args), loc,
           name, function) {
  CHECK(IsOpClass<ArrayConcat>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ArrayConcat`.";
  AddOperands(args);
}

absl::StatusOr<Node*> ArrayConcat::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<ArrayConcat>(loc(), new_operands,
                                                     GetNameView());
}

BinOp::BinOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op,
             std::string_view name, FunctionBase* function)
    : Node(op, lhs->GetType(), loc, name, function) {
  CHECK(IsOpClass<BinOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `BinOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
}

absl::StatusOr<Node*> BinOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<BinOp>(
      loc(), new_operands[0], new_operands[1], op(), GetNameView());
}

ArithOp::ArithOp(const SourceInfo& loc, Node* lhs, Node* rhs, int64_t width,
                 Op op, std::string_view name, FunctionBase* function)
    : Node(op, function->package()->GetBitsType(width), loc, name, function),
      width_(width) {
  CHECK(IsOpClass<ArithOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ArithOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
}

absl::StatusOr<Node*> ArithOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<ArithOp>(
      loc(), new_operands[0], new_operands[1], width(), op(), GetNameView());
}

bool ArithOp::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return width_ == other->As<ArithOp>()->width_;
}

PartialProductOp::PartialProductOp(const SourceInfo& loc, Node* lhs, Node* rhs,
                                   int64_t width, Op op, std::string_view name,
                                   FunctionBase* function)
    : Node(op,
           function->package()->GetTupleType(
               {function->package()->GetBitsType(width),
                function->package()->GetBitsType(width)}),
           loc, name, function),
      width_(width) {
  CHECK(IsOpClass<PartialProductOp>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `PartialProductOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
}

absl::StatusOr<Node*> PartialProductOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<PartialProductOp>(
      loc(), new_operands[0], new_operands[1], width(), op(), GetNameView());
}

bool PartialProductOp::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return width_ == other->As<PartialProductOp>()->width_;
}

Assert::Assert(const SourceInfo& loc, Node* token, Node* condition,
               std::string_view message, std::optional<std::string> label,
               std::optional<std::string> original_label, std::string_view name,
               FunctionBase* function)
    : Node(Op::kAssert, function->package()->GetTokenType(), loc, name,
           function),
      message_(message),
      label_(std::move(label)),
      original_label_(std::move(original_label)) {
  CHECK(IsOpClass<Assert>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Assert`.";
  AddOperand(token);
  AddOperand(condition);
}

absl::StatusOr<Node*> Assert::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Assert>(
      loc(), new_operands[0], new_operands[1], message(), label(),
      original_label(), GetNameView());
}

bool Assert::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return message_ == other->As<Assert>()->message_ &&
         label_ == other->As<Assert>()->label_ &&
         original_label_ == other->As<Assert>()->original_label_;
}

Trace::Trace(const SourceInfo& loc, Node* token, Node* condition,
             absl::Span<Node* const> args, absl::Span<FormatStep const> format,
             int64_t verbosity, std::string_view name, FunctionBase* function)
    : Node(Op::kTrace, function->package()->GetTokenType(), loc, name,
           function),
      format_(format.begin(), format.end()),
      verbosity_(verbosity) {
  CHECK(IsOpClass<Trace>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Trace`.";
  AddOperand(token);
  AddOperand(condition);
  AddOperands(args);
}

bool Trace::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return format_ == other->As<Trace>()->format_ &&
         verbosity_ == other->As<Trace>()->verbosity_;
}

Cover::Cover(const SourceInfo& loc, Node* condition, std::string_view label,
             std::optional<std::string> original_label, std::string_view name,
             FunctionBase* function)
    : Node(Op::kCover, function->package()->GetTupleType({}), loc, name,
           function),
      label_(label),
      original_label_(std::move(original_label)) {
  CHECK(IsOpClass<Cover>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Cover`.";
  AddOperand(condition);
}

absl::StatusOr<Node*> Cover::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Cover>(loc(), new_operands[0], label(),
                                               original_label(), GetNameView());
}

bool Cover::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return label_ == other->As<Cover>()->label_ &&
         original_label_ == other->As<Cover>()->original_label_;
}

BitwiseReductionOp::BitwiseReductionOp(const SourceInfo& loc, Node* operand,
                                       Op op, std::string_view name,
                                       FunctionBase* function)
    : Node(op, function->package()->GetBitsType(1), loc, name, function) {
  CHECK(IsOpClass<BitwiseReductionOp>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `BitwiseReductionOp`.";
  AddOperand(operand);
}

absl::StatusOr<Node*> BitwiseReductionOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<BitwiseReductionOp>(
      loc(), new_operands[0], op(), GetNameView());
}

absl::StatusOr<ChannelRef> ChannelNode::GetChannelRef() const {
  Proc* proc = function_base()->AsProcOrDie();
  if (proc->is_new_style_proc()) {
    return proc->GetChannelReference(channel_name(), direction());
  }
  return package()->GetChannel(channel_name());
}

Type* ChannelNode::GetPayloadType() const {
  return function_base()
      ->AsProcOrDie()
      ->GetChannelReferenceType(channel_name())
      .value();
}

absl::Status ChannelNode::ReplaceChannel(std::string_view new_channel_name) {
  Proc* proc = function_base()->AsProcOrDie();
  if (proc->is_new_style_proc()) {
    XLS_RETURN_IF_ERROR(
        proc->GetChannelReference(channel_name(), direction()).status());
  } else {
    XLS_RETURN_IF_ERROR(package()->GetChannel(new_channel_name).status());
  }
  channel_name_ = new_channel_name;
  return absl::OkStatus();
}

Receive::Receive(const SourceInfo& loc, Node* token,
                 std::optional<Node*> predicate, std::string_view channel_name,
                 bool is_blocking, std::string_view name,
                 FunctionBase* function)
    : ChannelNode(loc, Op::kReceive,
                  GetReceiveType(function, channel_name, is_blocking),
                  channel_name, Direction::kReceive, name, function),
      is_blocking_(is_blocking),
      has_predicate_(predicate.has_value()) {
  CHECK(IsOpClass<Receive>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Receive`.";
  AddOperand(token);
  AddOptionalOperand(predicate);
}

bool Receive::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return channel_name() == other->As<Receive>()->channel_name() &&
         is_blocking_ == other->As<Receive>()->is_blocking_ &&
         has_predicate_ == other->As<Receive>()->has_predicate_;
}

Send::Send(const SourceInfo& loc, Node* token, Node* data,
           std::optional<Node*> predicate, std::string_view channel_name,
           std::string_view name, FunctionBase* function)
    : ChannelNode(loc, Op::kSend, function->package()->GetTokenType(),
                  channel_name, Direction::kSend, name, function),
      has_predicate_(predicate.has_value()) {
  CHECK(IsOpClass<Send>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Send`.";
  AddOperand(token);
  AddOperand(data);
  AddOptionalOperand(predicate);
}

bool Send::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return channel_name() == other->As<Send>()->channel_name() &&
         has_predicate_ == other->As<Send>()->has_predicate_;
}

NaryOp::NaryOp(const SourceInfo& loc, absl::Span<Node* const> args, Op op,
               std::string_view name, FunctionBase* function)
    : Node(op, args[0]->GetType(), loc, name, function) {
  CHECK(IsOpClass<NaryOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `NaryOp`.";
  AddOperands(args);
}

absl::StatusOr<Node*> NaryOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<NaryOp>(loc(), new_operands, op(),
                                                GetNameView());
}

BitSlice::BitSlice(const SourceInfo& loc, Node* arg, int64_t start,
                   int64_t width, std::string_view name, FunctionBase* function)
    : Node(Op::kBitSlice, function->package()->GetBitsType(width), loc, name,
           function),
      start_(start),
      width_(width) {
  CHECK(IsOpClass<BitSlice>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `BitSlice`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> BitSlice::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<BitSlice>(
      loc(), new_operands[0], start(), width(), GetNameView());
}

bool BitSlice::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return start_ == other->As<BitSlice>()->start_ &&
         width_ == other->As<BitSlice>()->width_;
}

DynamicBitSlice::DynamicBitSlice(const SourceInfo& loc, Node* arg, Node* start,
                                 int64_t width, std::string_view name,
                                 FunctionBase* function)
    : Node(Op::kDynamicBitSlice, function->package()->GetBitsType(width), loc,
           name, function),
      width_(width) {
  CHECK(IsOpClass<DynamicBitSlice>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `DynamicBitSlice`.";
  AddOperand(arg);
  AddOperand(start);
}

absl::StatusOr<Node*> DynamicBitSlice::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<DynamicBitSlice>(
      loc(), new_operands[0], new_operands[1], width(), GetNameView());
}

bool DynamicBitSlice::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return width_ == other->As<DynamicBitSlice>()->width_;
}

BitSliceUpdate::BitSliceUpdate(const SourceInfo& loc, Node* arg, Node* start,
                               Node* value, std::string_view name,
                               FunctionBase* function)
    : Node(Op::kBitSliceUpdate, arg->GetType(), loc, name, function) {
  CHECK(IsOpClass<BitSliceUpdate>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `BitSliceUpdate`.";
  AddOperand(arg);
  AddOperand(start);
  AddOperand(value);
}

absl::StatusOr<Node*> BitSliceUpdate::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<BitSliceUpdate>(
      loc(), new_operands[0], new_operands[1], new_operands[2], GetNameView());
}

CompareOp::CompareOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op,
                     std::string_view name, FunctionBase* function)
    : Node(op, function->package()->GetBitsType(1), loc, name, function) {
  CHECK(IsOpClass<CompareOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `CompareOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
}

absl::StatusOr<Node*> CompareOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<CompareOp>(
      loc(), new_operands[0], new_operands[1], op(), GetNameView());
}

Concat::Concat(const SourceInfo& loc, absl::Span<Node* const> args,
               std::string_view name, FunctionBase* function)
    : Node(Op::kConcat, GetConcatType(function->package(), args), loc, name,
           function) {
  CHECK(IsOpClass<Concat>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Concat`.";
  AddOperands(args);
}

absl::StatusOr<Node*> Concat::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Concat>(loc(), new_operands,
                                                GetNameView());
}

CountedFor::CountedFor(const SourceInfo& loc, Node* initial_value,
                       absl::Span<Node* const> invariant_args,
                       int64_t trip_count, int64_t stride, Function* body,
                       std::string_view name, FunctionBase* function)
    : Node(Op::kCountedFor, initial_value->GetType(), loc, name, function),
      trip_count_(trip_count),
      stride_(stride),
      body_(body) {
  CHECK(IsOpClass<CountedFor>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `CountedFor`.";
  AddOperand(initial_value);
  AddOperands(invariant_args);
}

bool CountedFor::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return trip_count_ == other->As<CountedFor>()->trip_count_ &&
         stride_ == other->As<CountedFor>()->stride_ &&
         body_->IsDefinitelyEqualTo(other->As<CountedFor>()->body_);
}

DynamicCountedFor::DynamicCountedFor(const SourceInfo& loc, Node* initial_value,
                                     Node* trip_count, Node* stride,
                                     absl::Span<Node* const> invariant_args,
                                     Function* body, std::string_view name,
                                     FunctionBase* function)
    : Node(Op::kDynamicCountedFor, initial_value->GetType(), loc, name,
           function),
      body_(body) {
  CHECK(IsOpClass<DynamicCountedFor>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `DynamicCountedFor`.";
  AddOperand(initial_value);
  AddOperand(trip_count);
  AddOperand(stride);
  AddOperands(invariant_args);
}

bool DynamicCountedFor::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return body_->IsDefinitelyEqualTo(other->As<DynamicCountedFor>()->body_);
}

ExtendOp::ExtendOp(const SourceInfo& loc, Node* arg, int64_t new_bit_count,
                   Op op, std::string_view name, FunctionBase* function)
    : Node(op, function->package()->GetBitsType(new_bit_count), loc, name,
           function),
      new_bit_count_(new_bit_count) {
  CHECK(IsOpClass<ExtendOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `ExtendOp`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> ExtendOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<ExtendOp>(
      loc(), new_operands[0], new_bit_count(), op(), GetNameView());
}

bool ExtendOp::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return new_bit_count_ == other->As<ExtendOp>()->new_bit_count_;
}

Invoke::Invoke(const SourceInfo& loc, absl::Span<Node* const> args,
               Function* to_apply, std::string_view name,
               FunctionBase* function)
    : Node(Op::kInvoke, to_apply->return_value()->GetType(), loc, name,
           function),
      to_apply_(to_apply) {
  CHECK(IsOpClass<Invoke>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Invoke`.";
  AddOperands(args);
}

absl::StatusOr<Node*> Invoke::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Invoke>(loc(), new_operands, to_apply(),
                                                GetNameView());
}

bool Invoke::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return to_apply_->IsDefinitelyEqualTo(other->As<Invoke>()->to_apply_);
}

Literal::Literal(const SourceInfo& loc, Value value, std::string_view name,
                 FunctionBase* function)
    : Node(Op::kLiteral, function->package()->GetTypeForValue(value), loc, name,
           function),
      value_(std::move(value)) {
  CHECK(IsOpClass<Literal>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Literal`.";
}

absl::StatusOr<Node*> Literal::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Literal>(loc(), value(), GetNameView());
}

bool Literal::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return value_ == other->As<Literal>()->value_;
}

Map::Map(const SourceInfo& loc, Node* arg, Function* to_apply,
         std::string_view name, FunctionBase* function)
    : Node(Op::kMap, GetMapType(arg, to_apply), loc, name, function),
      to_apply_(to_apply) {
  CHECK(IsOpClass<Map>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Map`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> Map::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Map>(loc(), new_operands[0], to_apply(),
                                             GetNameView());
}

bool Map::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return to_apply_->IsDefinitelyEqualTo(other->As<Map>()->to_apply_);
}

OneHot::OneHot(const SourceInfo& loc, Node* input, LsbOrMsb priority,
               std::string_view name, FunctionBase* function)
    : Node(Op::kOneHot,
           function->package()->GetBitsType(input->BitCountOrDie() + 1), loc,
           name, function),
      priority_(priority) {
  CHECK(IsOpClass<OneHot>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `OneHot`.";
  AddOperand(input);
}

absl::StatusOr<Node*> OneHot::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<OneHot>(loc(), new_operands[0],
                                                priority(), GetNameView());
}

bool OneHot::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return priority_ == other->As<OneHot>()->priority_;
}

OneHotSelect::OneHotSelect(const SourceInfo& loc, Node* selector,
                           absl::Span<Node* const> cases, std::string_view name,
                           FunctionBase* function)
    : Node(Op::kOneHotSel, cases[0]->GetType(), loc, name, function) {
  CHECK(IsOpClass<OneHotSelect>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `OneHotSelect`.";
  AddOperand(selector);
  AddOperands(cases);
}

PrioritySelect::PrioritySelect(const SourceInfo& loc, Node* selector,
                               absl::Span<Node* const> cases,
                               Node* default_value, std::string_view name,
                               FunctionBase* function)
    : Node(Op::kPrioritySel, default_value->GetType(), loc, name, function),
      cases_size_(cases.size()) {
  CHECK(IsOpClass<PrioritySelect>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `PrioritySelect`.";
  AddOperand(selector);
  AddOperands(cases);
  AddOperand(default_value);
}

bool PrioritySelect::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return cases_size_ == other->As<PrioritySelect>()->cases_size_;
}

Param::Param(const SourceInfo& loc, Type* type, std::string_view name,
             FunctionBase* function)
    : Node(Op::kParam, type, loc, name, function) {
  CHECK(IsOpClass<Param>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Param`.";
}

Next::Next(const SourceInfo& loc, Node* param, Node* value,
           std::optional<Node*> predicate, std::string_view name,
           FunctionBase* function)
    : Node(Op::kNext, function->package()->GetTupleType({}), loc, name,
           function),
      has_predicate_(predicate.has_value()) {
  CHECK(IsOpClass<Next>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Next`.";
  AddOperand(param);
  AddOperand(value);
  AddOptionalOperand(predicate);
}

bool Next::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return has_predicate_ == other->As<Next>()->has_predicate_;
}

Select::Select(const SourceInfo& loc, Node* selector,
               absl::Span<Node* const> cases,
               std::optional<Node*> default_value, std::string_view name,
               FunctionBase* function)
    : Node(Op::kSel, cases[0]->GetType(), loc, name, function),
      cases_size_(cases.size()),
      has_default_value_(default_value.has_value()) {
  CHECK(IsOpClass<Select>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Select`.";
  AddOperand(selector);
  AddOperands(cases);
  AddOptionalOperand(default_value);
}

bool Select::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return cases_size_ == other->As<Select>()->cases_size_ &&
         has_default_value_ == other->As<Select>()->has_default_value_;
}

Tuple::Tuple(const SourceInfo& loc, absl::Span<Node* const> elements,
             std::string_view name, FunctionBase* function)
    : Node(Op::kTuple, GetTupleType(function->package(), elements), loc, name,
           function) {
  CHECK(IsOpClass<Tuple>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Tuple`.";
  AddOperands(elements);
}

absl::StatusOr<Node*> Tuple::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Tuple>(loc(), new_operands,
                                               GetNameView());
}

TupleIndex::TupleIndex(const SourceInfo& loc, Node* arg, int64_t index,
                       std::string_view name, FunctionBase* function)
    : Node(Op::kTupleIndex, arg->GetType()->AsTupleOrDie()->element_type(index),
           loc, name, function),
      index_(index) {
  CHECK(IsOpClass<TupleIndex>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `TupleIndex`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> TupleIndex::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<TupleIndex>(loc(), new_operands[0],
                                                    index(), GetNameView());
}

bool TupleIndex::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return index_ == other->As<TupleIndex>()->index_;
}

UnOp::UnOp(const SourceInfo& loc, Node* arg, Op op, std::string_view name,
           FunctionBase* function)
    : Node(op, arg->GetType(), loc, name, function) {
  CHECK(IsOpClass<UnOp>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `UnOp`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> UnOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<UnOp>(loc(), new_operands[0], op(),
                                              GetNameView());
}

Decode::Decode(const SourceInfo& loc, Node* arg, int64_t width,
               std::string_view name, FunctionBase* function)
    : Node(Op::kDecode, function->package()->GetBitsType(width), loc, name,
           function),
      width_(width) {
  CHECK(IsOpClass<Decode>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Decode`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> Decode::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Decode>(loc(), new_operands[0], width(),
                                                GetNameView());
}

bool Decode::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return width_ == other->As<Decode>()->width_;
}

Encode::Encode(const SourceInfo& loc, Node* arg, std::string_view name,
               FunctionBase* function)
    : Node(Op::kEncode,
           function->package()->GetBitsType(CeilOfLog2(arg->BitCountOrDie())),
           loc, name, function) {
  CHECK(IsOpClass<Encode>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Encode`.";
  AddOperand(arg);
}

absl::StatusOr<Node*> Encode::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Encode>(loc(), new_operands[0],
                                                GetNameView());
}

InputPort::InputPort(const SourceInfo& loc, std::string_view name, Type* type,
                     FunctionBase* function)
    : Node(Op::kInputPort, type, loc, name, function) {
  CHECK(IsOpClass<InputPort>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `InputPort`.";
}

absl::StatusOr<Node*> InputPort::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<InputPort>(loc(), name(), GetType());
}

OutputPort::OutputPort(const SourceInfo& loc, Node* operand,
                       std::string_view name, FunctionBase* function)
    : Node(Op::kOutputPort, function->package()->GetTupleType({}), loc, name,
           function) {
  CHECK(IsOpClass<OutputPort>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `OutputPort`.";
  AddOperand(operand);
}

absl::StatusOr<Node*> OutputPort::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<OutputPort>(loc(), new_operands[0],
                                                    name());
}

RegisterRead::RegisterRead(const SourceInfo& loc, Register* reg,
                           std::string_view name, FunctionBase* function)
    : Node(Op::kRegisterRead, reg->type(), loc, name, function), reg_(reg) {
  CHECK(IsOpClass<RegisterRead>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `RegisterRead`.";
}

absl::StatusOr<Node*> RegisterRead::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<RegisterRead>(loc(), GetRegister(),
                                                      GetNameView());
}

bool RegisterRead::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return reg_ == other->As<RegisterRead>()->reg_;
}

RegisterWrite::RegisterWrite(const SourceInfo& loc, Node* data,
                             std::optional<Node*> load_enable,
                             std::optional<Node*> reset, Register* reg,
                             std::string_view name, FunctionBase* function)
    : Node(Op::kRegisterWrite, function->package()->GetTupleType({}), loc, name,
           function),
      reg_(reg),
      has_load_enable_(load_enable.has_value()),
      has_reset_(reset.has_value()) {
  CHECK(IsOpClass<RegisterWrite>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `RegisterWrite`.";
  AddOperand(data);
  AddOptionalOperand(load_enable);
  AddOptionalOperand(reset);
}

absl::StatusOr<Node*> RegisterWrite::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<RegisterWrite>(
      loc(), new_operands[0], new_operands[1], new_operands[2], GetRegister(),
      GetNameView());
}

bool RegisterWrite::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return reg_ == other->As<RegisterWrite>()->reg_ &&
         has_load_enable_ == other->As<RegisterWrite>()->has_load_enable_ &&
         has_reset_ == other->As<RegisterWrite>()->has_reset_;
}

InstantiationOutput::InstantiationOutput(const SourceInfo& loc,
                                         Instantiation* instantiation,
                                         std::string_view port_name,
                                         std::string_view name,
                                         FunctionBase* function)
    : Node(Op::kInstantiationOutput,
           instantiation->GetOutputPort(port_name).value().type, loc, name,
           function),
      instantiation_(instantiation),
      port_name_(port_name) {
  CHECK(IsOpClass<InstantiationOutput>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `InstantiationOutput`.";
}

absl::StatusOr<Node*> InstantiationOutput::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<InstantiationOutput>(
      loc(), instantiation(), port_name(), GetNameView());
}

bool InstantiationOutput::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return instantiation_ == other->As<InstantiationOutput>()->instantiation_ &&
         port_name_ == other->As<InstantiationOutput>()->port_name_;
}

InstantiationInput::InstantiationInput(const SourceInfo& loc, Node* data,
                                       Instantiation* instantiation,
                                       std::string_view port_name,
                                       std::string_view name,
                                       FunctionBase* function)
    : Node(Op::kInstantiationInput, function->package()->GetTupleType({}), loc,
           name, function),
      instantiation_(instantiation),
      port_name_(port_name) {
  CHECK(IsOpClass<InstantiationInput>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `InstantiationInput`.";
  AddOperand(data);
}

absl::StatusOr<Node*> InstantiationInput::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<InstantiationInput>(
      loc(), new_operands[0], instantiation(), port_name(), GetNameView());
}

bool InstantiationInput::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return instantiation_ == other->As<InstantiationInput>()->instantiation_ &&
         port_name_ == other->As<InstantiationInput>()->port_name_;
}

Gate::Gate(const SourceInfo& loc, Node* condition, Node* data,
           std::string_view name, FunctionBase* function)
    : Node(Op::kGate, data->GetType(), loc, name, function) {
  CHECK(IsOpClass<Gate>(op_))
      << "Op `" << op_ << "` is not a valid op for Node class `Gate`.";
  AddOperand(condition);
  AddOperand(data);
}

absl::StatusOr<Node*> Gate::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<Gate>(loc(), new_operands[0],
                                              new_operands[1], GetNameView());
}

SliceData Concat::GetOperandSliceData(int64_t operandno) const {
  CHECK_GE(operandno, 0);
  int64_t start = 0;
  for (int64_t i = operands().size() - 1; i > operandno; --i) {
    Node* operand = this->operand(i);
    start += operand->BitCountOrDie();
  }
  return SliceData{.start = start,
                   .width = this->operand(operandno)->BitCountOrDie()};
}

absl::StatusOr<Node*> Param::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  XLS_ASSIGN_OR_RETURN(
      Type * new_type,
      new_function->package()->MapTypeFromOtherPackage(GetType()));
  return new_function->MakeNodeWithName<Param>(loc(), new_type, GetNameView());
}

absl::StatusOr<Node*> Array::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  XLS_RET_CHECK_EQ(size(), new_operands.size());
  XLS_ASSIGN_OR_RETURN(
      Type * new_element_type,
      new_function->package()->MapTypeFromOtherPackage(element_type()));
  return new_function->MakeNodeWithName<Array>(loc(), new_operands,
                                               new_element_type, GetNameView());
}

absl::StatusOr<Node*> CountedFor::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<CountedFor>(
      loc(), new_operands[0], new_operands.subspan(1), trip_count(), stride(),
      body(), GetNameView());
}

absl::StatusOr<Node*> DynamicCountedFor::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<DynamicCountedFor>(
      loc(), new_operands[0], new_operands[1], new_operands[2],
      new_operands.subspan(3), body(), GetNameView());
}

absl::StatusOr<Node*> Select::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  std::optional<Node*> new_default_value =
      default_value().has_value() ? std::optional<Node*>(new_operands.back())
                                  : std::nullopt;
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Select>(
      loc(), new_operands[0], new_operands.subspan(1, cases_size_),
      new_default_value, GetNameView());
}

absl::StatusOr<Node*> OneHotSelect::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<OneHotSelect>(
      loc(), new_operands[0], new_operands.subspan(1), GetNameView());
}

absl::StatusOr<Node*> PrioritySelect::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<PrioritySelect>(
      loc(), new_operands[0], new_operands.subspan(1, cases_size_),
      new_operands.back(), GetNameView());
}

absl::StatusOr<Node*> ArrayIndex::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<ArrayIndex>(
      loc(), new_operands[0], new_operands.subspan(1), GetNameView());
}

absl::StatusOr<Node*> ArrayUpdate::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<ArrayUpdate>(
      loc(), new_operands[0], new_operands[1], new_operands.subspan(2),
      GetNameView());
}

absl::StatusOr<Node*> Trace::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(amfv): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Trace>(
      loc(), new_operands[0], new_operands[1], new_operands.subspan(2),
      format(), verbosity(), GetNameView());
}

absl::StatusOr<Node*> Receive::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Receive>(
      loc(), new_operands[0],
      new_operands.size() > 1 ? std::optional<Node*>(new_operands[1])
                              : std::nullopt,
      channel_name(), is_blocking(), GetNameView());
}

absl::StatusOr<Node*> Send::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Send>(
      loc(), new_operands[0], new_operands[1],
      new_operands.size() > 2 ? std::optional<Node*>(new_operands[2])
                              : std::nullopt,
      channel_name(), GetNameView());
}

absl::StatusOr<Node*> Next::CloneInNewFunction(
    absl::Span<Node* const> new_operands, FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Next>(
      loc(), new_operands[0], new_operands[1],
      new_operands.size() > 2 ? std::optional<Node*>(new_operands[2])
                              : std::nullopt,
      GetNameView());
}

bool Select::AllCases(const std::function<bool(Node*)>& p) const {
  for (Node* case_ : cases()) {
    if (!p(case_)) {
      return false;
    }
  }
  if (default_value().has_value()) {
    if (!p(default_value().value())) {
      return false;
    }
  }
  return true;
}

}  // namespace xls
