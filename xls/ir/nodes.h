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

#ifndef XLS_IR_NODES_H_
#define XLS_IR_NODES_H_

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

// TODO(meheff): Add comments to classes and methods.

namespace xls {

class Function;
class Instantiation;

struct SliceData {
  int64_t start;
  int64_t width;
};

class AfterAll final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kAfterAll};
  AfterAll(const SourceInfo& loc, absl::Span<Node* const> dependencies,
           std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class ArithOp final : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kUMul, Op::kSMul};
  static constexpr int64_t kLhsOperand = 0;
  static constexpr int64_t kRhsOperand = 1;

  ArithOp(const SourceInfo& loc, Node* lhs, Node* rhs, int64_t width, Op op,
          std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t width() const { return width_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t width_;
};

class Array final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kArray};
  Array(const SourceInfo& loc, absl::Span<Node* const> elements,
        Type* element_type, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Type* element_type() const { return element_type_; }

  int64_t size() const { return operand_count(); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Type* element_type_;
};

class ArrayConcat final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kArrayConcat};
  ArrayConcat(const SourceInfo& loc, absl::Span<Node* const> args,
              std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class ArrayIndex final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kArrayIndex};
  static constexpr int64_t kArgOperand = 0;
  static constexpr int64_t kIndexOperandStart = 1;

  ArrayIndex(const SourceInfo& loc, Node* arg, absl::Span<Node* const> indices,
             bool assumed_in_bounds, std::string_view name,
             FunctionBase* function);

  ArrayIndex(const SourceInfo& loc, Node* arg, absl::Span<Node* const> indices,
             std::string_view name, FunctionBase* function)
      : ArrayIndex(loc, arg, indices, /*assumed_in_bounds=*/false, name,
                   function) {}

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* array() const { return operand(0); }

  absl::Span<Node* const> indices() const { return operands().subspan(1); }

  // Are the values of all indices known to be in-bounds for the array.
  bool assumed_in_bounds() const { return assumed_in_bounds_; }

  // Mark/unmark this array-index as having all of its bounds statically known
  // to be good.
  void SetAssumedInBounds(bool value = true) { assumed_in_bounds_ = value; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  bool assumed_in_bounds_ = false;
};

class ArraySlice final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kArraySlice};
  static constexpr int64_t kArrayOperand = 0;
  static constexpr int64_t kStartOperand = 1;

  ArraySlice(const SourceInfo& loc, Node* array, Node* start, int64_t width,
             std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t width() const { return width_; }

  Node* array() const { return operand(0); }

  Node* start() const { return operand(1); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t width_;
};

class ArrayUpdate final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kArrayUpdate};
  static constexpr int64_t kArgOperand = 0;
  static constexpr int64_t kUpdateValueOperand = 1;
  static constexpr int64_t kIndexOperandStart = 2;

  ArrayUpdate(const SourceInfo& loc, Node* arg, Node* update_value,
              absl::Span<Node* const> indices, bool assumed_in_bounds,
              std::string_view name, FunctionBase* function);
  ArrayUpdate(const SourceInfo& loc, Node* arg, Node* update_value,
              absl::Span<Node* const> indices, std::string_view name,
              FunctionBase* function)
      : ArrayUpdate(loc, arg, update_value, indices,
                    /*assumed_in_bounds=*/false, name, function) {}

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* array_to_update() const { return operand(0); }

  absl::Span<Node* const> indices() const { return operands().subspan(2); }

  Node* update_value() const { return operand(1); }

  // Are the values of all indices known to be in-bounds for the array.
  bool assumed_in_bounds() const { return assumed_in_bounds_; }

  // Mark/unmark this array-index as having all of its bounds statically known
  // to be good.
  void SetAssumedInBounds(bool value = true) { assumed_in_bounds_ = value; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  bool assumed_in_bounds_;
};

class Assert final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kAssert};
  static constexpr int64_t kTokenOperand = 0;
  static constexpr int64_t kConditionOperand = 1;

  Assert(const SourceInfo& loc, Node* token, Node* condition,
         std::string_view message, std::optional<std::string> label,
         std::optional<std::string> original_label, std::string_view name,
         FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  const std::string& message() const { return message_; }

  std::optional<std::string> label() const { return label_; }

  std::optional<std::string> original_label() const { return original_label_; }

  Node* token() const { return operand(0); }

  Node* condition() const { return operand(1); }

  void set_label(std::string new_label) { label_ = std::move(new_label); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  std::string message_;
  std::optional<std::string> label_;
  std::optional<std::string> original_label_;
};

class BinOp final : public Node {
 public:
  static constexpr std::array<Op, 9> kOps = {Op::kAdd,  Op::kSDiv, Op::kSMod,
                                             Op::kShll, Op::kShrl, Op::kShra,
                                             Op::kSub,  Op::kUDiv, Op::kUMod};
  static constexpr int64_t kLhsOperand = 0;
  static constexpr int64_t kRhsOperand = 1;

  BinOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op,
        std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class BitSlice final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kBitSlice};
  static constexpr int64_t kArgOperand = 0;

  BitSlice(const SourceInfo& loc, Node* arg, int64_t start, int64_t width,
           std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t start() const { return start_; }

  int64_t width() const { return width_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t start_;
  int64_t width_;
};

class BitSliceUpdate final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kBitSliceUpdate};
  static constexpr int64_t kArgOperand = 0;
  static constexpr int64_t kStartOperand = 1;
  static constexpr int64_t kValueOperand = 2;

  BitSliceUpdate(const SourceInfo& loc, Node* arg, Node* start, Node* value,
                 std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* to_update() const { return operand(0); }

  Node* start() const { return operand(1); }

  Node* update_value() const { return operand(2); }
};

class BitwiseReductionOp final : public Node {
 public:
  static constexpr std::array<Op, 3> kOps = {Op::kAndReduce, Op::kOrReduce,
                                             Op::kXorReduce};
  static constexpr int64_t kOperandOperand = 0;

  BitwiseReductionOp(const SourceInfo& loc, Node* operand, Op op,
                     std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class CompareOp final : public Node {
 public:
  static constexpr std::array<Op, 10> kOps = {
      Op::kEq,  Op::kNe,  Op::kSLe, Op::kSGe, Op::kSLt,
      Op::kSGt, Op::kULe, Op::kUGe, Op::kULt, Op::kUGt};
  static constexpr int64_t kLhsOperand = 0;
  static constexpr int64_t kRhsOperand = 1;

  CompareOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op,
            std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class Concat final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kConcat};
  Concat(const SourceInfo& loc, absl::Span<Node* const> args,
         std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  SliceData GetOperandSliceData(int64_t operandno) const;
};

class CountedFor final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kCountedFor};
  static constexpr int64_t kInitialValueOperand = 0;

  CountedFor(const SourceInfo& loc, Node* initial_value,
             absl::Span<Node* const> invariant_args, int64_t trip_count,
             int64_t stride, Function* body, std::string_view name,
             FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t trip_count() const { return trip_count_; }

  int64_t stride() const { return stride_; }

  Function* body() const { return body_; }

  Node* initial_value() const { return operand(0); }

  absl::Span<Node* const> invariant_args() const {
    return operands().subspan(1);
  }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t trip_count_;
  int64_t stride_;
  Function* body_;
};

class Cover final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kCover};
  static constexpr int64_t kConditionOperand = 0;

  Cover(const SourceInfo& loc, Node* condition, std::string_view label,
        std::optional<std::string> original_label, std::string_view name,
        FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  const std::string& label() const { return label_; }

  std::optional<std::string> original_label() const { return original_label_; }

  Node* condition() const { return operand(0); }

  void set_label(std::string new_label) { label_ = std::move(new_label); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  std::string label_;
  std::optional<std::string> original_label_;
};

class Decode final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kDecode};
  static constexpr int64_t kArgOperand = 0;

  Decode(const SourceInfo& loc, Node* arg, int64_t width, std::string_view name,
         FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t width() const { return width_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t width_;
};

class DynamicBitSlice final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kDynamicBitSlice};
  static constexpr int64_t kArgOperand = 0;
  static constexpr int64_t kStartOperand = 1;

  DynamicBitSlice(const SourceInfo& loc, Node* arg, Node* start, int64_t width,
                  std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t width() const { return width_; }

  Node* to_slice() const { return operand(0); }

  Node* start() const { return operand(1); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t width_;
};

class DynamicCountedFor final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kDynamicCountedFor};
  static constexpr int64_t kInitialValueOperand = 0;
  static constexpr int64_t kTripCountOperand = 1;
  static constexpr int64_t kStrideOperand = 2;

  DynamicCountedFor(const SourceInfo& loc, Node* initial_value,
                    Node* trip_count, Node* stride,
                    absl::Span<Node* const> invariant_args, Function* body,
                    std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Function* body() const { return body_; }

  Node* initial_value() const { return operand(0); }

  Node* trip_count() const { return operand(1); }

  Node* stride() const { return operand(2); }

  absl::Span<Node* const> invariant_args() const {
    return operands().subspan(3);
  }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Function* body_;
};

class Encode final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kEncode};
  static constexpr int64_t kArgOperand = 0;

  Encode(const SourceInfo& loc, Node* arg, std::string_view name,
         FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class ExtendOp final : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kZeroExt, Op::kSignExt};
  static constexpr int64_t kArgOperand = 0;

  ExtendOp(const SourceInfo& loc, Node* arg, int64_t new_bit_count, Op op,
           std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t new_bit_count() const { return new_bit_count_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t new_bit_count_;
};

class Gate final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kGate};
  static constexpr int64_t kConditionOperand = 0;
  static constexpr int64_t kDataOperand = 1;

  Gate(const SourceInfo& loc, Node* condition, Node* data,
       std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* condition() const { return operand(0); }

  Node* data() const { return operand(1); }
};

enum PortDirection : uint8_t { kInput, kOutput };

class PortNode : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kInputPort, Op::kOutputPort};
  PortNode(const SourceInfo& loc, Op op, Type* type, std::string_view name,
           std::optional<std::string> system_verilog_type,
           FunctionBase* function)
      : Node(op, type, loc, name, function),
        system_verilog_type_(system_verilog_type) {}

  virtual Type* port_type() const = 0;
  PortDirection direction() const {
    return op() == Op::kInputPort ? PortDirection::kInput
                                  : PortDirection::kOutput;
  }
  const std::optional<std::string>& system_verilog_type() const {
    return system_verilog_type_;
  }

  void set_system_verilog_type(std::optional<std::string> value) {
    system_verilog_type_ = value;
  }

 private:
  std::optional<std::string> system_verilog_type_;
};

class InputPort final : public PortNode {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kInputPort};
  InputPort(const SourceInfo& loc, std::string_view name, Type* type,
            FunctionBase* function)
      : PortNode(loc, Op::kInputPort, type, name,
                 /*system_verilog_type=*/std::nullopt, function) {}
  InputPort(const SourceInfo& loc, std::string_view name, Type* type,
            std::optional<std::string> system_verilog_type,
            FunctionBase* function)
      : PortNode(loc, Op::kInputPort, type, name, system_verilog_type,
                 function) {}

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  std::string_view name() const { return GetNameView(); }

  Type* port_type() const override { return GetType(); }
};

class OutputPort final : public PortNode {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kOutputPort};
  static constexpr int64_t kOperandOperand = 0;

  OutputPort(const SourceInfo& loc, Node* operand, std::string_view name,
             FunctionBase* function);
  OutputPort(const SourceInfo& loc, Node* operand, std::string_view name,
             std::optional<std::string> system_verilog_type,
             FunctionBase* function);

  // Get the value that this port sends.
  Node* output_source() const { return operand(kOperandOperand); }

  Type* port_type() const override { return output_source()->GetType(); }

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  std::string_view name() const { return GetNameView(); }
};

class InstantiationConnection : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kInstantiationInput,
                                             Op::kInstantiationOutput};
  InstantiationConnection(Op op, Type* type, const SourceInfo& loc,
                          Instantiation* instantiation,
                          std::string_view port_name, std::string_view name,
                          FunctionBase* function)
      : Node(op, type, loc, name, function),
        instantiation_(instantiation),
        port_name_(port_name) {}

  Instantiation* instantiation() const { return instantiation_; }

  const std::string& port_name() const { return port_name_; }

 private:
  Instantiation* instantiation_;
  std::string port_name_;
};

class InstantiationInput final : public InstantiationConnection {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kInstantiationInput};
  static constexpr int64_t kDataOperand = 0;

  InstantiationInput(const SourceInfo& loc, Node* data,
                     Instantiation* instantiation, std::string_view port_name,
                     std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

  Node* data() const { return operand(0); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Instantiation* instantiation_;
  std::string port_name_;
};

class InstantiationOutput final : public InstantiationConnection {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kInstantiationOutput};
  InstantiationOutput(const SourceInfo& loc, Instantiation* instantiation,
                      std::string_view port_name, std::string_view name,
                      FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Instantiation* instantiation_;
  std::string port_name_;
};

class Invoke final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kInvoke};
  Invoke(const SourceInfo& loc, absl::Span<Node* const> args,
         Function* to_apply, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Function* to_apply() const { return to_apply_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Function* to_apply_;
};

class Literal final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kLiteral};
  Literal(const SourceInfo& loc, Value value, std::string_view name,
          FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  const Value& value() const { return value_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Value value_;
};

class Map final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kMap};
  static constexpr int64_t kArgOperand = 0;

  Map(const SourceInfo& loc, Node* arg, Function* to_apply,
      std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Function* to_apply() const { return to_apply_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Function* to_apply_;
};

class MinDelay final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kMinDelay};
  static constexpr int64_t kTokenOperand = 0;

  MinDelay(const SourceInfo& loc, Node* token, int64_t delay,
           std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t delay() const { return delay_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t delay_;
};

class NaryOp final : public Node {
 public:
  static constexpr std::array<Op, 5> kOps = {Op::kAnd, Op::kNand, Op::kNor,
                                             Op::kOr, Op::kXor};
  NaryOp(const SourceInfo& loc, absl::Span<Node* const> args, Op op,
         std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

class StateRead final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kStateRead};
  StateRead(const SourceInfo& loc, StateElement* state_element,
            std::optional<Node*> predicate, std::string_view name,
            FunctionBase* function);

  StateElement* state_element() const { return state_element_; }

  std::optional<Node*> predicate() const {
    return has_predicate_ ? std::make_optional(operand(kPredicateOperand))
                          : std::nullopt;
  }

  absl::StatusOr<int64_t> predicate_operand_number() const {
    if (!has_predicate_) {
      return absl::InternalError("predicate is not present");
    }
    return kPredicateOperand;
  }

  absl::Status SetPredicate(Node* predicate) {
    XLS_RET_CHECK_NE(predicate, nullptr) << absl::StreamFormat(
        "Cannot set predicate of node `%s` to nullptr; use RemovePredicate() "
        "if you mean to remove it",
        GetName());

    if (has_predicate_) {
      return ReplaceOperandNumber(kPredicateOperand, predicate);
    }

    AddOptionalOperand(predicate);
    has_predicate_ = true;
    return absl::OkStatus();
  }

  absl::Status RemovePredicate() {
    XLS_RET_CHECK(has_predicate_) << absl::StreamFormat(
        "Cannot remove predicate of node `%s` as it has none", GetName());
    XLS_RETURN_IF_ERROR(RemoveOptionalOperand(kPredicateOperand));
    has_predicate_ = false;
    return absl::OkStatus();
  }

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  static constexpr int64_t kPredicateOperand = 0;

  StateElement* state_element_;
  bool has_predicate_;
};

class Next final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kNext};
  static constexpr int64_t kStateReadOperand = 0;
  static constexpr int64_t kValueOperand = 1;

  Next(const SourceInfo& loc, Node* state_read, Node* value,
       std::optional<Node*> predicate, std::string_view name,
       FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* state_read() const { return operand(0); }

  Node* value() const { return operand(1); }

  std::optional<Node*> predicate() const {
    return has_predicate_ ? std::make_optional(operand(kPredicateOperand))
                          : std::nullopt;
  }

  absl::StatusOr<int64_t> predicate_operand_number() const {
    if (!has_predicate_) {
      return absl::InternalError("predicate is not present");
    }
    return kPredicateOperand;
  }

  absl::Status SetPredicate(Node* predicate) {
    XLS_RET_CHECK_NE(predicate, nullptr) << absl::StreamFormat(
        "Cannot set predicate of node `%s` to nullptr; use RemovePredicate() "
        "if you mean to remove it",
        GetName());

    if (has_predicate_) {
      return ReplaceOperandNumber(kPredicateOperand, predicate);
    }

    AddOptionalOperand(predicate);
    has_predicate_ = true;
    return absl::OkStatus();
  }

  absl::Status RemovePredicate() {
    XLS_RET_CHECK(has_predicate_) << absl::StreamFormat(
        "Cannot remove predicate of node `%s` as it has none", GetName());
    XLS_RETURN_IF_ERROR(RemoveOptionalOperand(kPredicateOperand));
    has_predicate_ = false;
    return absl::OkStatus();
  }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  static constexpr int64_t kPredicateOperand = 2;

  bool has_predicate_;
};

class OneHot final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kOneHot};
  static constexpr int64_t kInputOperand = 0;

  OneHot(const SourceInfo& loc, Node* input, LsbOrMsb priority,
         std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  LsbOrMsb priority() const { return priority_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  LsbOrMsb priority_;
};

class OneHotSelect final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kOneHotSel};
  static constexpr int64_t kSelectorOperand = 0;

  OneHotSelect(const SourceInfo& loc, Node* selector,
               absl::Span<Node* const> cases, std::string_view name,
               FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* selector() const { return operand(0); }

  absl::Span<Node* const> cases() const { return operands().subspan(1); }

  Node* get_case(int64_t case_no) const { return cases().at(case_no); }
};

class Param final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kParam};
  Param(const SourceInfo& loc, Type* type, std::string_view name,
        FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  std::string_view name() const { return GetNameView(); }
};

class PartialProductOp final : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kSMulp, Op::kUMulp};
  static constexpr int64_t kLhsOperand = 0;
  static constexpr int64_t kRhsOperand = 1;

  PartialProductOp(const SourceInfo& loc, Node* lhs, Node* rhs, int64_t width,
                   Op op, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t width() const { return width_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t width_;
};

class PrioritySelect final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kPrioritySel};
  static constexpr int64_t kSelectorOperand = 0;

  PrioritySelect(const SourceInfo& loc, Node* selector,
                 absl::Span<Node* const> cases, Node* default_value,
                 std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* selector() const { return operand(0); }

  absl::Span<Node* const> cases() const {
    return operands().subspan(1, cases_size_);
  }

  Node* get_case(int64_t case_no) const { return cases().at(case_no); }

  Node* default_value() const { return operand(1 + cases_size_); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t cases_size_;
};

// Represents a new-style channel node. NOTE: this is still a work in progress
// and is not ready for use yet.
class NewChannel : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kNewChannel};

  NewChannel(const SourceInfo& loc, Type* type, const Channel* channel,
             FunctionBase* function)
      : Node(Op::kNewChannel, type, loc, channel->name(), function),
        channel_(channel) {}

  const Channel* channel() const { return channel_; }

  std::string_view channel_name() const { return channel_->name(); }

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

 private:
  const Channel* channel_;
};

// Represents a new-style channel "receiver" node. NOTE: this is still a work in
// progress and is not ready for use yet.
class RecvChannelEnd : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kRecvChannelEnd};

  RecvChannelEnd(const SourceInfo& loc, Type* type,
                 const ReceiveChannelInterface* channel_interface,
                 FunctionBase* function)
      : Node(Op::kRecvChannelEnd, type, loc, channel_interface->name(),
             function),
        channel_interface_(channel_interface) {}

  std::string_view channel_name() const { return channel_interface_->name(); }
  const ReceiveChannelInterface* channel_interface() const {
    return channel_interface_;
  }

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

 private:
  const ReceiveChannelInterface* channel_interface_;
};

// Represents a new-style channel "send" node. NOTE: this is still a work in
// progress and is not ready for use yet.
class SendChannelEnd : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kSendChannelEnd};

  SendChannelEnd(const SourceInfo& loc, Type* type,
                 const SendChannelInterface* channel_interface,
                 FunctionBase* function)
      : Node(Op::kSendChannelEnd, type, loc, channel_interface->name(),
             function),
        channel_interface_(channel_interface) {}

  std::string_view channel_name() const { return channel_interface_->name(); }
  const SendChannelInterface* channel_interface() const {
    return channel_interface_;
  }

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

 private:
  const SendChannelInterface* channel_interface_;
};

// Base class for nodes which communicate over channels.
class ChannelNode : public Node {
 public:
  static constexpr std::array<Op, 2> kOps = {Op::kReceive, Op::kSend};

  ChannelNode(const SourceInfo& loc, Op op, Type* type,
              std::string_view channel_name, ChannelDirection direction,
              bool has_predicate, std::string_view name, FunctionBase* function)
      : Node(op, type, loc, name, function),
        channel_name_(channel_name),
        direction_(direction),
        has_predicate_(has_predicate) {}

  const std::string& channel_name() const { return channel_name_; }

  // Return either the channel (old-style proc) or the channel reference
  // (new-style proc) for this channel op node.
  absl::StatusOr<ChannelRef> GetChannelRef() const;

  // Return the channel interface for this channel op node.
  // Returns an error if the function is not a new-style proc.
  absl::StatusOr<ChannelInterface*> GetChannelInterface() const;

  // Returns the direction this node communicates on the channel.
  ChannelDirection direction() const { return direction_; }

  // Returns the type of the data payload communicated on the channel.
  Type* GetPayloadType() const;

  // Replaces the channel with the channel with the given name. Returns an
  // error if no such channel/channel-ref exists.
  absl::Status ReplaceChannel(std::string_view new_channel_name);

  Node* token() const { return operand(0); }

  // Replaces the token operand with the given node. Returns an error if
  // `new_token` is not token-typed.
  absl::Status ReplaceToken(Node* new_token) {
    XLS_RET_CHECK(new_token->GetType()->IsToken()) << absl::StreamFormat(
        "Expected new token value to be token typed, is: %s",
        new_token->GetType()->ToString());
    XLS_RETURN_IF_ERROR(ReplaceOperandNumber(0, new_token));
    return absl::OkStatus();
  }

  std::optional<Node*> predicate() const {
    // The predicate is the last operand.
    return has_predicate_ ? std::optional<Node*>(operands().back())
                          : std::nullopt;
  }

  absl::Status SetPredicate(std::optional<Node*> predicate) {
    if (predicate) {
      Node* new_predicate = *predicate;
      XLS_RET_CHECK(new_predicate->GetType()->IsBits() &&
                    new_predicate->BitCountOrDie() == 1)
          << absl::StreamFormat(
                 "Expected predicate to be single-bit bits type, is: %s",
                 new_predicate->GetType()->ToString());
      if (has_predicate_) {
        return ReplaceOperandNumber(operand_count() - 1, new_predicate);
      }
      has_predicate_ = true;
      AddOperand(new_predicate);
      return absl::OkStatus();
    }
    if (!has_predicate_) {
      // Already don't have a predicate?
      return absl::OkStatus();
    }
    has_predicate_ = false;
    return RemoveOptionalOperand(operand_count() - 1);
  }

  absl::Status ReplacePredicate(Node* new_predicate) {
    XLS_RET_CHECK(has_predicate_) << absl::StreamFormat(
        "Cannot replace predicate of node `%s` as it does not currently have a "
        "predicate",
        GetName());
    XLS_RET_CHECK(new_predicate->GetType()->IsBits() &&
                  new_predicate->BitCountOrDie() == 1)
        << absl::StreamFormat(
               "Expected predicate to be single-bit bits type, is: %s",
               new_predicate->GetType()->ToString());
    XLS_RETURN_IF_ERROR(
        ReplaceOperandNumber(operand_count() - 1, new_predicate));
    return absl::OkStatus();
  }

 private:
  std::string channel_name_;
  ChannelDirection direction_;
  bool has_predicate_;
};

class Receive final : public ChannelNode {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kReceive};
  static constexpr int64_t kTokenOperand = 0;

  Receive(const SourceInfo& loc, Node* token, std::optional<Node*> predicate,
          std::string_view channel_name, bool is_blocking,
          std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

  bool is_blocking() const { return is_blocking_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

  absl::StatusOr<ReceiveChannelRef> GetReceiveChannelRef() const;

 private:
  bool is_blocking_;
};

class Send final : public ChannelNode {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kSend};
  static constexpr int64_t kTokenOperand = 0;
  static constexpr int64_t kDataOperand = 1;

  Send(const SourceInfo& loc, Node* token, Node* data,
       std::optional<Node*> predicate, std::string_view channel_name,
       std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;

  Node* data() const { return operand(1); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

  // Get the send channel reference
  absl::StatusOr<SendChannelRef> GetSendChannelRef() const;
};

class RegisterRead final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kRegisterRead};
  RegisterRead(const SourceInfo& loc, Register* reg, std::string_view name,
               FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Register* GetRegister() const { return reg_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Register* reg_;
};

class RegisterWrite final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kRegisterWrite};
  static constexpr int64_t kDataOperand = 0;

  RegisterWrite(const SourceInfo& loc, Node* data,
                std::optional<Node*> load_enable, std::optional<Node*> reset,
                Register* reg, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* data() const { return operand(0); }

  std::optional<Node*> load_enable() const {
    return has_load_enable_
               ? std::make_optional(operand(*load_enable_operand_number()))
               : std::nullopt;
  }

  std::optional<Node*> reset() const {
    return has_reset_ ? std::make_optional(operand(*reset_operand_number()))
                      : std::nullopt;
  }

  Register* GetRegister() const { return reg_; }

  absl::Status ReplaceExistingLoadEnable(Node* new_operand) {
    return has_load_enable_
               ? ReplaceOperandNumber(*load_enable_operand_number(),
                                      new_operand)
               : absl::InternalError(
                     "Unable to replace load enable on RegisterWrite -- "
                     "register does not have an existing load enable operand.");
  }

  absl::Status SetReset(std::optional<Node*> reset) {
    // Clear existing reset.
    if (has_reset_) {
      XLS_RETURN_IF_ERROR(RemoveOptionalOperand(*reset_operand_number()));
      has_reset_ = false;
    }
    if (!reset.has_value()) {
      return absl::OkStatus();
    }
    has_reset_ = true;
    AddOperand(reset.value());
    return absl::OkStatus();
  }

  absl::StatusOr<int64_t> load_enable_operand_number() const {
    if (!has_load_enable_) {
      return absl::InternalError("load_enable is not present");
    }
    return 1;
  }

  absl::StatusOr<int64_t> reset_operand_number() const {
    if (!has_reset_) {
      return absl::InternalError("reset is not present");
    }
    return has_load_enable_ ? 2 : 1;
  }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  Register* reg_;
  bool has_load_enable_;
  bool has_reset_;
};

class Select final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kSel};
  static constexpr int64_t kSelectorOperand = 0;

  Select(const SourceInfo& loc, Node* selector, absl::Span<Node* const> cases,
         std::optional<Node*> default_value, std::string_view name,
         FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  Node* selector() const { return operand(0); }

  absl::Span<Node* const> cases() const {
    return operands().subspan(1, cases_size_);
  }

  Node* get_case(int64_t case_no) const { return cases().at(case_no); }

  std::optional<Node*> default_value() const {
    return has_default_value_ ? std::optional<Node*>(operands().back())
                              : std::nullopt;
  }

  bool AllCases(const std::function<bool(Node*)>& p) const;
  Node* any_case() const {
    return !cases().empty()              ? cases().front()
           : default_value().has_value() ? default_value().value()
                                         : nullptr;
  }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t cases_size_;
  bool has_default_value_;
};

class Trace final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kTrace};
  static constexpr int64_t kTokenOperand = 0;
  static constexpr int64_t kConditionOperand = 1;

  Trace(const SourceInfo& loc, Node* token, Node* condition,
        absl::Span<Node* const> args, absl::Span<FormatStep const> format,
        int64_t verbosity, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  absl::Span<FormatStep const> format() const { return format_; }

  int64_t verbosity() const { return verbosity_; }

  Node* token() const { return operand(0); }

  Node* condition() const { return operand(1); }

  absl::Span<Node* const> args() const { return operands().subspan(2); }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  std::vector<FormatStep> format_;
  int64_t verbosity_;
};

class Tuple final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kTuple};
  Tuple(const SourceInfo& loc, absl::Span<Node* const> elements,
        std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t size() const { return operand_count(); }
};

class TupleIndex final : public Node {
 public:
  static constexpr std::array<Op, 1> kOps = {Op::kTupleIndex};
  static constexpr int64_t kArgOperand = 0;

  TupleIndex(const SourceInfo& loc, Node* arg, int64_t index,
             std::string_view name, FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
  int64_t index() const { return index_; }

  bool IsDefinitelyEqualTo(const Node* other) const final;

 private:
  int64_t index_;
};

class UnOp final : public Node {
 public:
  static constexpr std::array<Op, 4> kOps = {Op::kIdentity, Op::kNeg, Op::kNot,
                                             Op::kReverse};
  static constexpr int64_t kArgOperand = 0;

  UnOp(const SourceInfo& loc, Node* arg, Op op, std::string_view name,
       FunctionBase* function);

  absl::StatusOr<Node*> CloneInNewFunction(
      absl::Span<Node* const> new_operands,
      FunctionBase* new_function) const final;
};

}  // namespace xls

#endif  // XLS_IR_NODES_H_
