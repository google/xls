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

#ifndef XLS_INTERPRETER_IR_INTERPRETER_H_
#define XLS_INTERPRETER_IR_INTERPRETER_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Evaluates the given node using the given operand values and returns the
// result.
absl::StatusOr<Value> InterpretNode(Node* node,
                                    absl::Span<const Value> operand_values);

// A visitor for traversing and evaluating a Function.
class IrInterpreter : public DfsVisitor {
 public:
  IrInterpreter() = default;

  // Sets the evaluated value for 'node' to the given Value. 'value' must be
  // passed in by value (ha!) because a use case is passing in a previously
  // evaluated value and inserting a into flat_hash_map (done below) invalidates
  // all references to Values in the map.
  absl::Status SetValueResult(Node* node, Value result);

  // Returns the previously evaluated value of 'node' as a Value.
  const Value& ResolveAsValue(Node* node) const {
    return node_values_.at(node);
  }

  // Returns true if a value has been set for the result of the given node.
  bool HasResult(Node* node) const { return node_values_.contains(node); }

  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayConcat(ArrayConcat* concat) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArraySlice(ArraySlice* slice) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleAssert(Assert* assert_op) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleCover(Cover* cover) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleInputPort(InputPort* input_port) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* and_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* or_reduce) override;
  absl::Status HandleOutputPort(OutputPort* output_port) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleReceive(Receive* receive) override;
  absl::Status HandleRegisterRead(RegisterRead* reg_read) override;
  absl::Status HandleRegisterWrite(RegisterWrite* reg_write) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleShll(BinOp* shll) override;
  absl::Status HandleShra(BinOp* shra) override;
  absl::Status HandleShrl(BinOp* shrl) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSub(BinOp* sub) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* div) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMod(BinOp* mod) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* xor_reduce) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

 protected:
  // Returns an error if the given node or any of its operands are not Bits
  // types.
  absl::Status VerifyAllBitsTypes(Node* node);

  // Returns the previously evaluated value of 'node' as a bits type. CHECK
  // fails if it is not bits.
  const Bits& ResolveAsBits(Node* node);

  // Returns the previously evaluated value of 'node' as a boolean value. CHECK
  // fails if it is not bits value of width 1.
  bool ResolveAsBool(Node* node);

  // Returns the evaluated values of the given nodes as a vector of bits
  // values. Each node should be bits-typed.
  std::vector<Bits> ResolveAsBitsVector(absl::Span<Node* const> nodes);

  // Returns the previously evaluated value of 'node' as a uint64_t. If the
  // value is greater than 'upper_limit' then 'upper_limit' is returned.
  uint64_t ResolveAsBoundedUint64(Node* node, uint64_t upper_limit);

  // Sets the evaluated value for 'node' to the given uint64_t value. Returns an
  // error if 'node' is not a bits type or the result does not fit in the type.
  absl::Status SetUint64Result(Node* node, uint64_t result);

  // Sets the evaluated value for 'node' to the given bits value. Returns an
  // error if 'node' is not a bits type.
  absl::Status SetBitsResult(Node* node, const Bits& result);

  // Performs a logical OR of the given inputs. If 'inputs' is a not a Bits type
  // (ie, tuple or array) the element a recursively traversed and the Bits-typed
  // leaves are OR-ed.
  absl::StatusOr<Value> DeepOr(Type* input_type,
                               absl::Span<const Value* const> inputs);

  // The arguments to the Function being evaluated indexed by parameter name.
  std::vector<Value> args_;

  // The evaluated values for the nodes in the Function.
  absl::flat_hash_map<Node*, Value> node_values_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_IR_INTERPRETER_H_
