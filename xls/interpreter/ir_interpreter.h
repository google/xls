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
#include "xls/interpreter/ir_interpreter_stats.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"

namespace xls {

// A visitor for traversing and evaluating a Function.
class IrInterpreter : public DfsVisitor {
 public:
  IrInterpreter(absl::Span<const Value> args, InterpreterStats* stats)
      : stats_(stats), args_(args.begin(), args.end()) {}

  // Runs the interpreter on the given function. 'args' are the argument values
  // indexed by parameter name.
  static absl::StatusOr<Value> Run(Function* function,
                                   absl::Span<const Value> args,
                                   InterpreterStats* stats = nullptr);

  // Runs the interpreter on the function where the arguments are given by name.
  static absl::StatusOr<Value> RunKwargs(
      Function* function, const absl::flat_hash_map<std::string, Value>& args,
      InterpreterStats* stats = nullptr);

  // Evaluates the given node and returns the Value. Prerequisite: node must
  // have only literal operands.
  static absl::StatusOr<Value> EvaluateNodeWithLiteralOperands(Node* node);

  // Evaluates the given node using the given operand values and returns the
  // result.
  static absl::StatusOr<Value> EvaluateNode(
      Node* node, absl::Span<const Value* const> operand_values);

  absl::Status HandleAdd(BinOp* add) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* and_reduce) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleMultiArrayIndex(MultiArrayIndex* index) override;
  absl::Status HandleMultiArrayUpdate(MultiArrayUpdate* update) override;
  absl::Status HandleArrayConcat(ArrayConcat* concat) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleReceive(Receive* receive) override;
  absl::Status HandleReceiveIf(ReceiveIf* receive_if) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleSendIf(SendIf* send_if) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleIdentity(UnOp* identity) override;
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
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* div) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSMod(BinOp* mod) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSel(Select* sel) override;
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

  // Returns the previously evaluated value of 'node' as a Value.
  const Value& ResolveAsValue(Node* node) { return node_values_.at(node); }

 protected:
  // Returns an error if the given node or any of its operands are not Bits
  // types.
  absl::Status VerifyAllBitsTypes(Node* node);

  // Returns the previously evaluated value of 'node' as a bits type. CHECK
  // fails if it is not bits.
  const Bits& ResolveAsBits(Node* node);

  // Returns the evaluated values of the given nodes as a vector of bits
  // values. Each node should be bits-typed.
  std::vector<Bits> ResolveAsBitsVector(absl::Span<Node* const> nodes);

  // Returns the previously evaluated value of 'node' as a uint64. If the value
  // is greater than 'upper_limit' then 'upper_limit' is returned.
  uint64 ResolveAsBoundedUint64(Node* node, uint64 upper_limit);

  // Sets the evaluated value for 'node' to the given uint64 value. Returns an
  // error if 'node' is not a bits type or the result does not fit in the type.
  absl::Status SetUint64Result(Node* node, uint64 result);

  // Sets the evaluated value for 'node' to the given bits value. Returns an
  // error if 'node' is not a bits type.
  absl::Status SetBitsResult(Node* node, const Bits& result);

  // Sets the evaluated value for 'node' to the given Value. 'value' must be
  // passed in by value (ha!) because a use case is passing in a previously
  // evaluated value and inserting a into flat_hash_map (done below) invalidates
  // all references to Values in the map.
  absl::Status SetValueResult(Node* node, Value result);

  // Performs a logical OR of the given inputs. If 'inputs' is a not a Bits type
  // (ie, tuple or array) the element a recursively traversed and the Bits-typed
  // leaves are OR-ed.
  absl::StatusOr<Value> DeepOr(Type* input_type,
                               absl::Span<const Value* const> inputs);

  // Statistics on interpreter execution. May be nullptr.
  InterpreterStats* stats_;

  // The arguments to the Function being evaluated indexed by parameter name.
  std::vector<Value> args_;

  // The evaluated values for the nodes in the Function.
  absl::flat_hash_map<Node*, Value> node_values_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_IR_INTERPRETER_H_
