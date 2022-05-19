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
#ifndef XLS_JIT_FUNCTION_BUILDER_VISITOR_H_
#define XLS_JIT_FUNCTION_BUILDER_VISITOR_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

// Visitor to construct LLVM IR for each encountered XLS IR node. Based on
// DfsVisitorWithDefault to highlight any unhandled IR nodes.
class FunctionBuilderVisitor : public DfsVisitorWithDefault {
 public:
  // Args:
  //   xls_fn: the XLS function being translated.
  //   llvm_fn: the [empty] LLVM function being populated.
  //   is_top: true if this is the top-level function being translated,
  //     false if this is a function invocation from already inside "LLVM
  //     space".
  static absl::Status Visit(llvm::Module* module, llvm::Function* llvm_fn,
                            FunctionBase* xls_fn,
                            LlvmTypeConverter* type_converter, bool is_top);

  absl::Status DefaultHandler(Node* node) override {
    return absl::UnimplementedError(
        absl::StrCat("Unhandled node: ", node->ToString()));
  }

  absl::Status HandleAdd(BinOp* binop) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* op) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArraySlice(ArraySlice* slice) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleArrayConcat(ArrayConcat* concat) override;
  absl::Status HandleAssert(Assert* assert_op) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleCover(Cover* cover) override {
    // TODO(rspringer): 2021-09-17: Add coverpoint support to the JIT.
    // GitHub issue #499.
    return absl::OkStatus();
  }
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* op) override;
  absl::Status HandleParam(Param* param) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* binop) override;
  absl::Status HandleSMod(BinOp* binop) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleShll(BinOp* binop) override;
  absl::Status HandleShra(BinOp* binop) override;
  absl::Status HandleShrl(BinOp* binop) override;
  absl::Status HandleSub(BinOp* binop) override;
  absl::Status HandleTrace(Trace* trace_op) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* binop) override;
  absl::Status HandleUMod(BinOp* binop) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* op) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

  // Returns the node of the function/proc which is used as the return value of
  // the LLVM function. This is necessary because procs do not have return
  // values. In this case the recurrent next-state value is used.
  static Node* GetEffectiveReturnValue(FunctionBase* function_base);

  // Creates a zero-valued LLVM constant for the given type, be it a Bits,
  // Array, or Tuple.
  static llvm::Constant* CreateTypedZeroValue(llvm::Type* type);

  // Loads a value of type `data_type` from a location indicated by the pointer
  // at the `index`-th slot in the array pointed to by
  // `pointer_array`. `pointer_array_type` is the type of the array of pointers.
  static llvm::Value* LoadFromPointerArray(int64_t index, llvm::Type* data_type,
                                           llvm::Value* pointer_array,
                                           int64_t pointer_array_size,
                                           llvm::IRBuilder<>* builder);

  // Marks the given buffer of the given size (in bytes) as "unpoisoned" for
  // MSAN - in other words, prevent false positives from being thrown when
  // running under MSAN (since it can't yet follow values into LLVM space (it
  // might be able to _technically_, but we've not enabled it).
  static void UnpoisonBuffer(llvm::Value* buffer, int64_t size,
                             llvm::IRBuilder<>* builder);

 protected:
  FunctionBuilderVisitor(llvm::Module* module, llvm::Function* llvm_fn,
                         FunctionBase* xls_fn,
                         LlvmTypeConverter* type_converter, bool is_top);

  llvm::LLVMContext& ctx() { return ctx_; }
  llvm::Module* module() { return module_; }
  llvm::Function* llvm_fn() { return llvm_fn_; }
  llvm::IRBuilder<>* builder() { return builder_.get(); }
  LlvmTypeConverter* type_converter() { return type_converter_; }
  absl::flat_hash_map<Node*, llvm::Value*>& node_map() { return node_map_; }

  // Actual driver for the building process (Build is a creation/init shim).
  absl::Status BuildInternal();

  // Saves the assocation between the given XLS IR Node and the matching LLVM
  // Value.
  absl::Status StoreResult(Node* node, llvm::Value* value);

  // After the original arguments, JIT-compiled functions always end with
  // the following four pointer arguments: output buffer, interpreter events
  // temporary, user data and JIT runtime. These are descriptive convenience
  // functions for getting them.
  llvm::Value* GetJitRuntimePtr() {
    return llvm_fn_->getArg(llvm_fn_->arg_size() - 1);
  }
  llvm::Value* GetUserDataPtr() {
    return llvm_fn_->getArg(llvm_fn_->arg_size() - 2);
  }
  llvm::Value* GetInterpreterEventsPtr() {
    return llvm_fn_->getArg(llvm_fn_->arg_size() - 3);
  }
  llvm::Value* GetOutputPtr() {
    return llvm_fn_->getArg(llvm_fn_->arg_size() - 4);
  }

  // Updates the active IRBuilder. This is used when dealing with branches (and
  // _ONLY_ with branches), which require new BasicBlocks, and thus, new
  // IRBuilders. When branches re-converge, this function should be called with
  // the IRBuilder for that new BasicBlock.
  void set_builder(std::unique_ptr<llvm::IRBuilder<>> new_builder) {
    builder_ = std::move(new_builder);
  }

 private:
  // Common handler for all arithmetic ops.
  absl::Status HandleArithOp(ArithOp* arith_op);

  // Common handler for all binary ops.
  absl::Status HandleBinOp(BinOp* binop);

  // Generates all shift operations.
  llvm::Value* EmitShiftOp(Op op, llvm::Value* lhs, llvm::Value* rhs);

  // Generates a divide operation.
  llvm::Value* EmitDiv(llvm::Value* lhs, llvm::Value* rhs, bool is_signed);

  // Generates a modulo operation.
  llvm::Value* EmitMod(llvm::Value* lhs, llvm::Value* rhs, bool is_signed);

  // Local struct to hold the individual elements of a (possibly) compound
  // comparison.
  struct CompareTerm {
    llvm::Value* lhs;
    llvm::Value* rhs;
  };

  // Expand the lhs and rhs of a comparison into a vector of the individual leaf
  // terms to compare.
  absl::StatusOr<std::vector<CompareTerm>> ExpandTerms(Node* lhs, Node* rhs,
                                                       Node* src);

  // ORs together all elements in the two given values, be they Bits, Arrays, or
  // Tuples.
  llvm::Value* CreateAggregateOr(llvm::Value* lhs, llvm::Value* rhs);

  // Converts the given XLS Value into an LLVM constant.
  absl::StatusOr<llvm::Constant*> ConvertToLlvmConstant(Type* type,
                                                        const Value& value);

  // Looks up and returns the given function in the module, translating it into
  // LLVM first, if necessary.
  absl::StatusOr<llvm::Function*> GetModuleFunction(Function* xls_function);

  // Returns the result of indexing into 'array' using the scalar index value
  // 'index'. 'array_size' is the number of elements in the array.
  absl::StatusOr<llvm::Value*> IndexIntoArray(llvm::Value* array,
                                              llvm::Value* index,
                                              int64_t array_size);

  // Build the LLVM IR to invoke the callback that records assertions.
  absl::Status InvokeAssertCallback(llvm::IRBuilder<>* builder,
                                    const std::string& message);

  // Build the LLVM IR to invoke the callback that creates a trace buffer.
  absl::StatusOr<llvm::Value*> InvokeCreateBufferCallback(
      llvm::IRBuilder<>* builder);

  // Build the LLVM IR that handles string fragment format steps.
  absl::Status InvokeStringStepCallback(llvm::IRBuilder<>* builder,
                                        const std::string& step_string,
                                        llvm::Value* buffer_ptr);

  // Build the LLVM IR that handles formatting a runtime value according to a
  // format preference.
  absl::Status InvokeFormatStepCallback(llvm::IRBuilder<>* builder,
                                        FormatPreference format,
                                        xls::Type* operand_type,
                                        llvm::Value* operand,
                                        llvm::Value* buffer_ptr);

  // Build the LLVM IR to invoke the callback that records traces.
  absl::Status InvokeRecordTraceCallback(llvm::IRBuilder<>* builder,
                                         llvm::Value* buffer_ptr);

  // Get the required assertion status and user data arguments that need to be
  // included at the end of the argument list for every function call.
  std::vector<llvm::Value*> GetRequiredArgs() {
    return {GetInterpreterEventsPtr(), GetUserDataPtr(), GetJitRuntimePtr()};
  }

  llvm::LLVMContext& ctx_;
  llvm::Module* module_;
  llvm::Function* llvm_fn_;
  FunctionBase* xls_fn_;
  std::unique_ptr<llvm::IRBuilder<>> builder_;
  LlvmTypeConverter* type_converter_;

  // Per the Build() comment, is_top_ is true if this is the top-level function
  // being translated.
  bool is_top_;

  // The last value constructed during this traversal - represents the return
  // from calculation.
  llvm::Value* return_value_;

  // Maps an XLS Node to the resulting LLVM Value.
  absl::flat_hash_map<Node*, llvm::Value*> node_map_;

  // Holds storage for array indexing ops. Since we need to dump arrays to
  // storage to extract elements (i.e., for GEPs), it makes sense to only create
  // and store the array once.
  absl::flat_hash_map<llvm::Value*, llvm::AllocaInst*> array_storage_;
};

}  // namespace xls

#endif  // XLS_JIT_FUNCTION_BUILDER_VISITOR_H_
