// Copyright 2022 The XLS Authors
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
#ifndef XLS_JIT_IR_BUILDER_VISITOR_H_
#define XLS_JIT_IR_BUILDER_VISITOR_H_

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

// Abstraction gathering together the necessary context for emitting the LLVM IR
// for a given node. This data structure decouples IR generation for the the
// top-level function from the IR generation of each node. This enables, for
// example, emitting a node as a separate functoin.
class NodeIrContext {
 public:
  // The evironment pointers passed into the top-level function.
  struct Environment {
    // Pointer to the interpreter events.
    llvm::Value* events;
    // Pointer to the user data used in send/receive operations.
    llvm::Value* user_data;
    // Pointer to the JIT runtime.
    llvm::Value* jit_runtime;
  };

  // Args:
  //   node : The XLS node the LLVM IR is being generated for.
  //   operand_names : the names of the operand of `node`. If possible, these
  //     will be used to name the LLVM values of the operands.
  //   operands : the LLVM values of the operands of `node`.
  //   environment : optional top-level environment pointers. Should be
  //     specified if the LLVM IR generated for `node` needs access to these
  //     values.
  static absl::StatusOr<NodeIrContext> Create(
      Node* node, absl::Span<const std::string> operand_names,
      LlvmTypeConverter* type_converter, llvm::Module* module,
      std::optional<Environment> environment = std::nullopt);

  // Completes the LLVM function by adding a return statement with the given
  // result. If `exit_builder` is specified then it is used to build the return
  // statement. Otherwise `entry_builder()` is used.
  void Finalize(llvm::Value* result,
                std::optional<llvm::IRBuilder<>*> exit_builder);

  Node* node() const { return node_; }

  // The LLVM function that the generated code for the node is placed into,
  llvm::Function* llvm_function() const { return llvm_function_; }

  // Returns the IR builder to use for building code for this XLS node.
  llvm::IRBuilder<>& builder() const { return *builder_; }

  // Returns the arguments of the LLVM function corresponding to the operands of
  // the XLS node.
  absl::Span<llvm::Value* const> operands() const { return operands_; }

  // Returns the argument of the LLVM function corresponding to the i-th operand
  // of the XLS node.
  llvm::Value* operand(int64_t i) const { return operands_.at(i); }

  // Get one of the environment arguments. CHECK fails is the environment was
  // not specified on construction.
  llvm::Value* GetInterpreterEvents() const { return environment_->events; }
  llvm::Value* GetUserData() const { return environment_->user_data; }
  llvm::Value* GetJitRuntime() const { return environment_->jit_runtime; }

  bool HasEnvironment() const { return environment_.has_value(); }

  LlvmTypeConverter* type_converter() const { return type_converter_; }

 private:
  // Creates an empty LLVM function with the appropriate type signature.
  static llvm::Function* CreateFunction(Node* node,
                                        LlvmTypeConverter* type_converter,
                                        llvm::Module* module,
                                        std::optional<Environment> environment);

  Node* node_;
  llvm::Function* llvm_function_;
  std::unique_ptr<llvm::IRBuilder<>> builder_;
  LlvmTypeConverter* type_converter_;
  std::vector<llvm::Value*> operands_;
  std::optional<Environment> environment_;
};

// Visitor to construct LLVM IR for each encountered XLS IR node. Based on
// DfsVisitorWithDefault to highlight any unhandled IR nodes. This class handles
// translation of almost all XLS node types. The exception are Params and
// proc-specific or block-specific operands (e.g., Send or Receive) which should
// be handled in derived classes specific to XLS Functions, Procs or Blocks.
//
// The visitor builds up top-level "dispatch" function and one function for each
// XLS node. The dispatch function calls the node functions in topological order
// feeding the appropriate operand values.
class IrBuilderVisitor : public DfsVisitorWithDefault {
 public:
  // Args:
  //  llvm_fn: empty function which in which the translation of `xls_fn` will be
  //    built.
  //  xls_fn: XLS FunctionBase being translated.
  //  function_builder: a callable which creates an LLVM function implementing
  //    the given XLS function. This is used to build functions called by nodes
  //    in `xls_fn` (such as invoke).
  IrBuilderVisitor(llvm::Function* llvm_fn, FunctionBase* xls_fn,
                   LlvmTypeConverter* type_converter,
                   std::function<absl::StatusOr<llvm::Function*>(Function*)>
                       function_builder);
  absl::Status DefaultHandler(Node* node) override;

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
  absl::Status HandleCover(Cover* cover) override;
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
  absl::Status HandleSMulp(PartialProductOp* mul) override;
  absl::Status HandleUMulp(PartialProductOp* mul) override;
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
  absl::Status HandlePrioritySel(PrioritySelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* op) override;
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

  // Creates a zero-valued LLVM constant for the given type, be it a Bits,
  // Array, or Tuple.
  static llvm::Constant* CreateTypedZeroValue(llvm::Type* type);

  // Loads a value of type `data_type` from a location indicated by the pointer
  // at the `index`-th slot in the array pointed to by `pointer_array`.
  // `pointer_array_size` is the size of the array of pointers.
  static llvm::Value* LoadFromPointerArray(int64_t index, llvm::Type* data_type,
                                           llvm::Value* pointer_array,
                                           int64_t pointer_array_size,
                                           llvm::IRBuilder<>* builder);

  // Loads a value of type pointer to `data_type` from the `index`-th slot in
  // the array pointed to by `pointer_array`. `pointer_array_size` is the
  // size of the array of pointers.
  static llvm::Value* LoadPointerFromPointerArray(int64_t index,
                                                  llvm::Type* data_type,
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
  llvm::LLVMContext& ctx() { return dispatch_function()->getContext(); }
  llvm::Module* module() { return dispatch_function()->getParent(); }

  // Return the top-level dispatch function. The dispatch function calls each of
  // the XLS node functions in sequence.
  llvm::Function* dispatch_function() { return dispatch_fn_; }

  // Returns the top-level builder for the function. This builder is initialized
  // to the entry block of `dispatch_function()`.
  llvm::IRBuilder<>* dispatch_builder() { return dispatch_builder_.get(); }
  LlvmTypeConverter* type_converter() { return type_converter_; }
  absl::flat_hash_map<Node*, llvm::Value*>& node_map() { return node_map_; }

  // Saves the assocation between the given XLS IR Node and the matching LLVM
  // Value.
  absl::Status StoreResult(Node* node, llvm::Value* value);

  // After the original arguments, JIT-compiled functions always end with
  // the following three pointer arguments:
  //   interpreter events
  //   temporary user data
  //   JIT runtime.
  // These are descriptive convenience functions for getting them.
  llvm::Value* GetJitRuntimePtr() {
    return dispatch_fn_->getArg(dispatch_fn_->arg_size() - 1);
  }
  llvm::Value* GetUserDataPtr() {
    return dispatch_fn_->getArg(dispatch_fn_->arg_size() - 2);
  }
  llvm::Value* GetInterpreterEventsPtr() {
    return dispatch_fn_->getArg(dispatch_fn_->arg_size() - 3);
  }

  // Common handler for binary, unary, and nary nodes. `build_results` produces
  // the llvm Value to use as the result.
  absl::Status HandleUnaryOp(
      Node* node, std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>&)>,
      bool is_signed = false);
  absl::Status HandleBinaryOp(
      Node* node,
      std::function<llvm::Value*(llvm::Value*, llvm::Value*,
                                 llvm::IRBuilder<>&)>,
      bool is_signed = false);
  absl::Status HandleNaryOp(
      Node* node, std::function<llvm::Value*(absl::Span<llvm::Value* const>,
                                             llvm::IRBuilder<>&)>);

  // HandleBinaryOp variant which converts the operands to result type prior to
  // calling `build_result`.
  absl::Status HandleBinaryOpWithOperandConversion(
      Node* node,
      std::function<llvm::Value*(llvm::Value*, llvm::Value*,
                                 llvm::IRBuilder<>&)>
          build_result,
      bool is_signed);

  // Gets the built function representing the given XLS function, or builds it
  // if it has not yet been built.
  absl::StatusOr<llvm::Function*> GetOrBuildFunction(Function* function);

  // Creates and returns a new NodeIrContext for the given XLS node.
  absl::StatusOr<NodeIrContext> NewNodeIrContext(
      Node* node, absl::Span<const std::string> operand_names,
      bool include_environment = false);

  // Finalizes the given NodeIrContext (adds a return statement with the given
  // result) and adds a call in the top-level LLVM function to the node
  // function.
  absl::Status FinalizeNodeIrContext(
      NodeIrContext& node_context, llvm::Value* result,
      std::optional<std::unique_ptr<llvm::IRBuilder<>>> exit_builder =
          std::nullopt);

  llvm::Value* MaybeAsSigned(llvm::Value* v, Type* xls_type,
                             llvm::IRBuilder<>& builder, bool is_signed) {
    if (is_signed) {
      return type_converter()->AsSignedValue(v, xls_type, builder);
    }
    return v;
  }

  llvm::Function* dispatch_fn_;
  FunctionBase* xls_fn_;
  std::unique_ptr<llvm::IRBuilder<>> dispatch_builder_;
  LlvmTypeConverter* type_converter_;
  std::function<absl::StatusOr<llvm::Function*>(Function*)> function_builder_;

  // Maps an XLS Node to the resulting LLVM Value.
  absl::flat_hash_map<Node*, llvm::Value*> node_map_;
};

}  // namespace xls

#endif  // XLS_JIT_IR_BUILDER_VISITOR_H_
