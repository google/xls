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

#include "absl/status/statusor.h"
#include "llvm/include/llvm/IR/Function.h"
#include "xls/ir/node.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/orc_jit.h"

namespace xls {

// Returns whether the given node should be materialized at is uses rather than
// being written to a buffer to pass to the JITted node function. Only possible
// for nodes whose value is known at compile time (e.g., Literals).
bool ShouldMaterializeAtUse(Node* node);

// An object gathering necessary information for jitting XLS functions, procs,
// etc.
class JitBuilderContext {
 public:
  JitBuilderContext(
      OrcJit& orc_jit,
      std::optional<JitChannelQueueManager*> queue_mgr = std::nullopt)
      : module_(orc_jit.NewModule("__module")),
        orc_jit_(orc_jit),
        type_converter_(orc_jit.GetContext(),
                        orc_jit.CreateDataLayout().value()),
        queue_manager_(queue_mgr) {}

  llvm::Module* module() const { return module_.get(); }
  llvm::LLVMContext& context() const { return module_->getContext(); }
  OrcJit& orc_jit() { return orc_jit_; }
  LlvmTypeConverter& type_converter() { return type_converter_; }

  // Destructively returns the underlying llvm::Module.
  std::unique_ptr<llvm::Module> ConsumeModule() { return std::move(module_); }

  // Returns the llvm::Function implementing the given FunctionBase.
  llvm::Function* GetLlvmFunction(FunctionBase* xls_fn) const {
    return llvm_functions_.at(xls_fn);
  }

  // Sets the llvm::Function implementing the given FunctionBase to
  // `llvm_function`.
  void SetLlvmFunction(FunctionBase* xls_fn, llvm::Function* llvm_function) {
    llvm_functions_[xls_fn] = llvm_function;
  }

  std::optional<JitChannelQueueManager*> queue_manager() const {
    return queue_manager_;
  }

 private:
  std::unique_ptr<llvm::Module> module_;
  OrcJit& orc_jit_;
  LlvmTypeConverter type_converter_;
  std::optional<JitChannelQueueManager*> queue_manager_;

  // Map from FunctionBase to the associated JITed llvm::Function.
  absl::flat_hash_map<FunctionBase*, llvm::Function*> llvm_functions_;
};

// Abstraction representing an llvm::Function implementing an xls::Node. The
// function has the following signature:
//
//   bool f(void* operand_0_ptr, ..., void* operand_n_ptr,
//          void* output_0_ptr, ... void* output_m_ptr)
//
// The function can optionally include metadata arguments passed from the
// top-level jitted functions:
//
//   bool f(void* operand_0_ptr, ..., void* operand_n_ptr,
//          void* output_0_ptr, ... void* output_m_ptr,
//          void* inputs, void* outputs, void* tmp_buffer,
//          void* events, void* user_data, void* runtime)
//
// Operand pointer arguments point to buffers holding argument values. Output
// pointer arguments point to buffers which must be filled with the node's
// computed value. A node has more than one output pointer if it is, for
// example, the next state node for more than one state element in a proc (and
// possibly other corner cases).
//
// The return value of the function indicates whether the execution of the
// FunctionBase should be interrupted (return true) or continue (return
// false). The return value is only used for nodes which may block execution
// (blocking receives).
struct NodeFunction {
  Node* node;
  llvm::Function* function;

  // Vector of nodes which should be passed in as the operand arguments. This is
  // a deduplicated list of the operands of the node.
  std::vector<Node*> operand_arguments;

  // The number of output pointer arguments.
  int64_t output_arg_count;

  // Whether the function has metadata data arguments (events, jit runtime, temp
  // buffer, etc).
  bool has_metadata_args;
};

// Create an llvm::Function implementing `node`. `output_arg_count` is the
// number of output buffer arguments (see NodeFunction above).
absl::StatusOr<NodeFunction> CreateNodeFunction(Node* node,
                                                int64_t output_arg_count,
                                                JitBuilderContext& jit_context);

// Constructs a call to memcpy from `src` to `tgt` of `size` bytes.
llvm::Value* LlvmMemcpy(llvm::Value* tgt, llvm::Value* src, int64_t size,
                        llvm::IRBuilder<>& builder);

}  // namespace xls

#endif  // XLS_JIT_IR_BUILDER_VISITOR_H_
