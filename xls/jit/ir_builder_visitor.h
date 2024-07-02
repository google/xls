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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/Value.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

// Returns whether the given node should be materialized at is uses rather than
// being written to a buffer to pass to the JITted node function. Only possible
// for nodes whose value is known at compile time (e.g., Literals).
bool ShouldMaterializeAtUse(Node* node);

// An object gathering necessary information for jitting XLS functions, procs,
// etc.
class JitBuilderContext {
 public:
  explicit JitBuilderContext(LlvmCompiler& llvm_compiler, FunctionBase* top)
      : module_(llvm_compiler.NewModule("__module")),
        llvm_compiler_(llvm_compiler),
        top_(top),
        type_converter_(llvm_compiler_.GetContext(),
                        llvm_compiler_.CreateDataLayout().value()) {
    CHECK_EQ(module_->getTargetTriple(), llvm_compiler_.target_triple());
  }

  llvm::Module* module() const { return module_.get(); }
  llvm::LLVMContext& context() const { return module_->getContext(); }
  LlvmCompiler& llvm_compiler() { return llvm_compiler_; }
  LlvmTypeConverter& type_converter() { return type_converter_; }
  FunctionBase* top() const { return top_; }

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

  // Get (or allocate) a slot for the channel queue associated with the given
  // channel name. Returns the index of the slot.
  int64_t GetOrAllocateQueueIndex(std::string_view channel_name) {
    if (queue_indices_.contains(channel_name)) {
      return queue_indices_.at(channel_name);
    }
    int64_t index = queue_indices_.size();
    queue_indices_[channel_name] = index;
    return index;
  }

  // Returns map of channel name to queue index. The JITted function is passed a
  // vector of channel queues which the JITted code for sends/receives indexes
  // into to get the appropriate channel queue. These indices are baked into the
  // JITted code.
  const absl::btree_map<std::string, int64_t>& queue_indices() const {
    return queue_indices_;
  }

  std::string MangleFunctionName(FunctionBase* f) {
    if (f == top() || !llvm_compiler().IsSharedCompilation()) {
      return f->name();
    }
    return absl::StrFormat("%s____SUBROUTINE_OF_%s", f->name(), top()->name());
  }

 private:
  std::unique_ptr<llvm::Module> module_;
  LlvmCompiler& llvm_compiler_;
  FunctionBase* top_;
  LlvmTypeConverter type_converter_;

  // Map from FunctionBase to the associated JITed llvm::Function.
  absl::flat_hash_map<FunctionBase*, llvm::Function*> llvm_functions_;

  // A map from channel name to queue index.
  absl::btree_map<std::string, int64_t> queue_indices_;
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

// Information about the layout of the 'metadata' args that can be optionally
// requested for node functions.
class JitCompilationMetadata {
 public:
  virtual ~JitCompilationMetadata() = default;
  // Get the value of the node 'n' in the input arguments at base_ptr. The
  // base_ptr point to the full input array.
  virtual absl::StatusOr<llvm::Value*> GetInputBufferFrom(
      Node* n, llvm::Value* base_ptr, llvm::IRBuilder<>& builder) const = 0;
  // Is 'node' an input and therefore in the global input metadata.
  virtual bool IsInputNode(Node* n) const = 0;
};

// Create an llvm::Function implementing `node`. `output_arg_count` is the
// number of output buffer arguments (see NodeFunction above).
absl::StatusOr<NodeFunction> CreateNodeFunction(
    Node* node, int64_t output_arg_count,
    const JitCompilationMetadata& metadata, JitBuilderContext& jit_context);

// Constructs a call to memcpy from `src` to `tgt` of `size` bytes.
llvm::Value* LlvmMemcpy(llvm::Value* tgt, llvm::Value* src, int64_t size,
                        llvm::IRBuilder<>& builder);

}  // namespace xls

#endif  // XLS_JIT_IR_BUILDER_VISITOR_H_
