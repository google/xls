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
#include "xls/jit/function_base_jit.h"

#include "absl/container/flat_hash_map.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/IR/Value.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {
namespace {

// Loads a pointer from the `index`-th slot in the array pointed to by
// `pointer_array`.
llvm::Value* LoadPointerFromPointerArray(int64_t index,
                                         llvm::Value* pointer_array,
                                         llvm::IRBuilder<>* builder) {
  llvm::LLVMContext& context = builder->getContext();
  llvm::Type* pointer_array_type =
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(context), 0);
  llvm::Value* gep = builder->CreateGEP(
      pointer_array_type, pointer_array,
      {
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0),
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), index),
      });

  return builder->CreateLoad(llvm::PointerType::get(context, 0), gep);
}

// Loads a value of type `data_type` from a location indicated by the pointer at
// the `index`-th slot in the array pointed to by `pointer_array`.
llvm::Value* LoadFromPointerArray(int64_t index, llvm::Type* data_type,
                                  llvm::Value* pointer_array,
                                  llvm::IRBuilder<>* builder) {
  llvm::Value* data_ptr =
      LoadPointerFromPointerArray(index, pointer_array, builder);
  return builder->CreateLoad(data_type, data_ptr);
}

// Marks the given buffer of the given size (in bytes) as "unpoisoned" for MSAN
// - in other words, prevent false positives from being thrown when running
// under MSAN (since it can't yet follow values into LLVM space (it might be
// able to _technically_, but we've not enabled it).
void UnpoisonBuffer(llvm::Value* buffer, int64_t size,
                    llvm::IRBuilder<>* builder) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  llvm::LLVMContext& context = builder->getContext();
  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(context),
                             absl::bit_cast<uint64_t>(&__msan_unpoison));
  llvm::Type* void_type = llvm::Type::getVoidTy(context);
  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);
  llvm::Type* size_t_type =
      llvm::Type::getIntNTy(context, sizeof(size_t) * CHAR_BIT);
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, {ptr_type, size_t_type}, false);
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));

  std::vector<llvm::Value*> args = {buffer,
                                    llvm::ConstantInt::get(size_t_type, size)};

  builder->CreateCall(fn_type, fn_ptr, args);
#endif
}

// Abstraction around an LLVM function of type `JitFunctionType`. Signature of
// these functions:
//
//  ReturnT f(const uint8_t* const* inputs,
//            uint8_t* const* outputs, void* temp_buffer,
//            InterpreterEvents* events, void* user_data,
//            JitRuntime* jit_runtime)
//
// This type of function is used for the jitted functions implementing XLS
// functions, procs, etc, as well as for the partition functions called from
// within the jitted functions.
//
// `input_args` are the Nodes whose values are passed in the `inputs` function
// argument. `output_args` are Nodes whose values are written out to buffers
// indicated by the `outputs` function argument.
class LlvmFunctionWrapper {
 public:
  struct FunctionArg {
    std::string name;
    llvm::Type* type;
  };

  // Creates a new wrapper. Arguments are described above. `extra_arg`, if
  // specified will be added to the function signature after all other
  // args.
  static LlvmFunctionWrapper Create(
      std::string_view name, absl::Span<Node* const> input_args,
      absl::Span<Node* const> output_args, llvm::Type* return_type,
      const JitBuilderContext& jit_context,
      std::optional<FunctionArg> extra_arg = std::nullopt) {
    llvm::Type* ptr_type = llvm::PointerType::get(jit_context.context(), 0);
    std::vector<llvm::Type*> param_types(6, ptr_type);
    if (extra_arg.has_value()) {
      param_types.push_back(extra_arg->type);
    }
    llvm::FunctionType* function_type =
        llvm::FunctionType::get(return_type, param_types,
                                /*isVarArg=*/false);
    XLS_CHECK_EQ(jit_context.module()->getFunction(name), nullptr)
        << absl::StreamFormat(
               "Function named `%s` already exists in LLVM module", name);
    llvm::Function* fn = llvm::cast<llvm::Function>(
        jit_context.module()
            ->getOrInsertFunction(name, function_type)
            .getCallee());
    fn->getArg(0)->setName("input_ptrs");
    fn->getArg(1)->setName("output_ptrs");
    fn->getArg(2)->setName("tmp_buffer");
    fn->getArg(3)->setName("events");
    fn->getArg(4)->setName("user_data");
    fn->getArg(5)->setName("runtime");
    if (extra_arg.has_value()) {
      fn->getArg(6)->setName(extra_arg->name);
    }

    LlvmFunctionWrapper wrapper(input_args, output_args);
    wrapper.fn_ = fn;
    wrapper.fn_type_ = function_type;
    auto basic_block = llvm::BasicBlock::Create(fn->getContext(), "entry", fn,
                                                /*InsertBefore=*/nullptr);
    wrapper.entry_builder_ = std::make_unique<llvm::IRBuilder<>>(basic_block);

    return wrapper;
  }

  llvm::Function* function() const { return fn_; }
  llvm::FunctionType* function_type() const { return fn_type_; }
  llvm::IRBuilder<>& entry_builder() { return *entry_builder_; }

  // Returns whether `node` is one of the nodes whose value is passed in via the
  // `inputs` function argument.
  bool IsInputNode(Node* node) const { return input_indices_.contains(node); }

  // Returns the index within the array passed into the `input` function
  // argument corresponding to `node`. CHECK fails if `node` is not an input
  // node.
  int64_t GetInputArgIndex(Node* node) const { return input_indices_.at(node); }

  // Returns the input buffer for `node` by loading the pointer from the
  // `inputs` argument. `node` must be an input node.
  llvm::Value* GetInputBuffer(Node* node, llvm::IRBuilder<>& builder) {
    return LoadPointerFromPointerArray(GetInputArgIndex(node), GetInputsArg(),
                                       &builder);
  }

  // Returns whether `node` is one of the nodes whose value should be written to
  // one of the output buffers passed in via the `inputs` function argument.
  bool IsOutputNode(Node* node) const { return output_indices_.contains(node); }

  // Returns the index(es) within the array passed into the `output` function
  // argument corresponding to `node`. `node` must be an output node.
  absl::Span<const int64_t> GetOutputArgIndices(Node* node) const {
    return output_indices_.at(node);
  }

  // Returns the output buffer(s) passed in via `outputs` function argument
  // corresponding to `node`. Created by loading pointers from the `outputs`
  // array argument.  `node` must be an output node.
  std::vector<llvm::Value*> GetOutputBuffers(Node* node,
                                             llvm::IRBuilder<>& builder) {
    std::vector<llvm::Value*> output_buffers;
    for (int64_t i : GetOutputArgIndices(node)) {
      output_buffers.push_back(
          LoadPointerFromPointerArray(i, GetOutputsArg(), &builder));
    }
    return output_buffers;
  }

  // Returns the first output buffer(s) passed in via `outputs` function
  // argument corresponding to `node`. `node` must be an output node.
  llvm::Value* GetFirstOutputBuffer(Node* node, llvm::IRBuilder<>& builder) {
    return LoadPointerFromPointerArray(GetOutputArgIndices(node).front(),
                                       GetOutputsArg(), &builder);
  }

  // Returns one of the various function argument.
  llvm::Value* GetInputsArg() const { return fn_->getArg(0); }
  llvm::Value* GetOutputsArg() const { return fn_->getArg(1); }
  llvm::Value* GetTempBufferArg() const { return fn_->getArg(2); }
  llvm::Value* GetInterpreterEventsArg() const { return fn_->getArg(3); }
  llvm::Value* GetUserDataArg() const { return fn_->getArg(4); }
  llvm::Value* GetJitRuntimeArg() const { return fn_->getArg(5); }
  std::optional<llvm::Value*> GetExtraArg() const {
    if (fn_->arg_size() == 7) {
      return fn_->getArg(6);
    }
    return std::nullopt;
  }

  // Returns a computed offset into the temp buffer passed in as the
  // `temp_buffer` argument to the jitted function.
  llvm::Value* GetOffsetIntoTempBuffer(int64_t offset,
                                       llvm::IRBuilder<>& builder) {
    llvm::Type* i64 = llvm::Type::getInt64Ty(function()->getContext());
    llvm::Value* start_buffer = builder.CreatePtrToInt(GetTempBufferArg(), i64);
    llvm::Value* start_plus_offset =
        builder.CreateAdd(start_buffer, llvm::ConstantInt::get(i64, offset));
    return builder.CreateIntToPtr(
        start_plus_offset, llvm::PointerType::get(function()->getContext(), 0));
  }

 private:
  LlvmFunctionWrapper(absl::Span<Node* const> input_args,
                      absl::Span<Node* const> output_args)
      : input_args_(input_args.begin(), input_args.end()),
        output_args_(output_args.begin(), output_args.end()) {
    for (int64_t i = 0; i < input_args.size(); ++i) {
      XLS_CHECK(!input_indices_.contains(input_args[i]));
      input_indices_[input_args[i]] = i;
    }
    for (int64_t i = 0; i < output_args.size(); ++i) {
      output_indices_[output_args[i]].push_back(i);
    }
  }

  llvm::Function* fn_;
  llvm::FunctionType* fn_type_;
  std::unique_ptr<llvm::IRBuilder<>> entry_builder_;

  std::vector<Node*> input_args_;
  std::vector<Node*> output_args_;
  absl::flat_hash_map<Node*, int64_t> input_indices_;
  absl::flat_hash_map<Node*, std::vector<int64_t>> output_indices_;
};

// Allocator for the temporary buffer used within jitted functions. The
// temporary buffer holds node values which are not inputs or outputs and must
// be passed between partition functions and so cannot be allocated on the stack
// with alloca.
class TempBufferAllocator {
 public:
  explicit TempBufferAllocator(const LlvmTypeConverter* type_converter)
      : type_converter_(type_converter) {}

  // Allocate a buffer for `node`. Returns the offset within the temporary
  // buffer.
  int64_t Allocate(Node* node) {
    XLS_CHECK(!node_offsets_.contains(node));
    int64_t offset = current_offset_;
    int64_t node_size = type_converter_->GetTypeByteSize(node->GetType());
    node_offsets_[node] = offset;
    current_offset_ += RoundUpToNearest<int64_t>(node_size, 4);
    XLS_VLOG(3) << absl::StreamFormat(
        "Allocated %s at offset %d (size = %d): total size %d", node->GetName(),
        offset, node_size, current_offset_);
    return offset;
  }

  // Returns true if `node` has a allocated buffer.
  bool IsAllocated(Node* node) const { return node_offsets_.contains(node); }

  // Returns the offset of the buffer allocated for `node`.
  int64_t GetOffset(Node* node) const { return node_offsets_.at(node); }

  // Returns the total size of the allocated memory.
  int64_t size() const { return current_offset_; }

 private:
  const LlvmTypeConverter* type_converter_;
  absl::flat_hash_map<Node*, int64_t> node_offsets_;
  int64_t current_offset_ = 0;
};

// The maximum number of xls::Nodes in a partition.
static constexpr int64_t kMaxPartitionSize = 100;

// Data structure holding a set of nodes which should be emitted together in a
// single LLVM function.
struct Partition {
  // Whether this partition is a point at which execution of the FunctionBase
  // can be interrupted and resumed.
  std::optional<int64_t> continuation_point;
  std::vector<Node*> nodes;
};

bool IsContinuationPoint(Node* node) {
  return node->Is<Receive>() && node->As<Receive>()->is_blocking();
}

// Divides the nodes of the given function base into a topologically sorted
// sequence of partitions. Each partition then becomes a separate function in
// the LLVM module.
// TODO(meheff): 2022/09/09 Tune this algorithm to maximize performance.
std::vector<Partition> PartitionFunctionBase(FunctionBase* f) {
  absl::flat_hash_map<Node*, int64_t> partition_map;
  absl::flat_hash_set<int64_t> continuation_partitions;

  // Naively assign nodes to partitions based on a topological sort. First N
  // nodes go to partition 0, next N nodes go to partition 1, etc.
  int64_t current_partition = 0;
  int64_t partition_size = 0;
  for (Node* node : TopoSort(f)) {
    if (IsContinuationPoint(node)) {
      XLS_CHECK(f->IsProc())
          << "Receive nodes are only supported in procs in the JIT";
      // Nodes which form continuation points are placed in their own
      // partition.
      if (partition_size != 0) {
        ++current_partition;
      }
      partition_map[node] = current_partition;
      continuation_partitions.insert(current_partition);
      ++current_partition;
      partition_size = 0;
      continue;
    }
    partition_map[node] = current_partition;
    ++partition_size;
    if (partition_size > kMaxPartitionSize) {
      ++current_partition;
      partition_size = 0;
    }
  }

  // Move literals down as far as possible so they appear in the same partition
  // as their uses.
  for (Node* node : f->nodes()) {
    if (node->Is<Literal>()) {
      int64_t min_partition = std::numeric_limits<int64_t>::max();
      for (Node* user : node->users()) {
        min_partition = std::min(min_partition, partition_map.at(user));
      }
      // Avoid putting literals in continuation partitions. These partitions
      // should only have a single node.
      if (min_partition != std::numeric_limits<int64_t>::max() &&
          !continuation_partitions.contains(min_partition)) {
        partition_map[node] = min_partition;
      }
    }
  }

  // Assemble nodes into partitions;
  std::vector<Partition> partitions;
  for (Node* node : TopoSort(f)) {
    int64_t partition = partition_map.at(node);
    if (partitions.size() <= partition) {
      partitions.resize(partition + 1);
    }
    partitions[partition].nodes.push_back(node);
  }

  // Number continuation points starting at one because zero has special meaning
  // (beginning of proc).
  int64_t continuation_point = 1;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    if (continuation_partitions.contains(i)) {
      partitions[i].continuation_point = continuation_point++;
    }
  }

  return partitions;
}

// Builds an LLVM function of the given `name` which executes the given set of
// nodes. The signature of the partition function is the same as the jitted
// function implementing a FunctionBase (i.e., `JitFunctionType`). A partition
// function calls a sequence of node functions where each node function computes
// the value of a single node. The partition function is responsible for
// allocating or loading the buffers for the operands and results of each node
// function. For example, a partition function might look like (`x` and `y` are
// a couple nodes in the partition):
//
//   bool
//   __partition_f_0(const uint8_t* const* inputs,
//                   uint8_t* const* outputs,
//                   void* temp_buffer,
//                   InterpreterEvents* events,
//                   void* user_data,
//                   JitRuntime* jit_runtime) {
//     ...
//     x_operand_0 = /* load pointer from `inputs` */
//     x_operand_1 = /* load pointer from `inputs` */
//     x = alloca(...)
//     __x_function(x_operand_0, x_operand_1, x_output)
//     y_operand_0 = ...
//     y = /* load pointer from `outputs` */
//     __y_function(y_operand_0, y)
//     ...
//     return
//   }
//
// The partition functions are designed to be called sequentially inside of the
// jitted function implementing the function from which the partitions were
// created.
//
// The return value of the function indicates whether the execution of the
// FunctionBase should be interrupted (return true) or continue (return
// false). The return value is only checked for partitions which are
// continuation points (ie, contain a blocking receive).
//
// `global_input_nodes` and `global_output_nodes` are the set of nodes whose
// buffers are passed in via the `input`/`output` arguments of the function.
absl::StatusOr<llvm::Function*> BuildPartitionFunction(
    std::string_view name, const Partition& partition,
    absl::Span<Node* const> global_input_nodes,
    absl::Span<Node* const> global_output_nodes,
    const TempBufferAllocator& allocator, JitBuilderContext& jit_context) {
  LlvmFunctionWrapper wrapper = LlvmFunctionWrapper::Create(
      name, global_input_nodes, global_output_nodes,
      llvm::Type::getInt1Ty(jit_context.context()), jit_context);
  llvm::IRBuilder<>& b = wrapper.entry_builder();

  // Whether to interrupt execution of the FunctionBase. Only used for
  // partitions which are continuation points (ie, have a blocking receive).
  llvm::Value* interrupt_execution = nullptr;

  // The pointers to the buffers of nodes in the partition.
  absl::flat_hash_map<Node*, llvm::Value*> value_buffers;
  for (Node* node : partition.nodes) {
    if (wrapper.IsInputNode(node)) {
      // Node is an input node. There is no need to generate a node function for
      // this node (its value is an input and already computed). Simply copy the
      // value to output buffers associated with `node` (if any).
      XLS_RET_CHECK(!allocator.IsAllocated(node));
      if (wrapper.IsOutputNode(node)) {
        // `node` is also an output node. This can occur, for example, if a
        // state param is the next state value for a proc.
        llvm::Value* input_buffer = wrapper.GetInputBuffer(node, b);
        for (llvm::Value* output_buffer : wrapper.GetOutputBuffers(node, b)) {
          LlvmMemcpy(output_buffer, input_buffer,
                     jit_context.orc_jit().GetTypeConverter().GetTypeByteSize(
                         node->GetType()),
                     b);
        }
      }
      continue;
    }

    // Gather the pointers to the operands of `node`.
    std::vector<llvm::Value*> operand_buffers;
    for (Node* operand : node->operands()) {
      if (value_buffers.contains(operand)) {
        operand_buffers.push_back(value_buffers.at(operand));
        continue;
      }
      llvm::Value* arg;
      if (wrapper.IsInputNode(operand)) {
        // `operand` is a global input. Load the pointer to the buffer from the
        // input array argument.
        arg = wrapper.GetInputBuffer(operand, b);
      } else if (wrapper.IsOutputNode(operand)) {
        // `operand` is a global output. `operand` may have more than one buffer
        // in this case which is one of the pointer in the output array
        // argument. Arbitrarily choose the first.
        arg = wrapper.GetFirstOutputBuffer(operand, b);
      } else {
        // `operand` is stored inside the temporary buffer. These are values
        // which are passed between partitions but are not global inputs or
        // outputs.
        arg = wrapper.GetOffsetIntoTempBuffer(allocator.GetOffset(operand), b);
      }
      operand_buffers.push_back(arg);
      value_buffers[operand] = arg;
    }

    // Gather the buffers to which the value of `node` must be written.
    std::vector<llvm::Value*> output_buffers;
    if (wrapper.IsOutputNode(node)) {
      XLS_RET_CHECK(!allocator.IsAllocated(node));
      output_buffers = wrapper.GetOutputBuffers(node, b);
    } else if (allocator.IsAllocated(node)) {
      output_buffers = {
          wrapper.GetOffsetIntoTempBuffer(allocator.GetOffset(node), b)};
    } else {
      // `node` is used exclusively inside this partition (not an input, output,
      // nor has a temp buffer). Allocate a buffer on the stack with alloca.
      output_buffers = {b.CreateAlloca(
          jit_context.orc_jit().GetTypeConverter().ConvertToLlvmType(
              node->GetType()))};
    }
    value_buffers[node] = output_buffers.front();

    // Create the function which computes the node value.
    XLS_ASSIGN_OR_RETURN(
        NodeFunction node_function,
        CreateNodeFunction(node, output_buffers.size(), jit_context));

    // Call the node function.
    std::vector<llvm::Value*> args;
    args.insert(args.end(), operand_buffers.begin(), operand_buffers.end());
    args.insert(args.end(), output_buffers.begin(), output_buffers.end());
    if (node_function.has_metadata_args) {
      // The node function requires the top-level metadata arguments. This
      // occurs, for example, if the node is invoking another function.
      args.push_back(wrapper.GetInputsArg());
      args.push_back(wrapper.GetOutputsArg());
      args.push_back(wrapper.GetTempBufferArg());
      args.push_back(wrapper.GetInterpreterEventsArg());
      args.push_back(wrapper.GetUserDataArg());
      args.push_back(wrapper.GetJitRuntimeArg());
    }
    llvm::CallInst* node_blocked = b.CreateCall(node_function.function, args);

    if (partition.continuation_point.has_value()) {
      XLS_RET_CHECK_EQ(partition.nodes.size(), 1);
      interrupt_execution = node_blocked;
    }
  }
  // Return false to indicate that execution of the FunctionBase should not be
  // interrupted.
  b.CreateRet(interrupt_execution == nullptr ? b.getFalse()
                                             : interrupt_execution);

  XLS_VLOG(3) << "Partition function:";
  XLS_VLOG(3) << DumpLlvmObjectToString(*wrapper.function());
  return wrapper.function();
}

// Allocates the temporary buffers for nodes in the given partitions. A node
// needs a temporary buffer iff it is not an input or output node and its value
// is used in more than one partition.
absl::Status AllocateTempBuffers(absl::Span<const Partition> partitions,
                                 const LlvmFunctionWrapper& wrapper,
                                 TempBufferAllocator& allocator) {
  for (const Partition& partition : partitions) {
    absl::flat_hash_set<Node*> partition_set(partition.nodes.begin(),
                                             partition.nodes.end());
    for (Node* node : partition.nodes) {
      if (wrapper.IsInputNode(node) || wrapper.IsOutputNode(node)) {
        continue;
      }
      if (!std::all_of(node->users().begin(), node->users().end(),
                       [&](Node* u) { return partition_set.contains(u); })) {
        allocator.Allocate(node);
      }
    }
  }
  return absl::OkStatus();
}

// Returns the nodes which comprise the inputs to a jitted function implementing
// `function_base`. These nodes are passed in via the `inputs` argument.
std::vector<Node*> GetJittedFunctionInputs(FunctionBase* function_base) {
  std::vector<Node*> inputs(function_base->params().begin(),
                            function_base->params().end());
  return inputs;
}

// Returns the nodes whose values are passed out of a jitted function. Buffers
// to hold these node values are passed in via the `outputs` argument.
std::vector<Node*> GetJittedFunctionOutputs(FunctionBase* function_base) {
  if (function_base->IsFunction()) {
    // The output of a function is its return value.
    Function* f = function_base->AsFunctionOrDie();
    return {f->return_value()};
  }
  XLS_CHECK(function_base->IsProc());
  // The outputs of a proc are the next state values.
  Proc* proc = function_base->AsProcOrDie();
  std::vector<Node*> outputs;
  outputs.push_back(proc->NextToken());
  outputs.insert(outputs.end(), proc->NextState().begin(),
                 proc->NextState().end());
  return outputs;
}

// Build an llvm::Function implementing the given FunctionBase. The jitted
// function contains a sequence of calls to partition functions where each
// partition only implements a subset of the FunctionBase's nodes. This
// partitioning is performed to avoid slow compile time in LLVM due to function
// size scaling issues. Continuation points may be added to handle interruption
// of execution (for example, for blocked receives). The jitted function
// implementing `f` might look like:
//
//    int64_t
//    __f(const uint8_t* const* inputs,
//        uint8_t* const* outputs,
//        void* temp_buffer,
//        InterpreterEvents* events,
//        void* user_data,
//        JitRuntime* jit_runtime,
//        int64_t continuation_point) {
//     entry:
//      switch i64 %continuation_point, label %start [
//        i64 1, label %continuation_1
//        i64 2, label %continuation_2
//        ...
//      ]
//
//     start:
//      __f_partition_0(inputs, outputs, temp_buffer,
//                      events, user_data, jit_runtime)
//      __f_partition_1(inputs, outputs, temp_buffer,
//                      events, user_data, jit_runtime)
//      ...
//     continuation_1:
//      __f_partition_n(inputs, outputs, temp_buffer,
//                      events, user_data, jit_runtime)
//      ...
//      return 0;
//   }
//
// If `unpoison_outputs` is true then when built with sanizers enabled, the
// output buffers of the function will be unpoisoned to avoid uninitialized
// memory use errors.
struct PartitionedFunction {
  llvm::Function* function;
  std::vector<Partition> partitions;
};
absl::StatusOr<PartitionedFunction> BuildFunctionInternal(
    FunctionBase* xls_function, TempBufferAllocator& allocator,
    JitBuilderContext& jit_context, bool unpoison_outputs) {
  XLS_VLOG(4) << "BuildFunction:";
  XLS_VLOG(4) << xls_function->DumpIr();
  std::vector<Partition> partitions = PartitionFunctionBase(xls_function);

  std::vector<Node*> inputs = GetJittedFunctionInputs(xls_function);
  std::vector<Node*> outputs = GetJittedFunctionOutputs(xls_function);
  LlvmFunctionWrapper wrapper = LlvmFunctionWrapper::Create(
      xls_function->name(), inputs, outputs,
      llvm::Type::getInt64Ty(jit_context.context()), jit_context,
      LlvmFunctionWrapper::FunctionArg{
          .name = "continuation_point",
          .type = llvm::Type::getInt64Ty(jit_context.context())});

  XLS_RETURN_IF_ERROR(AllocateTempBuffers(partitions, wrapper, allocator));

  std::vector<llvm::Function*> partition_functions;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    std::string name =
        absl::StrFormat("__%s_partition_%d", xls_function->name(), i);
    XLS_ASSIGN_OR_RETURN(
        llvm::Function * partition_function,
        BuildPartitionFunction(name, partitions[i], inputs, outputs, allocator,
                               jit_context));
    partition_functions.push_back(partition_function);
  }

  // Args passed to each partition function.
  std::vector<llvm::Value*> args = {
      wrapper.GetInputsArg(),     wrapper.GetOutputsArg(),
      wrapper.GetTempBufferArg(), wrapper.GetInterpreterEventsArg(),
      wrapper.GetUserDataArg(),   wrapper.GetJitRuntimeArg()};

  // To handle continuation points, sequential basic blocks are created in the
  // body.
  llvm::BasicBlock* start_block = llvm::BasicBlock::Create(
      jit_context.context(), "start", wrapper.function(),
      /*InsertBefore=*/nullptr);
  auto builder = std::make_unique<llvm::IRBuilder<>>(start_block);

  std::vector<llvm::BasicBlock*> continuation_blocks;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    if (partitions[i].continuation_point.has_value()) {
      // Partition is a continuation point, create a new basic block which can
      // be the target of a branch.
      llvm::BasicBlock* continuation_block = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("continuation_%d",
                          partitions[i].continuation_point.value()),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      continuation_blocks.push_back(continuation_block);
      builder->CreateBr(continuation_block);
      builder = std::make_unique<llvm::IRBuilder<>>(continuation_block);
    }

    llvm::Function* partition_function = partition_functions[i];
    llvm::CallInst* interrupt_execution = builder->CreateCall(
        partition_function->getFunctionType(), partition_function, args);

    if (partitions[i].continuation_point.has_value()) {
      // Execution may also exit at continuation points. Check for early exit
      // (partition function returns true).
      llvm::BasicBlock* early_return = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("continuation_%d_early_return",
                          partitions[i].continuation_point.value()),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      llvm::IRBuilder<> early_return_builder(early_return);
      early_return_builder.CreateRet(early_return_builder.getInt64(
          partitions[i].continuation_point.value()));

      llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("continuation_%d_continue",
                          partitions[i].continuation_point.value()),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      builder->CreateCondBr(interrupt_execution, early_return, continue_block);
      builder = std::make_unique<llvm::IRBuilder<>>(continue_block);
    }
  }

  if (unpoison_outputs) {
    for (Node* output : outputs) {
      int64_t index = wrapper.GetOutputArgIndices(output).front();
      llvm::Value* output_buffer = LoadPointerFromPointerArray(
          index, wrapper.GetOutputsArg(), builder.get());
      UnpoisonBuffer(output_buffer,
                     jit_context.orc_jit().GetTypeConverter().GetTypeByteSize(
                         output->GetType()),
                     builder.get());
    }
  }
  // Return zero indicating that the execution of the FunctionBase completed.
  builder->CreateRet(builder->getInt64(0));

  if (continuation_blocks.empty()) {
    // Just jump from the entry block to the first block.
    wrapper.entry_builder().CreateBr(start_block);
  } else {
    // Add a switch at the entry block to jump to the appropriate continuation
    // point.
    llvm::SwitchInst* swtch = wrapper.entry_builder().CreateSwitch(
        wrapper.GetExtraArg().value(), start_block);
    for (int64_t i = 0; i < continuation_blocks.size(); ++i) {
      swtch->addCase(wrapper.entry_builder().getInt64(i + 1),
                     continuation_blocks[i]);
    }
  }
  return PartitionedFunction{.function = wrapper.function(),
                             .partitions = std::move(partitions)};
}

// Unpacks the packed bit vector representation in `packed_value` into a
// LLVM native data layout.
absl::StatusOr<llvm::Value*> UnpackValue(
    Type* param_type, llvm::Value* packed_value,
    const LlvmTypeConverter& type_converter, llvm::IRBuilder<>* builder) {
  switch (param_type->kind()) {
    case TypeKind::kBits:
      return builder->CreateZExt(
          builder->CreateTrunc(
              packed_value, type_converter.ConvertToPackedLlvmType(param_type)),
          type_converter.ConvertToLlvmType(param_type));
    case TypeKind::kArray: {
      // Create an empty array and plop in every element.
      ArrayType* array_type = param_type->AsArrayOrDie();
      Type* element_type = array_type->element_type();

      llvm::Value* array = LlvmTypeConverter::ZeroOfType(
          type_converter.ConvertToLlvmType(array_type));
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            llvm::Value * element,
            UnpackValue(element_type, packed_value, type_converter, builder));
        array = builder->CreateInsertValue(array, element, {i});
        packed_value =
            builder->CreateLShr(packed_value, element_type->GetFlatBitCount());
      }
      return array;
    }
    case TypeKind::kTuple: {
      // Create an empty tuple and plop in every element.
      TupleType* tuple_type = param_type->AsTupleOrDie();
      llvm::Value* tuple = LlvmTypeConverter::ZeroOfType(
          type_converter.ConvertToLlvmType(tuple_type));
      for (int32_t i = tuple_type->size() - 1; i >= 0; i--) {
        // Tuple elements are stored MSB -> LSB, so we need to extract in
        // reverse order to match native layout.
        Type* element_type = tuple_type->element_type(i);
        XLS_ASSIGN_OR_RETURN(
            llvm::Value * element,
            UnpackValue(element_type, packed_value, type_converter, builder));
        tuple = builder->CreateInsertValue(tuple, element,
                                           {static_cast<uint32_t>(i)});
        packed_value =
            builder->CreateLShr(packed_value, element_type->GetFlatBitCount());
      }
      return tuple;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled type kind: ", TypeKindToString(param_type->kind())));
  }
}

// Load the `index`-th packed argument from the given arg array. The argument is
// unpacked into the LLVM native data layout and returned.
absl::StatusOr<llvm::Value*> LoadAndUnpackArgument(
    int64_t arg_index, Type* xls_type, llvm::Value* arg_array,
    const LlvmTypeConverter& type_converter, llvm::IRBuilder<>* builder) {
  if (xls_type->GetFlatBitCount() == 0) {
    // Create an empty structure, etc.
    return LlvmTypeConverter::ZeroOfType(
        type_converter.ConvertToLlvmType(xls_type));
  }

  llvm::Type* packed_type = type_converter.ConvertToPackedLlvmType(xls_type);
  llvm::Value* packed_value =
      LoadFromPointerArray(arg_index, packed_type, arg_array, builder);

  // Now populate an Value of Param's type with the packed buffer contents.
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * unpacked_param,
      UnpackValue(xls_type, packed_value, type_converter, builder));
  return unpacked_param;
}

// Recursive helper for packing LLVM values representing structured XLS types
// into a flat bit vector.
absl::StatusOr<llvm::Value*> PackValueHelper(llvm::Value* element,
                                             Type* element_type,
                                             llvm::Value* buffer,
                                             int64_t bit_offset,
                                             llvm::IRBuilder<>* builder) {
  switch (element_type->kind()) {
    case TypeKind::kBits:
      if (element->getType() != buffer->getType()) {
        if (element->getType()->getIntegerBitWidth() >
            buffer->getType()->getIntegerBitWidth()) {
          // The LLVM type of the subelement is wider than the packed value of
          // the entire type. This can happen because bits types are padded up
          // to powers of two.
          element = builder->CreateTrunc(element, buffer->getType());
        } else {
          element = builder->CreateZExt(element, buffer->getType());
        }
      }
      element = builder->CreateShl(element, bit_offset);
      return builder->CreateOr(buffer, element);
    case TypeKind::kArray: {
      ArrayType* array_type = element_type->AsArrayOrDie();
      Type* array_element_type = array_type->element_type();
      for (uint32_t i = 0; i < array_type->size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            buffer,
            PackValueHelper(
                builder->CreateExtractValue(element, {i}), array_element_type,
                buffer, bit_offset + i * array_element_type->GetFlatBitCount(),
                builder));
      }
      return buffer;
    }
    case TypeKind::kTuple: {
      // Reverse tuple packing order to match native layout.
      TupleType* tuple_type = element_type->AsTupleOrDie();
      for (int64_t i = tuple_type->size() - 1; i >= 0; i--) {
        XLS_ASSIGN_OR_RETURN(
            buffer, PackValueHelper(builder->CreateExtractValue(
                                        element, {static_cast<uint32_t>(i)}),
                                    tuple_type->element_type(i), buffer,
                                    bit_offset, builder));
        bit_offset += tuple_type->element_type(i)->GetFlatBitCount();
      }
      return buffer;
    }
    case TypeKind::kToken: {
      // Tokens are zero-bit constructs, so there's nothing to do!
      return buffer;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled element kind: ", TypeKindToString(element_type->kind())));
  }
}

// Packs the given `value` in LLVM native data layout for XLS type
// `xls_type` into a flat bit vector and returns it.
absl::StatusOr<llvm::Value*> PackValue(llvm::Value* value, Type* xls_type,
                                       const LlvmTypeConverter& type_converter,
                                       llvm::IRBuilder<>* builder) {
  llvm::Value* packed_buffer = llvm::ConstantInt::get(
      type_converter.ConvertToPackedLlvmType(xls_type), 0);
  return PackValueHelper(value, xls_type, packed_buffer, 0, builder);
}

// Builds a wrapper around the jitted function `callee` which accepts inputs and
// produces outputs in a packed data layout.
absl::StatusOr<llvm::Function*> BuildPackedWrapper(
    FunctionBase* xls_function, llvm::Function* callee,
    JitBuilderContext& jit_context) {
  llvm::LLVMContext* context = &jit_context.context();
  std::vector<Node*> inputs = GetJittedFunctionInputs(xls_function);
  std::vector<Node*> outputs = GetJittedFunctionOutputs(xls_function);
  LlvmFunctionWrapper wrapper = LlvmFunctionWrapper::Create(
      absl::StrFormat("%s_packed", xls_function->name()),
      GetJittedFunctionInputs(xls_function),
      GetJittedFunctionOutputs(xls_function), llvm::Type::getInt64Ty(*context),
      jit_context,
      LlvmFunctionWrapper::FunctionArg{
          .name = "continuation_point",
          .type = llvm::Type::getInt64Ty(*context)});

  // First load and unpack the arguments then store them in LLVM native data
  // layout. These unpacked values are pointed to by an array of pointers passed
  // on to the wrapped function.
  llvm::Value* input_arg_array = wrapper.entry_builder().CreateAlloca(
      llvm::ArrayType::get(llvm::PointerType::get(*context, 0), inputs.size()));
  llvm::Type* pointer_array_type =
      llvm::ArrayType::get(llvm::Type::getInt8PtrTy(*context), 0);
  for (int64_t i = 0; i < inputs.size(); ++i) {
    Node* input = inputs[i];
    llvm::Value* input_buffer = wrapper.entry_builder().CreateAlloca(
        jit_context.orc_jit().GetTypeConverter().ConvertToLlvmType(
            input->GetType()));
    llvm::Value* gep = wrapper.entry_builder().CreateGEP(
        pointer_array_type, input_arg_array,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), i),
        });
    wrapper.entry_builder().CreateStore(input_buffer, gep);

    XLS_ASSIGN_OR_RETURN(
        llvm::Value * unpacked_arg,
        LoadAndUnpackArgument(i, input->GetType(), wrapper.GetInputsArg(),
                              jit_context.orc_jit().GetTypeConverter(),
                              &wrapper.entry_builder()));
    wrapper.entry_builder().CreateStore(unpacked_arg, input_buffer);
  }

  llvm::Value* output_arg_array =
      wrapper.entry_builder().CreateAlloca(llvm::ArrayType::get(
          llvm::PointerType::get(*context, 0), outputs.size()));
  for (int64_t i = 0; i < outputs.size(); ++i) {
    Node* output = outputs[i];
    llvm::Value* output_buffer = wrapper.entry_builder().CreateAlloca(
        jit_context.orc_jit().GetTypeConverter().ConvertToLlvmType(
            output->GetType()));
    llvm::Value* gep = wrapper.entry_builder().CreateGEP(
        pointer_array_type, output_arg_array,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), i),
        });
    wrapper.entry_builder().CreateStore(output_buffer, gep);
  }

  std::vector<llvm::Value*> args;
  args.push_back(input_arg_array);
  args.push_back(output_arg_array);
  args.push_back(wrapper.GetTempBufferArg());
  args.push_back(wrapper.GetInterpreterEventsArg());
  args.push_back(wrapper.GetUserDataArg());
  args.push_back(wrapper.GetJitRuntimeArg());
  args.push_back(wrapper.GetExtraArg().value());

  llvm::Value* continuation_result =
      wrapper.entry_builder().CreateCall(callee, args);

  // After returning, pack the value into the return value buffer.
  for (int64_t i = 0; i < outputs.size(); ++i) {
    Node* output = outputs[i];

    // Declare the return argument as an iX, and pack the actual data as such
    // an integer.
    llvm::Value* output_value = LoadFromPointerArray(
        i,
        jit_context.orc_jit().GetTypeConverter().ConvertToLlvmType(
            output->GetType()),
        output_arg_array, &wrapper.entry_builder());

    XLS_ASSIGN_OR_RETURN(llvm::Value * packed_output,
                         PackValue(output_value, output->GetType(),
                                   jit_context.orc_jit().GetTypeConverter(),
                                   &wrapper.entry_builder()));
    llvm::Value* packed_output_buffer = LoadPointerFromPointerArray(
        i, wrapper.GetOutputsArg(), &wrapper.entry_builder());
    wrapper.entry_builder().CreateStore(packed_output, packed_output_buffer);

    UnpoisonBuffer(packed_output_buffer,
                   jit_context.orc_jit().GetTypeConverter().GetTypeByteSize(
                       output->GetType()),
                   &wrapper.entry_builder());
  }

  // Return value of zero means that the functoinbase completed execution.
  wrapper.entry_builder().CreateRet(continuation_result);

  return wrapper.function();
}

// Jits a function implementing `xls_function`. Also jits all transitively
// dependent xls::Functions which may be called by `xls_function`.
absl::StatusOr<JittedFunctionBase> BuildFunctionAndDependencies(
    FunctionBase* xls_function, JitBuilderContext& jit_context) {
  std::vector<FunctionBase*> functions = GetDependentFunctions(xls_function);
  TempBufferAllocator allocator(&jit_context.orc_jit().GetTypeConverter());
  llvm::Function* top_function = nullptr;
  std::vector<Partition> top_partitions;
  for (FunctionBase* f : functions) {
    XLS_ASSIGN_OR_RETURN(
        PartitionedFunction partitioned_function,
        BuildFunctionInternal(f, allocator, jit_context,
                              /*unpoison_outputs=*/f == xls_function));
    jit_context.SetLlvmFunction(f, partitioned_function.function);
    if (f == xls_function) {
      top_function = partitioned_function.function;
      top_partitions = std::move(partitioned_function.partitions);
    }
  }
  XLS_RET_CHECK(top_function != nullptr);

  std::string function_name = top_function->getName().str();
  XLS_ASSIGN_OR_RETURN(
      llvm::Function * packed_wrapper_function,
      BuildPackedWrapper(xls_function, top_function, jit_context));
  std::string packed_wrapper_name = packed_wrapper_function->getName().str();

  XLS_RETURN_IF_ERROR(
      jit_context.orc_jit().CompileModule(jit_context.ConsumeModule()));

  JittedFunctionBase jitted_function;

  jitted_function.function_name = function_name;
  XLS_ASSIGN_OR_RETURN(auto fn_address,
                       jit_context.orc_jit().LoadSymbol(function_name));
  jitted_function.function = absl::bit_cast<JitFunctionType>(fn_address);

  jitted_function.packed_function_name = packed_wrapper_name;
  XLS_ASSIGN_OR_RETURN(auto packed_fn_address,
                       jit_context.orc_jit().LoadSymbol(packed_wrapper_name));
  jitted_function.packed_function =
      absl::bit_cast<JitFunctionType>(packed_fn_address);

  for (const Node* input : GetJittedFunctionInputs(xls_function)) {
    jitted_function.input_buffer_sizes.push_back(
        jit_context.orc_jit().GetTypeConverter().GetTypeByteSize(
            input->GetType()));
  }
  for (const Node* output : GetJittedFunctionOutputs(xls_function)) {
    jitted_function.output_buffer_sizes.push_back(
        jit_context.orc_jit().GetTypeConverter().GetTypeByteSize(
            output->GetType()));
  }
  jitted_function.temp_buffer_size = allocator.size();

  // Indicate which nodes correspond to which continuation points.
  for (const Partition& partition : top_partitions) {
    if (partition.continuation_point.has_value()) {
      XLS_RET_CHECK_EQ(partition.nodes.size(), 1);
      jitted_function
          .continuation_points[partition.continuation_point.value()] =
          partition.nodes.front();
    }
  }

  return std::move(jitted_function);
}

}  // namespace

absl::StatusOr<JittedFunctionBase> BuildFunction(Function* xls_function,
                                                 OrcJit& orc_jit) {
  JitBuilderContext jit_context(orc_jit);
  return BuildFunctionAndDependencies(xls_function, jit_context);
}

absl::StatusOr<JittedFunctionBase> BuildProcFunction(
    Proc* proc, JitChannelQueueManager* queue_mgr, OrcJit& orc_jit) {
  JitBuilderContext jit_context(orc_jit, queue_mgr);
  return BuildFunctionAndDependencies(proc, jit_context);
}

}  // namespace xls
