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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/Function.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/IR/Value.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/events.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {
namespace {

// A fake function entrypoint we can use if we didn't actually load the compiled
// code.
int64_t InvalidJitFunctionUse(const uint8_t* const* inputs,
                              uint8_t* const* outputs, void* temp_buffer,
                              InterpreterEvents* events,
                              InstanceContext* instance_context,
                              JitRuntime* jit_runtime,
                              int64_t continuation_point) {
  static_assert(
      std::is_same_v<decltype(&InvalidJitFunctionUse), JitFunctionType>);
  LOG(FATAL)
      << "Attempt to call invalid function pointer in JitObjectCode structure!";
}

// Loads a pointer from the `index`-th slot in the array pointed to by
// `pointer_array`.
llvm::Value* LoadPointerFromPointerArray(int64_t index,
                                         llvm::Value* pointer_array,
                                         llvm::IRBuilder<>* builder) {
  llvm::LLVMContext& context = builder->getContext();
  llvm::Type* pointer_type = llvm::PointerType::getUnqual(context);
  llvm::Value* gep = builder->CreateGEP(
      pointer_type, pointer_array,
      {
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), index),
      });

  return builder->CreateLoad(llvm::PointerType::get(context, 0), gep);
}

// Abstraction around an LLVM function of type `JitFunctionType`. Signature of
// these functions:
//
//  ReturnT f(const uint8_t* const* inputs,
//            uint8_t* const* outputs, void* temp_buffer,
//            InterpreterEvents* events, InstanceContext* instance_context,
//            JitRuntime* jit_runtime)
//
// This type of function is used for the jitted functions implementing XLS
// functions, procs, etc, as well as for the partition functions called from
// within the jitted functions.
//
// `input_args` are the Nodes whose values are passed in the `inputs` function
// argument. `output_args` are Nodes whose values are written out to buffers
// indicated by the `outputs` function argument.
class LlvmFunctionWrapper final : public JitCompilationMetadata {
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
    CHECK_EQ(jit_context.module()->getFunction(name), nullptr)
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
    fn->getArg(4)->setName("instance_context");
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
  bool IsInputNode(Node* node) const final {
    return input_indices_.contains(node);
  }

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

  absl::StatusOr<llvm::Value*> GetInputBufferFrom(
      Node* node, llvm::Value* base_ptr,
      llvm::IRBuilder<>& builder) const final {
    XLS_RET_CHECK(IsInputNode(node));
    return LoadPointerFromPointerArray(GetInputArgIndex(node), base_ptr,
                                       &builder);
  }

  // Returns whether `node` is one of the nodes whose value should be written to
  // one of the output buffers passed in via the `inputs` function argument.
  bool IsOutputNode(Node* node) const { return output_indices_.contains(node); }
  bool IsOutputPortOrRegister(Node* node) const {
    return IsOutputNode(node) &&
           (node->Is<OutputPort>() || node->Is<RegisterWrite>());
  }

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
  llvm::Value* GetInstanceContextArg() const { return fn_->getArg(4); }
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
      CHECK(!input_indices_.contains(input_args[i]));
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

// The kinds of allocations assigned to xls::Nodes by BufferAllocator.
enum class AllocationKind : uint8_t {
  // The node should be allocated a buffer in the temp block. The temp block is
  // passed in to the top-level JITted functions so temp buffers persist across
  // partitions and continuation points.
  kTempBlock,

  // The node should be allocated a buffer using alloca. These buffers are
  // scoped within a partition function.
  kAlloca,

  // The node needs no buffer to be allocated. Examples include Params which are
  // passed in via already-allocated input buffers and some Bits-typed literals
  // which are materialized as LLVM constants at their uses.
  kNone,
};

// Allocator for the buffers used to hold xls::Node values within jitted
// functions.
class BufferAllocator {
 public:
  explicit BufferAllocator(LlvmTypeConverter* type_converter)
      : type_converter_(type_converter) {}

  void SetAllocationKind(Node* node, AllocationKind kind) {
    CHECK(!allocation_kinds_.contains(node));
    allocation_kinds_[node] = kind;
    if (kind == AllocationKind::kTempBlock) {
      AllocateTempBuffer(node);
    }
  }

  AllocationKind GetAllocationKind(Node* node) const {
    return allocation_kinds_.at(node);
  }

  // Returns the offset within the temp block for the buffer allocated for
  // `node`. Node must be assigned allocation kind kTempblock.
  int64_t GetOffset(Node* node) const {
    CHECK(allocation_kinds_.at(node) == AllocationKind::kTempBlock);
    return temp_block_offsets_.at(node);
  }

  // Returns the total size of the allocated memory.
  int64_t size() const { return current_offset_; }
  int64_t alignment() const { return alignment_; }

 private:
  void AllocateTempBuffer(Node* node) {
    CHECK(!node->Is<RegisterWrite>());
    CHECK(!node->Is<OutputPort>());
    CHECK(!temp_block_offsets_.contains(node));
    int64_t offset =
        type_converter_->AlignFor(node->GetType(), current_offset_);
    int64_t node_size = type_converter_->GetTypeByteSize(node->GetType());
    temp_block_offsets_[node] = offset;
    alignment_ =
        std::max(alignment_,
                 type_converter_->GetTypePreferredAlignment(node->GetType()));
    current_offset_ = offset + node_size;
    VLOG(3) << absl::StreamFormat(
        "Allocated %s at offset %d (size = %d): total size %d", node->GetName(),
        offset, node_size, current_offset_);
  }

  LlvmTypeConverter* type_converter_;
  absl::flat_hash_map<Node*, int64_t> temp_block_offsets_;
  int64_t current_offset_ = 0;
  int64_t alignment_ = 1;
  absl::flat_hash_map<Node*, AllocationKind> allocation_kinds_;
};

// The maximum number of xls::Nodes in a partition.
static constexpr int64_t kMaxPartitionSize = 100;

// Abstraction representing a point (partition function) at which an early exit
// can occur.
struct EarlyExitPoint {
  // A unique identifier for this early exit point. ID's are numbered starting
  // at 1 (0 is a special value).
  int64_t id;
  // The ID of the resume point (partition) at which execution should continue
  // when execution is restarted after the early exit.
  int64_t resume_point_id;
};

// Abstraction representing a point (partition function) at which execution can
// resume after an early exit.
struct ResumePoint {
  // A unique identifier for this resume point. ID's are numbered starting
  // at 0.
  int64_t id;
};

// Data structure holding a set of nodes which should be emitted together in a
// single LLVM function.
struct Partition {
  // Whether this partition is a point at which execution of the FunctionBase
  // can be interrupted and resumed.
  std::optional<EarlyExitPoint> early_exit_point;
  std::optional<ResumePoint> resume_point;
  std::vector<Node*> nodes;
};

bool IsEarlyExitPoint(Node* node) {
  return (node->Is<Receive>() && node->As<Receive>()->is_blocking()) ||
         node->Is<Send>();
}

// For early exit point node `node`, returns whether the execution should
// continue after `node` when execution resumes. If false, then execution
// continues before `node`.
bool ExecutionContinuesAfterNode(Node* node) {
  CHECK(IsEarlyExitPoint(node));
  return node->Is<Send>();
}

// Divides the nodes of the given function base into a topologically sorted
// sequence of partitions. Each partition then becomes a separate function in
// the LLVM module.
// TODO(meheff): 2022/09/09 Tune this algorithm to maximize performance.
std::vector<Partition> PartitionFunctionBase(FunctionBase* f) {
  absl::flat_hash_map<Node*, int64_t> partition_map;
  // Partitions at which execution may resume after an early exit.
  absl::flat_hash_set<int64_t> resume_partitions;
  // Partitions at which execution may exit early. Value is whether execution
  // resumes after or at the point where execution broke.
  enum class Resume : uint8_t { kNextPartition, kThisPartition };
  absl::flat_hash_map<int64_t, Resume> early_exit_partitions;

  // Naively assign nodes to partitions based on a topological sort. First N
  // nodes go to partition 0, next N nodes go to partition 1, etc.
  int64_t current_partition = 0;
  int64_t partition_size = 0;
  for (Node* node : TopoSort(f)) {
    if (IsEarlyExitPoint(node)) {
      CHECK(f->IsProc())
          << "Early exit points are only supported in procs in the JIT";
      // Nodes which are early exits are placed in their own partition.
      if (partition_size != 0) {
        ++current_partition;
      }
      partition_map[node] = current_partition;
      if (ExecutionContinuesAfterNode(node)) {
        early_exit_partitions.insert(
            {current_partition, Resume::kNextPartition});
        resume_partitions.insert(current_partition + 1);
      } else {
        early_exit_partitions.insert(
            {current_partition, Resume::kThisPartition});
        resume_partitions.insert(current_partition);
      }
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
      // Avoid putting literals in early exit partitions. These partitions
      // should only have a single node.
      if (min_partition != std::numeric_limits<int64_t>::max() &&
          !early_exit_partitions.contains(min_partition)) {
        partition_map[node] = min_partition;
      }
    }
  }

  // Assemble nodes into partitions. `current_partition` is the maximum number
  // of any partition.
  std::vector<Partition> partitions(current_partition + 1);
  for (Node* node : TopoSort(f)) {
    int64_t partition = partition_map.at(node);
    partitions.at(partition).nodes.push_back(node);
  }

  // Set up partitions which may be resume points.
  int64_t resume_point_id = 0;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    if (resume_partitions.contains(i)) {
      partitions[i].resume_point = ResumePoint{.id = resume_point_id};
      ++resume_point_id;
    }
  }

  // Set up partitions which may be early exit points. Number early exit points
  // starting at one because zero has special meaning (beginning of proc).
  int64_t early_exit_point_id = 1;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    if (early_exit_partitions.contains(i)) {
      partitions[i].early_exit_point = EarlyExitPoint{
          .id = early_exit_point_id,
          .resume_point_id =
              early_exit_partitions.at(i) == Resume::kNextPartition
                  ? partitions.at(i + 1).resume_point->id
                  : partitions.at(i).resume_point->id};
      ++early_exit_point_id;
    }
  }

  return partitions;
}

// Get the type this node reserves as input.
Type* InputType(const Node* node) { return node->GetType(); }

// Get the type this node outputs into the functions return value.
Type* OutputType(const Node* node) {
  if (node->Is<RegisterWrite>() || node->Is<OutputPort>()) {
    // Operand 0 is the data we write to the register/port
    return node->operand(0)->GetType();
  }
  return node->GetType();
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
//                   InstanceContext* instance_context,
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
// early exit points (e.g., contain a blocking receive).
//
// `global_input_nodes` and `global_output_nodes` are the set of nodes whose
// buffers are passed in via the `input`/`output` arguments of the function.
absl::StatusOr<llvm::Function*> BuildPartitionFunction(
    std::string_view name, const Partition& partition,
    absl::Span<Node* const> global_input_nodes,
    absl::Span<Node* const> global_output_nodes,
    const BufferAllocator& allocator, JitBuilderContext& jit_context) {
  LlvmFunctionWrapper wrapper = LlvmFunctionWrapper::Create(
      name, global_input_nodes, global_output_nodes,
      llvm::Type::getInt1Ty(jit_context.context()), jit_context);
  llvm::IRBuilder<>& b = wrapper.entry_builder();

  // Whether to interrupt execution of the FunctionBase. Only used for
  // partitions which are early exit points (e.g., have a blocking receive).
  llvm::Value* interrupt_execution = nullptr;

  // The pointers to the buffers of nodes in the partition.
  absl::flat_hash_map<Node*, llvm::Value*> value_buffers;
  for (Node* node : partition.nodes) {
    if (wrapper.IsInputNode(node)) {
      // Node is an input node. We need to generate a node function for  this
      // node to call callbacks on the node.
      XLS_RET_CHECK(allocator.GetAllocationKind(node) == AllocationKind::kNone);
      if (wrapper.IsOutputNode(node)) {
        // `node` is also an output node. This can occur, for example, if a
        // state param is the next state value for a proc.
        llvm::Value* input_buffer = wrapper.GetInputBuffer(node, b);
        for (llvm::Value* output_buffer : wrapper.GetOutputBuffers(node, b)) {
          LlvmMemcpy(
              output_buffer, input_buffer,
              jit_context.type_converter().GetTypeByteSize(OutputType(node)),
              b);
        }
      }
    }

    // Gather the buffers to which the value of `node` must be written.
    std::vector<llvm::Value*> output_buffers;
    if (node->Is<Next>()) {
      // next_value nodes store their output in the state-read's location, and
      // return nothing themselves.
      StateRead* state_read = node->As<Next>()->state_read()->As<StateRead>();
      XLS_RET_CHECK(allocator.GetAllocationKind(state_read) ==
                    AllocationKind::kNone);
      output_buffers = wrapper.GetOutputBuffers(state_read, b);
    } else if (wrapper.IsInputNode(node)) {
      XLS_RET_CHECK(allocator.GetAllocationKind(node) == AllocationKind::kNone);
      output_buffers = {};
    } else if (wrapper.IsOutputNode(node)) {
      XLS_RET_CHECK(allocator.GetAllocationKind(node) == AllocationKind::kNone);
      output_buffers = wrapper.GetOutputBuffers(node, b);
    } else if (allocator.GetAllocationKind(node) ==
               AllocationKind::kTempBlock) {
      XLS_RET_CHECK(!node->Is<RegisterWrite>());
      XLS_RET_CHECK(!node->Is<OutputPort>());
      output_buffers = {
          wrapper.GetOffsetIntoTempBuffer(allocator.GetOffset(node), b)};
    } else if (allocator.GetAllocationKind(node) == AllocationKind::kAlloca) {
      // `node` is used exclusively inside this partition (not an input, output,
      // nor has a temp buffer). Allocate a buffer on the stack with alloca.
      XLS_RET_CHECK(!node->Is<RegisterWrite>());
      XLS_RET_CHECK(!node->Is<OutputPort>());
      output_buffers = {b.CreateAlloca(
          jit_context.type_converter().ConvertToLlvmType(node->GetType()))};
    } else {
      // Node has no allocation and is not an output buffer. Nothing to emit for
      // this node.
      XLS_RET_CHECK(allocator.GetAllocationKind(node) == AllocationKind::kNone);
      XLS_RET_CHECK(!OpIsSideEffecting(node->op()));
      continue;
    }

    if (!output_buffers.empty()) {
      value_buffers[node] = output_buffers.front();
    }

    // Create the function which computes the node value.
    XLS_ASSIGN_OR_RETURN(
        NodeFunction node_function,
        CreateNodeFunction(node, output_buffers.size(), wrapper, jit_context));

    // Gather the operand values to be passed to the node function.
    std::vector<llvm::Value*> operand_buffers;
    for (Node* operand : node_function.operand_arguments) {
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
        // `operand` is stored inside the temporary buffer.
        XLS_RET_CHECK(allocator.GetAllocationKind(operand) ==
                      AllocationKind::kTempBlock)
            << operand;
        arg = wrapper.GetOffsetIntoTempBuffer(allocator.GetOffset(operand), b);
      }
      operand_buffers.push_back(arg);
      value_buffers[operand] = arg;
    }

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
      args.push_back(wrapper.GetInstanceContextArg());
      args.push_back(wrapper.GetJitRuntimeArg());
    } else {
      // The instance context is always passed so the node can upcall.
      args.push_back(wrapper.GetInstanceContextArg());
    }
    XLS_RET_CHECK_EQ(node_function.function->arg_size(), args.size());
    llvm::CallInst* node_blocked = b.CreateCall(node_function.function, args);

    if (partition.early_exit_point.has_value()) {
      XLS_RET_CHECK_EQ(partition.nodes.size(), 1);
      interrupt_execution = node_blocked;
    }
  }
  // Return false to indicate that execution of the FunctionBase should not be
  // interrupted.
  b.CreateRet(interrupt_execution == nullptr ? b.getFalse()
                                             : interrupt_execution);

  VLOG(3) << "Partition function:";
  VLOG(3) << DumpLlvmObjectToString(*wrapper.function());
  return wrapper.function();
}

// Determine the type of buffers required by each node. Allocates the temporary
// buffers for nodes as needed.
absl::Status AllocateBuffers(absl::Span<const Partition> partitions,
                             const LlvmFunctionWrapper& wrapper,
                             BufferAllocator& allocator) {
  for (const Partition& partition : partitions) {
    absl::flat_hash_set<Node*> partition_set(partition.nodes.begin(),
                                             partition.nodes.end());
    for (Node* node : partition.nodes) {
      if (wrapper.IsInputNode(node) || wrapper.IsOutputNode(node) ||
          ShouldMaterializeAtUse(node)) {
        allocator.SetAllocationKind(node, AllocationKind::kNone);
      } else if (!node->function_base()->HasImplicitUse(node) &&
                 std::all_of(
                     node->users().begin(), node->users().end(),
                     [&](Node* u) { return partition_set.contains(u); })) {
        // All of the uses of node are in the partition.
        allocator.SetAllocationKind(node, AllocationKind::kAlloca);
      } else {
        // Node has a use in another partition.
        allocator.SetAllocationKind(node, AllocationKind::kTempBlock);
      }
    }
  }
  return absl::OkStatus();
}

// Returns the nodes which comprise the inputs to a jitted function implementing
// `function_base`. These nodes are passed in via the `inputs` argument.
std::vector<Node*> GetJittedFunctionInputs(FunctionBase* function_base) {
  if (function_base->IsBlock()) {
    Block* block = function_base->AsBlockOrDie();
    std::vector<Node*> out;
    out.reserve(block->GetInputPorts().size() + block->GetRegisters().size());
    absl::c_copy(block->GetInputPorts(), std::back_inserter(out));
    absl::c_transform(
        block->GetRegisters(), std::back_inserter(out),
        [&](Register* r) -> Node* { return *block->GetRegisterRead(r); });
    return out;
  }
  if (function_base->IsProc()) {
    Proc* proc = function_base->AsProcOrDie();
    std::vector<Node*> out;
    absl::c_transform(proc->StateElements(), std::back_inserter(out),
                      [&](StateElement* st) { return proc->GetStateRead(st); });
    return out;
  }
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
  if (function_base->IsBlock()) {
    // Order of block outputs is:
    //   (1) output ports
    //   (2) First RegisterWrite of each register.
    //   (3) Second, and later RegisterWrites of each register (if any).
    // Multiple RegisterWrites are reconciled at the end of each cycle.
    Block* block = function_base->AsBlockOrDie();
    std::vector<Node*> out;
    out.reserve(block->GetOutputPorts().size() + block->GetRegisters().size());
    absl::c_copy(block->GetOutputPorts(), std::back_inserter(out));
    for (Register* reg : block->GetRegisters()) {
      out.push_back(block->GetRegisterWrites(reg)->front());
    }
    for (Register* reg : block->GetRegisters()) {
      for (RegisterWrite* rw : block->GetRegisterWrites(reg)->subspan(1)) {
        out.push_back(rw);
      }
    }
    return out;
  }
  // The outputs of a proc are the next state values - which will be stored in
  // the memory locations for the state reads.
  Proc* proc = function_base->AsProcOrDie();
  std::vector<Node*> outputs;
  outputs.reserve(proc->StateElements().size());
  absl::c_transform(proc->StateElements(), std::back_inserter(outputs),
                    [&](StateElement* st) { return proc->GetStateRead(st); });
  return outputs;
}

// Build an llvm::Function implementing the given FunctionBase. The jitted
// function contains a sequence of calls to partition functions where each
// partition only implements a subset of the FunctionBase's nodes. This
// partitioning is performed to avoid slow compile time in LLVM due to function
// size scaling issues. Early exit and resume points may be added to handle
// interruption of execution (for example, for blocked receives). The jitted
// function implementing `f` might look like:
//
//    int64_t
//    __f(const uint8_t* const* inputs,
//        uint8_t* const* outputs,
//        void* temp_buffer,
//        InterpreterEvents* events,
//        InstanceContext* instance_context,
//        JitRuntime* jit_runtime,
//        int64_t continuation_point) {
//     entry:
//      switch i64 %continuation_point, label %start [
//        i64 1, label %resume_point_1
//        i64 2, label %resume_point_2
//        ...
//      ]
//
//     start:
//      __f_partition_0(inputs, outputs, temp_buffer,
//                      events, instance_context, jit_runtime)
//      __f_partition_1(inputs, outputs, temp_buffer,
//                      events, instance_context, jit_runtime)
//      ...
//     resume_point_1:
//      __f_partition_n(inputs, outputs, temp_buffer,
//                      events, instance_context, jit_runtime)
//      ...
//     resume_point_n:
//      return 0;
//   }
//
struct PartitionedFunction {
  llvm::Function* function;
  std::vector<Partition> partitions;
};
absl::StatusOr<PartitionedFunction> BuildFunctionInternal(
    FunctionBase* xls_function, BufferAllocator& allocator,
    JitBuilderContext& jit_context) {
  VLOG(4) << "BuildFunction:";
  VLOG(4) << xls_function->DumpIr();
  std::vector<Partition> partitions = PartitionFunctionBase(xls_function);

  // For shared compilations (ie AOT proc-networks) we need to have multiple
  // different procs all hitting the same function. The issue is that we
  // assign slots in the temp buffer globally starting from 0 for each call to
  // JittedFunctionBase::Build. This means that different 'procs' have
  // different ideas about where in the temp buffer things go. For now to
  // avoid this issue simple create a different copy all dependent functions
  // for each top.
  // TODO(allight): Long term it would be good to avoid this headache and
  // extra code. Since the function call graph is a DAG we should be able to
  // have each function assign its own tmp buffer starting from 0 and make the
  // overall tmp-buffer the topo sort.
  std::string base_name = jit_context.MangleFunctionName(xls_function);
  std::vector<Node*> inputs = GetJittedFunctionInputs(xls_function);
  std::vector<Node*> outputs = GetJittedFunctionOutputs(xls_function);
  LlvmFunctionWrapper wrapper = LlvmFunctionWrapper::Create(
      base_name, inputs, outputs, llvm::Type::getInt64Ty(jit_context.context()),
      jit_context,
      LlvmFunctionWrapper::FunctionArg{
          .name = "continuation_point",
          .type = llvm::Type::getInt64Ty(jit_context.context())});

  XLS_RETURN_IF_ERROR(AllocateBuffers(partitions, wrapper, allocator));

  std::vector<llvm::Function*> partition_functions;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    std::string name = absl::StrFormat("__%s_partition_%d", base_name, i);
    XLS_ASSIGN_OR_RETURN(
        llvm::Function * partition_function,
        BuildPartitionFunction(name, partitions[i], inputs, outputs, allocator,
                               jit_context));
    partition_functions.push_back(partition_function);
  }

  // Args passed to each partition function.
  std::vector<llvm::Value*> args = {
      wrapper.GetInputsArg(),          wrapper.GetOutputsArg(),
      wrapper.GetTempBufferArg(),      wrapper.GetInterpreterEventsArg(),
      wrapper.GetInstanceContextArg(), wrapper.GetJitRuntimeArg()};

  // To handle continuation points, sequential basic blocks are created in the
  // body.
  llvm::BasicBlock* start_block = llvm::BasicBlock::Create(
      jit_context.context(), "start", wrapper.function(),
      /*InsertBefore=*/nullptr);
  auto builder = std::make_unique<llvm::IRBuilder<>>(start_block);

  std::vector<llvm::BasicBlock*> resume_blocks;
  for (int64_t i = 0; i < partitions.size(); ++i) {
    if (partitions[i].resume_point.has_value()) {
      // Partition is a point at which execution can resume, create a new basic
      // block which can be the target of a branch.
      llvm::BasicBlock* resume_block = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("resume_point_%d", partitions[i].resume_point->id),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      // The index in the resume block should correspond to the resume point id.
      CHECK_EQ(resume_blocks.size(), partitions[i].resume_point->id);
      resume_blocks.push_back(resume_block);
      builder->CreateBr(resume_block);
      builder = std::make_unique<llvm::IRBuilder<>>(resume_block);
    }

    llvm::Function* partition_function = partition_functions[i];
    llvm::CallInst* interrupt_execution = builder->CreateCall(
        partition_function->getFunctionType(), partition_function, args);

    if (partitions[i].early_exit_point.has_value()) {
      // Execution may exit at this point. Check for early exit (partition
      // function returns true).
      llvm::BasicBlock* early_return = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("early_exit_%d_return",
                          partitions[i].early_exit_point->id),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      llvm::IRBuilder<> early_return_builder(early_return);
      early_return_builder.CreateRet(
          early_return_builder.getInt64(partitions[i].early_exit_point->id));

      llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(
          jit_context.context(),
          absl::StrFormat("early_exit_%d_continue",
                          partitions[i].early_exit_point->id),
          wrapper.function(),
          /*InsertBefore=*/nullptr);
      builder->CreateCondBr(interrupt_execution, early_return, continue_block);
      builder = std::make_unique<llvm::IRBuilder<>>(continue_block);
    }
  }

  // Return zero indicating that the execution of the FunctionBase completed.
  builder->CreateRet(builder->getInt64(0));

  if (resume_blocks.empty()) {
    // Just jump from the entry block to the first block.
    wrapper.entry_builder().CreateBr(start_block);
  } else {
    // Add a switch at the entry block to jump to the appropriate continuation
    // point. The continuation point value passed into the function is the exit
    // point id.
    llvm::SwitchInst* swtch = wrapper.entry_builder().CreateSwitch(
        wrapper.GetExtraArg().value(), start_block);
    for (const Partition& partition : partitions) {
      if (partition.early_exit_point.has_value()) {
        swtch->addCase(
            wrapper.entry_builder().getInt64(partition.early_exit_point->id),
            resume_blocks[partition.early_exit_point->resume_point_id]);
      }
    }
  }
  return PartitionedFunction{.function = wrapper.function(),
                             .partitions = std::move(partitions)};
}

// Unpacks the packed value in `packed_buffer` and writes it to
// `unpacked_buffer`. `bit_offset` is a value maintained across recursive
// calls of this function indicating the offset within `packed_buffer` to read
// the packed value.
// TODO(meheff): 2022/10/03 Consider loading values in granularity larger than
// bytes when unpacking values.
absl::Status UnpackValue(llvm::Value* packed_buffer,
                         llvm::Value* unpacked_buffer, Type* xls_type,
                         int64_t bit_offset, LlvmTypeConverter& type_converter,
                         llvm::IRBuilder<>* builder) {
  switch (xls_type->kind()) {
    case TypeKind::kBits: {
      // Compute the byte offset into `packed_buffer` where first bit of data
      // for this Bits value lives.
      llvm::Type* byte_array_type =
          llvm::ArrayType::get(llvm::Type::getInt8Ty(builder->getContext()), 0);
      int64_t byte_offset = FloorOfRatio(bit_offset, int64_t{8});
      llvm::Value* byte_ptr = builder->CreateGEP(
          byte_array_type, packed_buffer,
          {builder->getInt32(0), builder->getInt32(byte_offset)});

      // Determine how many bytes need to be loaded to capture all of the bits
      // for this Bits value.
      int64_t remainder = bit_offset - byte_offset * 8;
      int64_t bytes_to_load =
          CeilOfRatio(xls_type->GetFlatBitCount() + remainder, int64_t{8});

      // The packed interface has no alignment assumptions, so make accesses
      // align(1).
      llvm::Align packed_alignment(1);
      // Load the bits and shift by the remainder.
      llvm::Value* loaded_value = builder->CreateLShr(
          builder->CreateAlignedLoad(
              builder->getIntNTy(static_cast<unsigned int>(bytes_to_load * 8)),
              byte_ptr, packed_alignment),
          remainder);

      // Convert to the native type and mask off any extra bits.
      llvm::Value* value = builder->CreateAnd(
          type_converter.PaddingMask(xls_type, *builder),
          builder->CreateIntCast(loaded_value,
                                 type_converter.ConvertToLlvmType(xls_type),
                                 /*isSigned=*/false));
      builder->CreateStore(value, unpacked_buffer);
      return absl::OkStatus();
    }
    case TypeKind::kArray: {
      ArrayType* array_type = xls_type->AsArrayOrDie();
      Type* element_xls_type = array_type->element_type();
      llvm::Type* array_llvm_type =
          type_converter.ConvertToLlvmType(array_type);
      for (uint32_t i = 0; i < array_type->size(); i++) {
        llvm::Value* unpacked_element_ptr =
            builder->CreateGEP(array_llvm_type, unpacked_buffer,
                               {
                                   builder->getInt32(0),
                                   builder->getInt32(i),
                               });
        XLS_RETURN_IF_ERROR(UnpackValue(packed_buffer, unpacked_element_ptr,
                                        element_xls_type, bit_offset,
                                        type_converter, builder));
        bit_offset += element_xls_type->GetFlatBitCount();
      }
      return absl::OkStatus();
    }
    case TypeKind::kTuple: {
      TupleType* tuple_type = xls_type->AsTupleOrDie();
      llvm::Type* tuple_llvm_type =
          type_converter.ConvertToLlvmType(tuple_type);
      for (int32_t i = tuple_type->size() - 1; i >= 0; i--) {
        // Tuple elements are stored MSB -> LSB, so we need to extract in
        // reverse order to match native layout.
        Type* element_type = tuple_type->element_type(i);
        llvm::Value* unpacked_element_ptr =
            builder->CreateGEP(tuple_llvm_type, unpacked_buffer,
                               {
                                   builder->getInt32(0),
                                   builder->getInt32(i),
                               });
        XLS_RETURN_IF_ERROR(UnpackValue(packed_buffer, unpacked_element_ptr,
                                        element_type, bit_offset,
                                        type_converter, builder));
        bit_offset += element_type->GetFlatBitCount();
      }
      return absl::OkStatus();
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled type kind: ", TypeKindToString(xls_type->kind())));
  }
}

// Loads the value in `unpacked_buffer`, packs it and writes it to
// `packed_buffer`. `bit_offset` is a value maintained across recursive calls of
// this function indicating the offset within `packed_buffer` to wite the packed
// value.
absl::Status PackValue(llvm::Value* unpacked_buffer, llvm::Value* packed_buffer,
                       Type* xls_type, int64_t bit_offset,
                       LlvmTypeConverter& type_converter,
                       llvm::IRBuilder<>* builder) {
  if (xls_type->GetFlatBitCount() == 0) {
    return absl::OkStatus();
  }
  switch (xls_type->kind()) {
    case TypeKind::kBits: {
      // Compute the byte offset into `packed_buffer` where this data should be
      // written to.
      llvm::Type* byte_array_type =
          llvm::ArrayType::get(llvm::Type::getInt8Ty(builder->getContext()), 0);
      int64_t byte_offset = FloorOfRatio(bit_offset, int64_t{8});
      llvm::Value* byte_ptr = builder->CreateGEP(
          byte_array_type, packed_buffer,
          {builder->getInt32(0), builder->getInt32(byte_offset)});

      // Determine how many bytes will be touched when writing this Bits value.
      int64_t remainder = bit_offset - byte_offset * 8;
      int64_t bytes_to_load =
          CeilOfRatio(xls_type->GetFlatBitCount() + remainder, int64_t{8});
      llvm::IntegerType* loaded_type = builder->getIntNTy(bytes_to_load * 8);

      // Load the unpacked value and cast it to the type used for loading and
      // storing to the packed buffer.
      llvm::Value* unpacked_value = builder->CreateIntCast(
          builder->CreateLoad(type_converter.ConvertToLlvmType(xls_type),
                              unpacked_buffer),
          loaded_type, /*isSigned=*/false);

      // The packed interface has no alignment assumptions, so make accesses
      // align(1).
      llvm::Align packed_alignment(1);
      if (remainder == 0) {
        // The packed value is on a byte boundary. Just write the value into the
        // buffer.
        builder->CreateAlignedStore(unpacked_value, byte_ptr, packed_alignment);
      } else {
        // Packed value is not on a byte boundary. Load in the packed value at
        // the location and do some masking and shifting. First load the packed
        // bits in the location to be written to.
        llvm::Value* loaded_packed_value =
            builder->CreateAlignedLoad(loaded_type, byte_ptr, packed_alignment);

        // Mask off any beyond the remainder bits.
        llvm::Value* remainder_mask =
            builder->CreateLShr(llvm::ConstantInt::getSigned(loaded_type, -1),
                                loaded_type->getBitWidth() - remainder);
        llvm::Value* masked_loaded_packed_value =
            builder->CreateAnd(remainder_mask, loaded_packed_value);

        // Shift the unpacked value over by the remainder.
        llvm::Value* shifted_unpacked_value =
            builder->CreateShl(unpacked_value, remainder);

        // Or the value to write with the existing bits in the loaded value.
        llvm::Value* value = builder->CreateOr(shifted_unpacked_value,
                                               masked_loaded_packed_value);
        builder->CreateAlignedStore(value, byte_ptr, packed_alignment);
      }
      return absl::OkStatus();
    }
    case TypeKind::kArray: {
      ArrayType* array_type = xls_type->AsArrayOrDie();
      Type* element_xls_type = array_type->element_type();
      llvm::Type* array_llvm_type =
          type_converter.ConvertToLlvmType(array_type);
      for (uint32_t i = 0; i < array_type->size(); i++) {
        llvm::Value* unpacked_element_ptr =
            builder->CreateGEP(array_llvm_type, unpacked_buffer,
                               {
                                   builder->getInt32(0),
                                   builder->getInt32(i),
                               });
        XLS_RETURN_IF_ERROR(PackValue(unpacked_element_ptr, packed_buffer,
                                      element_xls_type, bit_offset,
                                      type_converter, builder));
        bit_offset += element_xls_type->GetFlatBitCount();
      }
      return absl::OkStatus();
    }
    case TypeKind::kTuple: {
      // Write the unpacked elements into the buffer one-by-one.
      TupleType* tuple_type = xls_type->AsTupleOrDie();
      llvm::Type* tuple_llvm_type =
          type_converter.ConvertToLlvmType(tuple_type);
      for (int32_t i = tuple_type->size() - 1; i >= 0; i--) {
        // Tuple elements are stored MSB -> LSB, so we need to extract in
        // reverse order to match native layout.
        Type* element_type = tuple_type->element_type(i);
        llvm::Value* unpacked_element_ptr =
            builder->CreateGEP(tuple_llvm_type, unpacked_buffer,
                               {
                                   builder->getInt32(0),
                                   builder->getInt32(i),
                               });
        XLS_RETURN_IF_ERROR(PackValue(unpacked_element_ptr, packed_buffer,
                                      element_type, bit_offset, type_converter,
                                      builder));
        bit_offset += element_type->GetFlatBitCount();
      }
      return absl::OkStatus();
    }
    case TypeKind::kToken: {
      // Tokens are zero-bit constructs, so there's nothing to do!
      return absl::OkStatus();
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unhandled element kind: ", TypeKindToString(xls_type->kind())));
  }
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
      absl::StrFormat("%s_packed",
                      jit_context.MangleFunctionName(xls_function)),
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
      llvm::ArrayType::get(llvm::PointerType::getUnqual(*context), 0);
  for (int64_t i = 0; i < inputs.size(); ++i) {
    Node* input = inputs[i];
    llvm::Value* input_buffer = wrapper.entry_builder().CreateAlloca(
        jit_context.type_converter().ConvertToLlvmType(input->GetType()));
    llvm::Value* gep = wrapper.entry_builder().CreateGEP(
        pointer_array_type, input_arg_array,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(*context), i),
        });
    wrapper.entry_builder().CreateStore(input_buffer, gep);

    if (input->GetType()->GetFlatBitCount() > 0) {
      llvm::Value* packed_buffer = LoadPointerFromPointerArray(
          i, wrapper.GetInputsArg(), &wrapper.entry_builder());
      XLS_RETURN_IF_ERROR(UnpackValue(
          packed_buffer, input_buffer, input->GetType(), /*bit_offset=*/0,
          jit_context.type_converter(), &wrapper.entry_builder()));
    }
  }

  llvm::Value* output_arg_array =
      wrapper.entry_builder().CreateAlloca(llvm::ArrayType::get(
          llvm::PointerType::get(*context, 0), outputs.size()));
  for (int64_t i = 0; i < outputs.size(); ++i) {
    Node* output = outputs[i];
    llvm::Value* output_buffer = wrapper.entry_builder().CreateAlloca(
        jit_context.type_converter().ConvertToLlvmType(OutputType(output)));
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
  args.push_back(wrapper.GetInstanceContextArg());
  args.push_back(wrapper.GetJitRuntimeArg());
  args.push_back(wrapper.GetExtraArg().value());

  llvm::Value* continuation_result =
      wrapper.entry_builder().CreateCall(callee, args);

  // After returning, pack the value into the return value buffer.
  for (int64_t i = 0; i < outputs.size(); ++i) {
    Node* output = outputs[i];

    // Declare the return argument as an iX, and pack the actual data as such
    // an integer.
    llvm::Value* unpacked_output_buffer = LoadPointerFromPointerArray(
        i, output_arg_array, &wrapper.entry_builder());
    llvm::Value* packed_output_buffer = LoadPointerFromPointerArray(
        i, wrapper.GetOutputsArg(), &wrapper.entry_builder());

    XLS_RETURN_IF_ERROR(PackValue(
        unpacked_output_buffer, packed_output_buffer, OutputType(output),
        /*bit_offset=*/0, jit_context.type_converter(),
        &wrapper.entry_builder()));
  }

  // Return value of zero means that the FunctionBase completed execution.
  wrapper.entry_builder().CreateRet(continuation_result);

  return wrapper.function();
}

}  // namespace

std::unique_ptr<JitArgumentSetOwnedBuffer>
JittedFunctionBase::CreateInputBuffer(bool zero) const {
  return JitArgumentSetOwnedBuffer::CreateInput(this, GetInputBufferMetadata(),
                                                zero);
}

std::unique_ptr<JitArgumentSetOwnedBuffer>
JittedFunctionBase::CreateOutputBuffer() const {
  return JitArgumentSetOwnedBuffer::CreateOutput(this,
                                                 GetOutputBufferMetadata());
}

absl::StatusOr<std::unique_ptr<JitArgumentSetOwnedBuffer>>
JittedFunctionBase::CreateInputOutputBuffer() const {
  return JitArgumentSetOwnedBuffer::CreateInputOutput(
      this, GetInputBufferMetadata(), GetOutputBufferMetadata());
}

JitTempBuffer JittedFunctionBase::CreateTempBuffer() const {
  return JitTempBuffer(this, temp_buffer_alignment(), temp_buffer_size());
}

// Jits a function implementing `xls_function`. Also jits all transitively
// dependent xls::Functions which may be called by `xls_function`.
absl::StatusOr<JittedFunctionBase> JittedFunctionBase::BuildInternal(
    FunctionBase* xls_function, JitBuilderContext& jit_context,
    bool build_packed_wrapper) {
  std::vector<FunctionBase*> functions = GetDependentFunctions(xls_function);
  BufferAllocator allocator(&jit_context.type_converter());
  llvm::Function* top_function = nullptr;
  std::vector<Partition> top_partitions;
  for (FunctionBase* f : functions) {
    // For shared compilations (ie AOT proc-networks) we need to have multiple
    // different procs all hitting the same function. The issue is that we
    // assign slots in the temp buffer globally starting from 0 for each call to
    // JittedFunctionBase::Build. This means that different 'procs' have
    // different ideas about where in the temp buffer things go. For now to
    // avoid this issue simple create a different copy all dependent functions
    // for each top.
    // TODO(allight): Long term it would be good to avoid this headache and
    // extra code. Since the function call graph is a DAG we should be able to
    // have each function assign its own tmp buffer starting from 0 and make the
    // overall tmp-buffer the topo sort.
    XLS_RET_CHECK_EQ(
        jit_context.module()->getFunction(jit_context.MangleFunctionName(f)),
        nullptr)
        << "Multiple copies of the same function created";
    XLS_ASSIGN_OR_RETURN(PartitionedFunction partitioned_function,
                         BuildFunctionInternal(f, allocator, jit_context));
    jit_context.SetLlvmFunction(f, partitioned_function.function);
    if (f == xls_function) {
      top_function = partitioned_function.function;
      top_partitions = std::move(partitioned_function.partitions);
    }
  }
  XLS_RET_CHECK(top_function != nullptr);

  std::string function_name = jit_context.MangleFunctionName(xls_function);
  std::string packed_wrapper_name;
  if (build_packed_wrapper) {
    XLS_ASSIGN_OR_RETURN(
        llvm::Function * packed_wrapper_function,
        BuildPackedWrapper(xls_function, top_function, jit_context));
    packed_wrapper_name = packed_wrapper_function->getName().str();
  }

  XLS_RETURN_IF_ERROR(
      jit_context.llvm_compiler().CompileModule(jit_context.ConsumeModule()));

  JittedFunctionBase jitted_function;

  jitted_function.function_name_ = function_name;
  if (jit_context.llvm_compiler().IsOrcJit()) {
    XLS_ASSIGN_OR_RETURN(auto* orc_jit, jit_context.llvm_compiler().AsOrcJit());
    XLS_ASSIGN_OR_RETURN(auto fn_address, orc_jit->LoadSymbol(function_name));
    jitted_function.function_ = absl::bit_cast<JitFunctionType>(fn_address);
  } else {
    // Give it a function that will give a sort of useful error message if you
    // actually try to invoke it.
    jitted_function.function_ = InvalidJitFunctionUse;
  }

  if (build_packed_wrapper) {
    jitted_function.packed_function_name_ = packed_wrapper_name;
    if (jit_context.llvm_compiler().IsOrcJit()) {
      XLS_ASSIGN_OR_RETURN(auto* orc_jit,
                           jit_context.llvm_compiler().AsOrcJit());
      XLS_ASSIGN_OR_RETURN(auto packed_fn_address,
                           orc_jit->LoadSymbol(packed_wrapper_name));
      jitted_function.packed_function_ =
          absl::bit_cast<JitFunctionType>(packed_fn_address);
    } else {
      // Give it a function that will give a sort of useful error message if you
      // actually try to invoke it.
      jitted_function.packed_function_ = InvalidJitFunctionUse;
    }
  }

  for (const Node* input : GetJittedFunctionInputs(xls_function)) {
    Type* input_type = InputType(input);
    jitted_function.input_buffer_metadata_.push_back(
        jit_context.type_converter().GetTypeBufferMetadata(input_type));
  }
  for (const Node* output : GetJittedFunctionOutputs(xls_function)) {
    Type* output_type = OutputType(output);
    jitted_function.output_buffer_metadata_.push_back(
        jit_context.type_converter().GetTypeBufferMetadata(output_type));
  }
  jitted_function.temp_buffer_size_ = allocator.size();
  jitted_function.temp_buffer_alignment_ = allocator.alignment();

  // Indicate which nodes correspond to which early exit points.
  for (const Partition& partition : top_partitions) {
    if (partition.early_exit_point.has_value()) {
      XLS_RET_CHECK_EQ(partition.nodes.size(), 1);
      jitted_function.continuation_points_[partition.early_exit_point->id] =
          partition.nodes.front()->id();
    }
  }

  jitted_function.queue_indices_ = jit_context.queue_indices();

  return std::move(jitted_function);
}

absl::StatusOr<JittedFunctionBase> JittedFunctionBase::Build(
    Function* xls_function, LlvmCompiler& compiler,
    std::string_view symbol_salt) {
  JitBuilderContext jit_context(compiler, xls_function, symbol_salt);
  return JittedFunctionBase::BuildInternal(xls_function, jit_context,
                                           /*build_packed_wrapper=*/true);
}

absl::StatusOr<JittedFunctionBase> JittedFunctionBase::Build(
    Proc* proc, LlvmCompiler& compiler, std::string_view symbol_salt) {
  JitBuilderContext jit_context(compiler, proc, symbol_salt);
  return JittedFunctionBase::BuildInternal(proc, jit_context,
                                           /*build_packed_wrapper=*/false);
}

absl::StatusOr<JittedFunctionBase> JittedFunctionBase::Build(
    Block* block, LlvmCompiler& compiler, std::string_view symbol_salt) {
  JitBuilderContext jit_context(compiler, block, symbol_salt);
  return JittedFunctionBase::BuildInternal(block, jit_context,
                                           /*build_packed_wrapper=*/false);
}

absl::StatusOr<JittedFunctionBase> JittedFunctionBase::BuildFromAot(
    const AotEntrypointProto& abi, JitFunctionType entrypoint,
    std::optional<JitFunctionType> packed_entrypoint) {
  XLS_RET_CHECK(abi.has_function_symbol());
  XLS_RET_CHECK_EQ(abi.input_buffer_sizes().size(),
                   abi.input_buffer_alignments().size());
  XLS_RET_CHECK_EQ(abi.input_buffer_sizes().size(),
                   abi.input_buffer_abi_alignments().size());
  XLS_RET_CHECK_EQ(abi.output_buffer_sizes().size(),
                   abi.output_buffer_alignments().size());
  XLS_RET_CHECK_EQ(abi.output_buffer_sizes().size(),
                   abi.output_buffer_abi_alignments().size());
  std::optional<std::string> packed_name;
  if (packed_entrypoint) {
    XLS_RET_CHECK(abi.has_packed_function_symbol());
    XLS_RET_CHECK_EQ(abi.packed_input_buffer_sizes().size(),
                     abi.input_buffer_sizes().size());
    XLS_RET_CHECK_EQ(abi.packed_output_buffer_sizes().size(),
                     abi.output_buffer_sizes().size());
    packed_name = abi.packed_function_symbol();
  }
  XLS_RET_CHECK(abi.has_temp_buffer_size());
  XLS_RET_CHECK(abi.has_temp_buffer_alignment());
  XLS_RET_CHECK_EQ(abi.type() == AotEntrypointProto::PROC,
                   abi.has_proc_metadata());
  absl::flat_hash_map<int64_t, int64_t> continuation_points;
  absl::btree_map<std::string, int64_t> queue_indices;

  if (abi.type() == AotEntrypointProto::PROC) {
    continuation_points.insert(
        abi.proc_metadata().continuation_point_node_ids().begin(),
        abi.proc_metadata().continuation_point_node_ids().end());
    queue_indices.insert(abi.proc_metadata().channel_queue_indices().begin(),
                         abi.proc_metadata().channel_queue_indices().end());
  }

  std::vector<TypeBufferMetadata> input_buffer_metadata;
  input_buffer_metadata.reserve(abi.input_buffer_sizes().size());
  for (int64_t i = 0; i < abi.input_buffer_sizes().size(); ++i) {
    input_buffer_metadata.push_back(TypeBufferMetadata{
        .size = abi.input_buffer_sizes()[i],
        .preferred_alignment = abi.input_buffer_alignments()[i],
        .abi_alignment = abi.input_buffer_abi_alignments()[i],
        .packed_size =
            packed_entrypoint ? abi.packed_input_buffer_sizes()[i] : 0});
  }

  std::vector<TypeBufferMetadata> output_buffer_metadata;
  output_buffer_metadata.reserve(abi.output_buffer_sizes().size());
  for (int64_t i = 0; i < abi.output_buffer_sizes().size(); ++i) {
    output_buffer_metadata.push_back(TypeBufferMetadata{
        .size = abi.output_buffer_sizes()[i],
        .preferred_alignment = abi.output_buffer_alignments()[i],
        .abi_alignment = abi.output_buffer_abi_alignments()[i],
        .packed_size =
            packed_entrypoint ? abi.packed_output_buffer_sizes()[i] : 0});
  }

  return JittedFunctionBase(
      abi.function_symbol(), entrypoint, packed_name, packed_entrypoint,
      input_buffer_metadata, output_buffer_metadata, abi.temp_buffer_size(),
      abi.temp_buffer_alignment(), std::move(continuation_points),
      std::move(queue_indices));
}

int64_t JittedFunctionBase::RunJittedFunction(
    const JitArgumentSet& inputs, JitArgumentSet& outputs,
    JitTempBuffer& temp_buffer, InterpreterEvents* events,
    InstanceContext* instance_context, JitRuntime* jit_runtime,
    int64_t continuation_point) const {
  CHECK(inputs.is_inputs());
  CHECK_EQ(inputs.source(), this);
  CHECK(outputs.is_outputs());
  CHECK_EQ(outputs.source(), this);
  CHECK_EQ(temp_buffer.source(), this);
  return function_(inputs.get_base_pointer(), outputs.get_base_pointer(),
                   temp_buffer.get_base_pointer(), events, instance_context,
                   jit_runtime, continuation_point);
}

namespace {
bool IsAligned(const void* ptr, int64_t align) {
  return (absl::bit_cast<uintptr_t>(ptr) % align) == 0;
}

absl::Status VerifyOffsetAbiAlignments(
    uint8_t const* const* const ptrs,
    absl::Span<const TypeBufferMetadata> alignments) {
  for (int64_t i = 0; i < alignments.size(); ++i) {
    if (absl::bit_cast<uintptr_t>(ptrs[i]) % alignments[i].abi_alignment != 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("element %d of input vector does not have alignment "
                          "of %d. Pointer is %p",
                          i, alignments[i].abi_alignment, ptrs[i]));
    }
  }
  return absl::OkStatus();
}
}  // namespace

template <bool kForceZeroCopy>
int64_t JittedFunctionBase::RunUnalignedJittedFunction(
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    InterpreterEvents* events, InstanceContext* instance_context,
    JitRuntime* jit_runtime, int64_t continuation) const {
  if constexpr (kForceZeroCopy) {
    DCHECK_OK(VerifyOffsetAbiAlignments(inputs, GetInputBufferMetadata()));
    DCHECK_OK(VerifyOffsetAbiAlignments(outputs, GetOutputBufferMetadata()));
    DCHECK(IsAligned(temp_buffer, temp_buffer_alignment_));
  } else {
    if (!VerifyOffsetAbiAlignments(inputs, GetInputBufferMetadata()).ok() ||
        !VerifyOffsetAbiAlignments(outputs, GetOutputBufferMetadata()).ok() ||
        !IsAligned(temp_buffer, temp_buffer_alignment_)) {
      std::unique_ptr<JitArgumentSetOwnedBuffer> aligned_input =
          CreateInputBuffer();
      std::unique_ptr<JitArgumentSetOwnedBuffer> aligned_output =
          CreateOutputBuffer();
      JitTempBuffer temp(CreateTempBuffer());
      memcpy(temp.get_base_pointer(), temp_buffer, temp_buffer_size_);
      for (int i = 0; i < GetInputBufferMetadata().size(); ++i) {
        memcpy(aligned_input->get_element_pointers()[i], inputs[i],
               GetInputBufferMetadata()[i].size);
      }
      auto result =
          RunJittedFunction(*aligned_input, *aligned_output, temp, events,
                            instance_context, jit_runtime, continuation);
      memcpy(temp_buffer, temp.get_base_pointer(), temp_buffer_size_);
      for (int i = 0; i < GetOutputBufferMetadata().size(); ++i) {
        memcpy(outputs[i], aligned_output->get_element_pointers()[i],
               GetOutputBufferMetadata()[i].size);
      }
      return result;
    }
  }
  return function_(inputs, outputs, temp_buffer, events, instance_context,
                   jit_runtime, continuation);
}

template int64_t
JittedFunctionBase::RunUnalignedJittedFunction</*kForceZeroCopy=*/true>(
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    InterpreterEvents* events, InstanceContext* instance_context,
    JitRuntime* jit_runtime, int64_t continuation) const;

template int64_t
JittedFunctionBase::RunUnalignedJittedFunction</*kForceZeroCopy=*/false>(
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    InterpreterEvents* events, InstanceContext* instance_context,
    JitRuntime* jit_runtime, int64_t continuation) const;

std::optional<int64_t> JittedFunctionBase::RunPackedJittedFunction(
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    InterpreterEvents* events, InstanceContext* instance_context,
    JitRuntime* jit_runtime, int64_t continuation_point) const {
  // Packed Jit makes no alignment assumptions, so nothing to check.
  if (packed_function_) {
    return (*packed_function_)(inputs, outputs, temp_buffer, events,
                               instance_context, jit_runtime,
                               continuation_point);
  }
  return std::nullopt;
}
}  // namespace xls
