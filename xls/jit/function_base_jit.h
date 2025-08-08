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
#ifndef XLS_JIT_FUNCTION_BASE_JIT_H_
#define XLS_JIT_FUNCTION_BASE_JIT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {

// Type alias for the jitted functions implementing XLS FunctionBases. Argument
// descriptions:
//   inputs: array of pointers to input buffers (e.g., parameter values). Note
//        that for Block* functions specifically the inputs are all the input
//        ports followed by all the registers.
//   outputs: array of pointers to output buffers (e.g., function return value,
//        proc next state values). Note that for Block* specifically the outputs
//        are all the output-ports followed by all the new register values.
//   temp_buffer: heap-allocated scratch space for the JITed funcion. This
//       buffer hold temporary node values which cannot be stack allocated via
//       allocas.
//   events: pointer to events objects which records information from
//       instructions like trace.
//   instance_context: pointer to an InstanceContext. Only used in procs. Holds
//       information about the proc instance being evaluated.
//   jit_runtime: pointer to a JitRuntime object which is a set of functions the
//       JITted code may need to use, for example, to copy data to Values.
//   continuation_point: an opaque value indicating the point in the
//      FunctionBase to start execution when the jitted function is called.
//      Used to enable interruption and resumption of execution of the
//      the FunctionBase due to blocking operations such as receives.
//
// Returns the continuation point at which execution stopped or 0 if the tick
// completed.
using JitFunctionType = int64_t (*)(const uint8_t* const* inputs,
                                    uint8_t* const* outputs, void* temp_buffer,
                                    InterpreterEvents* events,
                                    InstanceContext* instance_context,
                                    JitRuntime* jit_runtime,
                                    int64_t continuation_point);

// Abstraction holding function pointers and metadata about a jitted function
// implementing a XLS Function, Proc, etc.
//
// TODO(allight): We should rename this to CompiledFunctionType or something.
class JittedFunctionBase {
 public:
  JittedFunctionBase() = default;
  // Builds and returns an LLVM IR function implementing the given XLS
  // function.
  static absl::StatusOr<JittedFunctionBase> Build(
      Function* xls_function, LlvmCompiler& compiler,
      std::string_view symbol_salt = "");

  // Builds and returns an LLVM IR function implementing the given XLS
  // proc.
  static absl::StatusOr<JittedFunctionBase> Build(
      Proc* proc, LlvmCompiler& compiler, std::string_view symbol_salt = "");

  // Builds and returns an LLVM IR function implementing the given XLS
  // block.
  static absl::StatusOr<JittedFunctionBase> Build(
      Block* block, LlvmCompiler& compiler, std::string_view symbol_salt = "");

  // Builds and returns a JittedFunctionBase using code and ABIs provided by an
  // earlier AOT compile.
  static absl::StatusOr<JittedFunctionBase> BuildFromAot(
      const AotEntrypointProto& abi, JitFunctionType entrypoint,
      std::optional<JitFunctionType> packed_entrypoint = std::nullopt);

  // Create a buffer with space for all inputs, correctly aligned.
  //
  // If 'zero' then zero initialize the buffer contents.
  std::unique_ptr<JitArgumentSetOwnedBuffer> CreateInputBuffer(
      bool zero = false) const;

  // Create a buffer with space for all outputs, correctly aligned.
  std::unique_ptr<JitArgumentSetOwnedBuffer> CreateOutputBuffer() const;

  // Return if the required alignments and sizes of both the inputs and outputs
  // are identical.
  bool InputsAndOutputsAreEquivalent() const {
    return absl::c_equal(input_buffer_metadata_, output_buffer_metadata_);
  }

  // Create a buffer capable of being used for both the input and output of a
  // jitted function.
  //
  // Returns an error if `InputsAndOutputsAreEquivalent()` is not true.
  absl::StatusOr<std::unique_ptr<JitArgumentSetOwnedBuffer>>
  CreateInputOutputBuffer() const;

  // Create a buffer usable as the temporary storage, correctly aligned.
  JitTempBuffer CreateTempBuffer() const;

  // Execute the actual function (after verifying some invariants)
  int64_t RunJittedFunction(const JitArgumentSet& inputs,
                            JitArgumentSet& outputs, JitTempBuffer& temp_buffer,
                            InterpreterEvents* events,
                            InstanceContext* instance_context,
                            JitRuntime* jit_runtime,
                            int64_t continuation_point) const;

  // Execute the jitted function using inputs not created by this function.
  // If kForceZeroCopy is false the inputs will be memcpy'd if needed to aligned
  // temporary buffers.
  template <bool kForceZeroCopy = false>
  int64_t RunUnalignedJittedFunction(const uint8_t* const* inputs,
                                     uint8_t* const* outputs, void* temp_buffer,
                                     InterpreterEvents* events,
                                     InstanceContext* instance_context,
                                     JitRuntime* jit_runtime,
                                     int64_t continuation) const;

  // Execute the actual function (after verifying some invariants)
  std::optional<int64_t> RunPackedJittedFunction(
      const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
      InterpreterEvents* events, InstanceContext* instance_context,
      JitRuntime* jit_runtime, int64_t continuation_point) const;

  // Checks if we have a packed version of the function.
  bool HasPackedFunction() const { return packed_function_.has_value(); }
  std::optional<std::string_view> packed_function_name() const {
    return HasPackedFunction()
               ? std::make_optional<std::string_view>(*packed_function_name_)
               : std::nullopt;
  }

  std::string_view function_name() const { return function_name_; }

  absl::Span<const TypeBufferMetadata> GetInputBufferMetadata() const {
    return input_buffer_metadata_;
  }

  absl::Span<const TypeBufferMetadata> GetOutputBufferMetadata() const {
    return output_buffer_metadata_;
  }

  int64_t temp_buffer_size() const { return temp_buffer_size_; }

  int64_t temp_buffer_alignment() const { return temp_buffer_alignment_; }

  const absl::flat_hash_map<int64_t, int64_t>& continuation_points() const {
    return continuation_points_;
  }

  // The map from channel reference name to the index of the respective queue in
  // the instance context.
  const absl::btree_map<std::string, int64_t>& queue_indices() const {
    return queue_indices_;
  }

  JittedFunctionBase WithCodePointers(
      JitFunctionType entrypoint,
      std::optional<JitFunctionType> packed_entrypoint = std::nullopt) const {
    JittedFunctionBase res = *this;
    res.function_ = entrypoint;
    res.packed_function_ = packed_entrypoint;
    return res;
  }

 private:
  JittedFunctionBase(std::string function_name, JitFunctionType function,
                     std::optional<std::string> packed_function_name,
                     std::optional<JitFunctionType> packed_function,
                     std::vector<TypeBufferMetadata> input_buffer_metadata,
                     std::vector<TypeBufferMetadata> output_buffer_metadata,
                     int64_t temp_buffer_size, int64_t temp_buffer_alignment,
                     absl::flat_hash_map<int64_t, int64_t> continuation_points,
                     absl::btree_map<std::string, int64_t> queue_indices)
      : function_name_(std::move(function_name)),
        function_(function),
        packed_function_name_(std::move(packed_function_name)),
        packed_function_(packed_function),
        input_buffer_metadata_(std::move(input_buffer_metadata)),
        output_buffer_metadata_(std::move(output_buffer_metadata)),
        temp_buffer_size_(temp_buffer_size),
        temp_buffer_alignment_(temp_buffer_alignment),
        continuation_points_(std::move(continuation_points)),
        queue_indices_(std::move(queue_indices)) {}

  static absl::StatusOr<JittedFunctionBase> BuildInternal(
      FunctionBase* function, JitBuilderContext& jit_context,
      bool build_packed_wrapper);

  // Name and function pointer for the jitted function which accepts/produces
  // arguments/results in LLVM native format.
  std::string function_name_;
  JitFunctionType function_;

  // Name and function pointer for the jitted function which accepts/produces
  // arguments/results in a packed format. Only exists for JITted
  // xls::Functions, not procs.
  std::optional<std::string> packed_function_name_;
  std::optional<JitFunctionType> packed_function_;

  // Sizes of the inputs/outputs in native LLVM format for `function_base`.
  std::vector<TypeBufferMetadata> input_buffer_metadata_;
  std::vector<TypeBufferMetadata> output_buffer_metadata_;

  // Size of the temporary buffer required by `function`.
  int64_t temp_buffer_size_ = -1;
  // Alignment of the temporary buffer required by `function`
  int64_t temp_buffer_alignment_ = -1;

  // Map from the continuation point return value to the corresponding node id
  // at which execution was interrupted.
  absl::flat_hash_map<int64_t, int64_t> continuation_points_;

  // The map from channel reference name to the index of the respective queue in
  // the instance context.
  absl::btree_map<std::string, int64_t> queue_indices_;
};

struct FunctionEntrypoint {
  FunctionBase* function;
  JittedFunctionBase jit_info;

  // Extra field information.
  absl::flat_hash_map<std::string, std::string> register_aliases = {};
  absl::flat_hash_map<std::string, Type*> added_registers = {};
};

// Data structure containing jitted object code and metadata about how to call
// it.
struct JitObjectCode {
  std::vector<uint8_t> object_code;
  std::vector<FunctionEntrypoint> entrypoints;
  llvm::DataLayout data_layout;

  // If set a package that needs to stay alive to keep 'FunctionEntrypoint' (ie
  // FunctionBase*, type maps etc) valid.
  std::unique_ptr<Package> package = nullptr;
};

}  // namespace xls

#endif  // XLS_JIT_FUNCTION_BASE_JIT_H_
