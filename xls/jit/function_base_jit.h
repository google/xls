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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/proc.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

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
//   user_data: pointer to arbitrary data passed to send/receive functions
//       in procs.
//   jit_runtime: pointer to a JitRuntime object.
//   continuation_point: an opaque value indicating the point in the
//      FunctionBase to start execution when the jitted function is called.
//      Used to enable interruption and resumption of execution of the
//      the FunctionBase due to blocking operations such as receives.
//
// Returns the continuation point at which execution stopped or 0 if the tick
// completed.
using JitFunctionType = int64_t (*)(const uint8_t* const* inputs,
                                    uint8_t* const* outputs, void* temp_buffer,
                                    InterpreterEvents* events, void* user_data,
                                    JitRuntime* jit_runtime,
                                    int64_t continuation_point);

// Abstraction holding function pointers and metadata about a jitted function
// implementing a XLS Function, Proc, etc.
struct JittedFunctionBase {
  // The XLS FunctionBase this jitted function implements.
  FunctionBase* function_base;

  // Name and function pointer for the jitted function which accepts/produces
  // arguments/results in LLVM native format.
  std::string function_name;
  JitFunctionType function;

  // Execute the actual function (after verifying some invariants)
  int64_t RunJittedFunction(const uint8_t* const* inputs,
                            uint8_t* const* outputs, void* temp_buffer,
                            InterpreterEvents* events, void* user_data,
                            JitRuntime* jit_runtime,
                            int64_t continuation_point) const;

  // Name and function pointer for the jitted function which accepts/produces
  // arguments/results in a packed format. Only exists for JITted
  // xls::Functions, not procs.
  std::optional<std::string> packed_function_name;
  std::optional<JitFunctionType> packed_function;

  // Execute the actual function (after verifying some invariants)
  std::optional<int64_t> RunPackedJittedFunction(
      const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
      InterpreterEvents* events, void* user_data, JitRuntime* jit_runtime,
      int64_t continuation_point) const;

  // Sizes of the inputs/outputs in native LLVM format for `function_base`.
  std::vector<int64_t> input_buffer_sizes;
  std::vector<int64_t> output_buffer_sizes;

  // alignment preferences of each input/output buffer.
  std::vector<int64_t> input_buffer_prefered_alignments;
  std::vector<int64_t> output_buffer_prefered_alignments;

  // alignment ABI requirements of each input/output buffer.
  std::vector<int64_t> input_buffer_abi_alignments;
  std::vector<int64_t> output_buffer_abi_alignments;

  // Sizes of the inputs/outputs in packed format for `function_base`.
  std::vector<int64_t> packed_input_buffer_sizes;
  std::vector<int64_t> packed_output_buffer_sizes;

  // Size of the temporary buffer required by `function`.
  int64_t temp_buffer_size;

  // Map from the continuation point return value to the corresponding node at
  // which execution was interrupted.
  absl::flat_hash_map<int64_t, Node*> continuation_points;
};

// Builds and returns an LLVM IR function implementing the given XLS
// function.
absl::StatusOr<JittedFunctionBase> BuildFunction(Function* xls_function,
                                                 OrcJit& orc_jit);

// Builds and returns an LLVM IR function implementing the given XLS
// proc.
absl::StatusOr<JittedFunctionBase> BuildProcFunction(
    Proc* proc, JitChannelQueueManager* queue_mgr, OrcJit& orc_jit);

// Builds and returns an LLVM IR function implementing the given XLS
// block.
absl::StatusOr<JittedFunctionBase> BuildBlockFunction(Block* block,
                                                      OrcJit& jit);

}  // namespace xls

#endif  // XLS_JIT_FUNCTION_BASE_JIT_H_
