// Copyright 2023 The XLS Authors
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

#ifndef XLS_JIT_BLOCK_JIT_H_
#define XLS_JIT_BLOCK_JIT_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"
namespace xls {

class BlockJitContinuation;
class BlockJit {
 public:
  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(Block* block,
                                                          JitRuntime* runtime);

  // Create a new blank block with no registers or ports set. Can be cycled
  // independently of other blocks/continuations.
  std::unique_ptr<BlockJitContinuation> NewContinuation();

  // Runs a single cycle of a block with the given continuation.
  absl::Status RunOneCycle(BlockJitContinuation& continuation);

  OrcJit& orc_jit() const { return *jit_; }

  // Get how large each pointer buffer for the input ports are.
  absl::Span<const int64_t> input_port_sizes() const {
    return absl::MakeConstSpan(function_.input_buffer_sizes)
        .subspan(0, block_->GetInputPorts().size());
  }

  // Get how large each pointer buffer for the registers are.
  absl::Span<int64_t const> register_sizes() const {
    return absl::MakeConstSpan(function_.input_buffer_sizes)
        .subspan(block_->GetInputPorts().size());
  }

 private:
  BlockJit(Block* block, JitRuntime* runtime, std::unique_ptr<OrcJit> jit,
           JittedFunctionBase function)
      : block_(block),
        runtime_(runtime),
        jit_(std::move(jit)),
        function_(std::move(function)) {}

  Block* block_;
  JitRuntime* runtime_;
  std::unique_ptr<OrcJit> jit_;
  JittedFunctionBase function_;
};

class BlockJitContinuation {
 private:
  class IOSpace {
   public:
    enum class RegisterSpace : uint8_t { kLeft, kRight };
    IOSpace(std::vector<uint8_t*>&& left, std::vector<uint8_t*>&& right,
            RegisterSpace initial_space = RegisterSpace::kLeft)
        : left_(left), right_(right), current_side_(initial_space) {}

    // Switch the currently active space to the alternate.
    void Swap() {
      current_side_ = (current_side_ == RegisterSpace::kLeft)
                          ? RegisterSpace::kRight
                          : RegisterSpace::kLeft;
    }

    absl::Span<uint8_t* const> current() const {
      switch (current_side_) {
        case RegisterSpace::kLeft:
          return left_;
        case RegisterSpace::kRight:
          return right_;
      }
    }

    absl::Span<uint8_t* const> left() const { return left_; }

    absl::Span<uint8_t* const> right() const { return right_; }

   private:
    std::vector<uint8_t*> left_;
    std::vector<uint8_t*> right_;
    RegisterSpace current_side_;
  };

 public:
  // Overwrite all input-ports with given values.
  absl::Status SetInputPorts(absl::Span<const Value> values);
  absl::Status SetInputPorts(std::initializer_list<const Value> values) {
    return SetInputPorts(
        absl::Span<const Value>(values.begin(), values.size()));
  }
  // Overwrite all input-ports with given values.
  absl::Status SetInputPorts(absl::Span<const uint8_t* const> inputs);
  // Overwrite all input-ports with given values.
  absl::Status SetInputPorts(
      const absl::flat_hash_map<std::string, Value>& inputs);
  // Overwrite all registers with given values.
  absl::Status SetRegisters(absl::Span<const Value> values);
  // Overwrite all registers with given values.
  absl::Status SetRegisters(absl::Span<const uint8_t* const> regs);
  // Overwrite all registers with given values.
  absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs);

  std::vector<Value> GetOutputPorts() const;
  absl::flat_hash_map<std::string, Value> GetOutputPortsMap() const;
  std::vector<Value> GetRegisters() const;
  absl::flat_hash_map<std::string, Value> GetRegistersMap() const;

  absl::flat_hash_map<std::string, int64_t> GetInputPortIndices() const;
  absl::flat_hash_map<std::string, int64_t> GetOutputPortIndices() const;
  absl::flat_hash_map<std::string, int64_t> GetRegisterIndices() const;

  // Gets pointers to the JIT ABI struct input pointers for each input port
  // Write to the pointed to memory to manually set an input port value for the
  // next cycle.
  absl::Span<uint8_t* const> input_port_pointers() const {
    return input_port_pointers_;
  }
  // Gets pointers to the JIT ABI struct pointers for each register.
  // Write to the pointed-to memory to manually set a register.
  absl::Span<uint8_t* const> register_pointers() const {
    return register_pointers_.current();
  }
  // Gets the pointers to the JIT ABI output pointers for each output port.
  absl::Span<uint8_t* const> output_port_pointers() const {
    return output_port_pointers_;
  }

  const InterpreterEvents& GetEvents() const { return events_; }
  InterpreterEvents& GetEvents() { return events_; }
  void ClearEvents() { events_.Clear(); }

  absl::Span<const uint8_t> temp_buffer() const {
    return absl::MakeConstSpan(temp_data_arena_);
  }

  absl::Span<uint8_t> temp_buffer() { return absl::MakeSpan(temp_data_arena_); }

 private:
  BlockJitContinuation(Block* block, BlockJit* jit, JitRuntime* runtime,
                       size_t temp_size,
                       absl::Span<const int64_t> register_sizes,
                       absl::Span<const int64_t> register_alignments,
                       absl::Span<const int64_t> output_port_sizes,
                       absl::Span<const int64_t> output_port_alignments,
                       absl::Span<const int64_t> input_port_sizes,
                       absl::Span<const int64_t> input_port_alignments);

  void SwapRegisters() {
    register_pointers_.Swap();
    full_output_pointer_set_.Swap();
    full_input_pointer_set_.Swap();
  }
  absl::Span<uint8_t* const> function_inputs() const {
    return full_input_pointer_set_.current();
  }
  absl::Span<uint8_t* const> function_outputs() const {
    return full_output_pointer_set_.current();
  }

  const Block* block_;
  BlockJit* block_jit_;
  JitRuntime* runtime_;

  // Data to store registers in. Reused for each invoke. These are not directly
  // used but merely hold memory live for the pointers.
  std::vector<uint8_t> register_arena_left_;
  std::vector<uint8_t> register_arena_right_;
  // Data to store port output in. This is not directly used but merely holds
  // memory live for the pointers.
  std::vector<uint8_t> output_port_arena_;
  // Data to store port input in. This is not directly used but merely holds
  // memory live for the pointers.
  std::vector<uint8_t> input_port_arena_;

  // The register pointer file, aligned as required.
  IOSpace register_pointers_;

  // The output port pointers, aligned as required.
  const std::vector<uint8_t*> output_port_pointers_;

  // The input port pointers, aligned as required.
  const std::vector<uint8_t*> input_port_pointers_;
  // The input value pointers the register pointers, aligned as required.
  IOSpace full_input_pointer_set_;
  // The output value pointers followed by the register pointers, aligned as
  // required.
  IOSpace full_output_pointer_set_;

  // Data block to store temporary data in.
  std::vector<uint8_t> temp_data_arena_;
  // Data block to store temporary data in, aligned as required.
  uint8_t* temp_data_ptr_;

  InterpreterEvents events_;

  friend class BlockJit;
};

// Most basic jit-evaluator. This is basically only for testing the core
// jit-behaviors in isolation from the continuation update behaviors.
class JitBlockEvaluator : public BlockEvaluator {
 public:
  constexpr JitBlockEvaluator() : JitBlockEvaluator("Jit") {}
  absl::StatusOr<BlockRunResult> EvaluateBlock(
      const absl::flat_hash_map<std::string, Value>& inputs,
      const absl::flat_hash_map<std::string, Value>& reg_state,
      Block* block) const final;

 protected:
  constexpr explicit JitBlockEvaluator(std::string_view name)
      : BlockEvaluator(name) {}
};

// An block-evaluator that doesn't attempt to use continuations to save register
// state between calls. Should only be used for testing.
static const JitBlockEvaluator kJitBlockEvaluator;

// A jit block evaluator that tries to use the jit's register saving as
// possible.
class StreamingJitBlockEvaluator : public JitBlockEvaluator {
 public:
  constexpr StreamingJitBlockEvaluator() : JitBlockEvaluator("StreamingJit") {}
  absl::StatusOr<std::vector<absl::flat_hash_map<std::string, Value>>>
  EvaluateSequentialBlock(
      Block* block,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs)
      const final;

  absl::StatusOr<BlockIOResults> EvaluateChannelizedSequentialBlock(
      Block* block, absl::Span<ChannelSource> channel_sources,
      absl::Span<ChannelSink> channel_sinks,
      absl::Span<const absl::flat_hash_map<std::string, Value>> inputs,
      const std::optional<verilog::ResetProto>& reset,
      int64_t seed) const override;

  absl::StatusOr<std::unique_ptr<BlockContinuation>> NewContinuation(
      Block* block,
      const absl::flat_hash_map<std::string, Value>& initial_registers)
      const override;
};

static const StreamingJitBlockEvaluator kStreamingJitBlockEvaluator;

// Runs a single cycle of a block with the given register values and input
// values. Returns the value sent to the output port and the next register
// state. This is a compatibility API that matches the interpreter runner.
inline absl::StatusOr<BlockRunResult> JitBlockRun(
    const absl::flat_hash_map<std::string, Value>& inputs,
    const absl::flat_hash_map<std::string, Value>& reg_state, Block* block) {
  return kJitBlockEvaluator.EvaluateBlock(inputs, reg_state, block);
}

}  // namespace xls

#endif  // XLS_JIT_BLOCK_JIT_H_
