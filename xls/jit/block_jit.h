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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/orc_jit.h"

namespace xls {

class BlockJitContinuation;
class BlockJit {
 public:
  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(Block* block);

  // Create a new blank block with no registers or ports set. Can be cycled
  // independently of other blocks/continuations.
  std::unique_ptr<BlockJitContinuation> NewContinuation();

  // Runs a single cycle of a block with the given continuation.
  absl::Status RunOneCycle(BlockJitContinuation& continuation);

  OrcJit& orc_jit() const { return *jit_; }

  JitRuntime* runtime() const { return runtime_.get(); }

  // Get how large each pointer buffer for the input ports are.
  absl::Span<const int64_t> input_port_sizes() const {
    return absl::MakeConstSpan(function_.input_buffer_sizes())
        .subspan(0, block_->GetInputPorts().size());
  }

  // Get how large each pointer buffer for the registers are.
  absl::Span<int64_t const> register_sizes() const {
    return absl::MakeConstSpan(function_.input_buffer_sizes())
        .subspan(block_->GetInputPorts().size());
  }

 private:
  BlockJit(Block* block, std::unique_ptr<JitRuntime> runtime,
           std::unique_ptr<OrcJit> jit, JittedFunctionBase function)
      : block_(block),
        runtime_(std::move(runtime)),
        jit_(std::move(jit)),
        function_(std::move(function)) {}

  Block* block_;
  std::unique_ptr<JitRuntime> runtime_;
  std::unique_ptr<OrcJit> jit_;
  JittedFunctionBase function_;
};

class BlockJitContinuation {
 private:
  class IOSpace {
   public:
    enum class RegisterSpace : uint8_t { kLeft, kRight };
    IOSpace(JitArgumentSet left, JitArgumentSet right,
            RegisterSpace initial_space = RegisterSpace::kLeft)
        : left_(std::move(left)),
          right_(std::move(right)),
          current_side_(initial_space) {}

    // Switch the currently active space to the alternate.
    void Swap() {
      current_side_ = (current_side_ == RegisterSpace::kLeft)
                          ? RegisterSpace::kRight
                          : RegisterSpace::kLeft;
    }

    // Force a particular set of inputs to be the active inputs.
    void SetActive(RegisterSpace space) { current_side_ = space; }

    const JitArgumentSet& current() const {
      switch (current_side_) {
        case RegisterSpace::kLeft:
          return left_;
        case RegisterSpace::kRight:
          return right_;
      }
    }

    JitArgumentSet& current() {
      switch (current_side_) {
        case RegisterSpace::kLeft:
          return left_;
        case RegisterSpace::kRight:
          return right_;
      }
    }

    const JitArgumentSet& left() const { return left_; }

    const JitArgumentSet& right() const { return right_; }

   private:
    JitArgumentSet left_;
    JitArgumentSet right_;
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
    // Registers follow the input-ports in the input vector.
    return function_inputs().subspan(0, /*len=*/block_->GetInputPorts().size());
  }
  // Gets pointers to the JIT ABI struct pointers for each register.
  // Write to the pointed-to memory to manually set a register.
  absl::Span<uint8_t* const> register_pointers() const {
    // Previous register values got swapped over to the inputs.
    // Registers follow the input-ports in the input vector.
    return function_inputs().subspan(block_->GetInputPorts().size());
  }
  // Gets the pointers to the JIT ABI output pointers for each output port.
  absl::Span<uint8_t const* const> output_port_pointers() const {
    // output ports are before the registers.
    return function_outputs().subspan(0,
                                      /*len=*/block_->GetOutputPorts().size());
  }

  const InterpreterEvents& GetEvents() const { return events_; }
  InterpreterEvents& GetEvents() { return events_; }
  void ClearEvents() { events_.Clear(); }

  const JitTempBuffer& temp_buffer() const { return temp_buffer_; }

 private:
  using BufferPair = std::array<JitArgumentSet, 2>;
  BlockJitContinuation(Block* block, BlockJit* jit,
                       const JittedFunctionBase& jit_func);
  static IOSpace MakeCombinedBuffers(const JittedFunctionBase& jit_func,
                                     const Block* block,
                                     const JitArgumentSet& ports,
                                     const BufferPair& regs, bool input);

  // Create a new aligned buffer with the first 'left_count' elements of left
  // and the rest from right.
  //
  // Both left and right must live longer than the returned buffer.
  // This should only be used to enable some elements to be shared between 2
  // input & output buffers for block-jit.
  static absl::StatusOr<JitArgumentSet> CombineBuffers(
      const JittedFunctionBase& jit_func,
      const JitArgumentSet& left ABSL_ATTRIBUTE_LIFETIME_BOUND,
      int64_t left_count,
      const JitArgumentSet& rest ABSL_ATTRIBUTE_LIFETIME_BOUND,
      int64_t rest_start, bool is_inputs);

  void SwapRegisters() {
    input_buffers_.Swap();
    output_buffers_.Swap();
  }
  absl::Span<uint8_t* const> function_inputs() const {
    return input_buffers_.current().pointers();
  }
  absl::Span<uint8_t* const> function_outputs() const {
    return output_buffers_.current().pointers();
  }

  const Block* block_;
  BlockJit* block_jit_;

  // Buffers for the registers. Note this includes (unused) space for the input
  // ports.
  BufferPair register_buffers_memory_;
  // Buffers for the input ports. Note this includes (unused) space for the
  // registers.
  JitArgumentSet input_port_buffers_memory_;
  // Buffers for the output ports. Note this includes (unused) space for the
  // registers.
  JitArgumentSet output_port_buffers_memory_;

  // Input pointers. Memory is owned by register_buffers_memory_ and
  // input_port_buffers_memory_. Not thread safe. NB The inputs are organized as
  // <input_ports><Registers>.
  IOSpace input_buffers_;
  // Output pointers. Memory is owned by register_buffers_memory_ and
  // input_port_buffers_memory_. Not thread safe. NB The outputs are organized
  // as <output_ports><Registers>.
  IOSpace output_buffers_;

  // Temporary scratch storage. Not thread safe.
  JitTempBuffer temp_buffer_;

  InstanceContext callbacks_;

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
      const BlockElaboration& elaboration) const final;

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

  // Expose the overload without `initial_registers`.
  using BlockEvaluator::NewContinuation;

 protected:
  absl::StatusOr<std::unique_ptr<BlockContinuation>> NewContinuation(
      BlockElaboration&& elaboration,
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
  XLS_ASSIGN_OR_RETURN(BlockElaboration elaboration,
                       BlockElaboration::Elaborate(block));
  return kJitBlockEvaluator.EvaluateBlock(inputs, reg_state, elaboration);
}

}  // namespace xls

#endif  // XLS_JIT_BLOCK_JIT_H_
