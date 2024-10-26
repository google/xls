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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/events.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"
#include "xls/jit/orc_jit.h"

namespace xls {

class BlockJitContinuation;
class BlockJit {
 public:
  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(
      Block* block, bool support_observer_callbacks = false);
  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(
      const BlockElaboration& elab, bool support_observer_callbacks = false);

  static absl::StatusOr<std::unique_ptr<BlockJit>> CreateFromAot(
      Block* inlined_block, const AotEntrypointProto& entrypoint,
      std::string_view data_layout, JitFunctionType func_ptr);

  // Returns the bytes of an object file containing the compiled XLS function.
  static absl::StatusOr<JitObjectCode> CreateObjectCode(
      const BlockElaboration& elab, int64_t opt_level, bool include_msan,
      JitObserver* obs);

  virtual ~BlockJit() = default;

  // Create a new blank block with no registers or ports set. Can be cycled
  // independently of other blocks/continuations.
  virtual std::unique_ptr<BlockJitContinuation> NewContinuation();

  // Runs a single cycle of a block with the given continuation.
  virtual absl::Status RunOneCycle(BlockJitContinuation& continuation);

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

  bool supports_observer() const { return supports_observer_; }

 protected:
  BlockJit(Block* block, std::unique_ptr<JitRuntime> runtime,
           std::unique_ptr<OrcJit> jit, JittedFunctionBase function,
           bool supports_observer)
      : block_(block),
        runtime_(std::move(runtime)),
        jit_(std::move(jit)),
        function_(std::move(function)),
        supports_observer_(supports_observer) {}

  Block* block_;
  std::unique_ptr<JitRuntime> runtime_;
  std::unique_ptr<OrcJit> jit_;
  JittedFunctionBase function_;
  bool supports_observer_;
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
  virtual ~BlockJitContinuation() = default;
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
  virtual absl::Status SetRegisters(
      const absl::flat_hash_map<std::string, Value>& regs);

  std::vector<Value> GetOutputPorts() const;
  absl::flat_hash_map<std::string, Value> GetOutputPortsMap() const;
  std::vector<Value> GetRegisters() const;
  virtual absl::flat_hash_map<std::string, Value> GetRegistersMap() const;

  absl::flat_hash_map<std::string, int64_t> GetInputPortIndices() const;
  absl::flat_hash_map<std::string, int64_t> GetOutputPortIndices() const;
  virtual absl::flat_hash_map<std::string, int64_t> GetRegisterIndices() const;

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

  absl::Status SetObserver(RuntimeObserver* obs) {
    if (!block_jit_->supports_observer()) {
      return absl::UnimplementedError("runtime observer not supported");
    }
    callbacks_.observer = obs;
    return absl::OkStatus();
  }
  void ClearObserver() { callbacks_.observer = nullptr; }
  RuntimeObserver* observer() const { return callbacks_.observer; }

 protected:
  BlockJitContinuation(Block* block, BlockJit* jit,
                       const JittedFunctionBase& jit_func);

 private:
  using BufferPair = std::array<JitArgumentSet, 2>;
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

// A jit block evaluator that tries to use the jit's register saving as
// possible.
class JitBlockEvaluator : public BlockEvaluator {
 public:
  explicit constexpr JitBlockEvaluator(bool supports_observer = false)
      : BlockEvaluator(supports_observer ? "ObservableJit" : "Jit"),
        supports_observer_(supports_observer) {}
  absl::StatusOr<JitRuntime*> GetRuntime(BlockContinuation* cont) const;

 protected:
  absl::StatusOr<std::unique_ptr<BlockContinuation>> MakeNewContinuation(
      BlockElaboration&& elaboration,
      const absl::flat_hash_map<std::string, Value>& initial_registers)
      const override;

 private:
  bool supports_observer_;
};

inline constexpr JitBlockEvaluator kJitBlockEvaluator(false);
inline constexpr JitBlockEvaluator kObservableJitBlockEvaluator(true);

}  // namespace xls

#endif  // XLS_JIT_BLOCK_JIT_H_
