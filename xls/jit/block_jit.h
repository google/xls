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

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/ir/block.h"
#include "xls/ir/block_elaboration.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"
#include "xls/jit/orc_jit.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {

class BlockJitContinuation;
class BlockJit {
 public:
  struct InterfaceMetadata {
    std::string block_name;

    // Owns types in input/output/register type vectors.
    TypeManager type_manager;
    std::vector<std::string> input_port_names;
    std::vector<Type*> input_port_types;
    std::vector<std::string> output_port_names;
    std::vector<Type*> output_port_types;
    std::vector<std::string> register_names;
    std::vector<Type*> register_types;

    static absl::StatusOr<InterfaceMetadata> CreateFromBlock(Block* block);
    static absl::StatusOr<InterfaceMetadata> CreateFromAotEntrypoint(
        const AotEntrypointProto& entrypoint);

    int64_t InputPortCount() const { return input_port_names.size(); }
    int64_t OutputPortCount() const { return output_port_names.size(); }
    int64_t RegisterCount() const { return register_names.size(); }
  };

  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(
      Block* block, bool support_observer_callbacks = false);
  static absl::StatusOr<std::unique_ptr<BlockJit>> Create(
      const BlockElaboration& elab, bool support_observer_callbacks = false);

  static absl::StatusOr<std::unique_ptr<BlockJit>> CreateFromAot(
      const AotEntrypointProto& entrypoint, std::string_view data_layout,
      JitFunctionType func_ptr);

  // Returns the bytes of an object file containing the compiled XLS function.
  static absl::StatusOr<JitObjectCode> CreateObjectCode(
      const BlockElaboration& elab, int64_t opt_level, bool include_msan,
      JitObserver* obs, std::string_view symbol_salt = "");

  virtual ~BlockJit() = default;

  // Create a new blank block with no registers or ports set. Can be cycled
  // independently of other blocks/continuations.
  virtual std::unique_ptr<BlockJitContinuation> NewContinuation(
      BlockEvaluator::OutputPortSampleTime sample_time);

  // Runs a single cycle of a block with the given continuation.
  virtual absl::Status RunOneCycle(BlockJitContinuation& continuation);

  OrcJit& orc_jit() const { return *jit_; }

  JitRuntime* runtime() const { return runtime_.get(); }

  // Get metadata about the buffers for the input ports.
  absl::Span<const TypeBufferMetadata> GetInputPortBufferMetadata() const {
    return absl::MakeConstSpan(function_.GetInputBufferMetadata())
        .subspan(0, metadata_.InputPortCount());
  }

  // Get metadata about the buffers for the registers. This metadata is used for
  // the input and output register values.
  absl::Span<const TypeBufferMetadata> GetRegisterBufferMetadata() const {
    return absl::MakeConstSpan(function_.GetInputBufferMetadata())
        .subspan(metadata_.InputPortCount());
  }

  // Get metadata about the buffers for the output ports.
  absl::Span<const TypeBufferMetadata> GetOutputPortBufferMetadata() const {
    return absl::MakeConstSpan(function_.GetOutputBufferMetadata())
        .subspan(0, metadata_.OutputPortCount());
  }

  // Get metadata about the buffers for the "extra" register writes. These
  // register writes are those writes beyond the first register write for a
  // register.
  absl::Span<const TypeBufferMetadata> GetExtraRegisterWriteBufferMetadata()
      const {
    return absl::MakeConstSpan(function_.GetOutputBufferMetadata())
        .subspan(metadata_.OutputPortCount() + metadata_.RegisterCount());
  }

  bool supports_observer() const { return supports_observer_; }

 protected:
  BlockJit(InterfaceMetadata&& metadata, std::unique_ptr<JitRuntime>&& runtime,
           std::unique_ptr<OrcJit>&& jit, JittedFunctionBase&& function,
           bool supports_observer)
      : metadata_(std::move(metadata)),
        runtime_(std::move(runtime)),
        jit_(std::move(jit)),
        function_(std::move(function)),
        supports_observer_(supports_observer) {}

  // A register may have multiple writes. Reconcile the potentially multiple
  // writes and copy the value from the active register write into the write
  // place in the output buffers. Returns an error if multiple register writes
  // are active for a single register.
  absl::Status ReconcileMultipleRegisterWrites(
      BlockJitContinuation& continuation);

  InterfaceMetadata metadata_;
  std::unique_ptr<JitRuntime> runtime_;
  std::unique_ptr<OrcJit> jit_;
  JittedFunctionBase function_;
  bool supports_observer_;
};

class BlockJitContinuation {
 public:
  // Use the same time step enum as interpreter.
  using OutputPortSampleTime = BlockEvaluator::OutputPortSampleTime;
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
    return input_arg_set().get_element_pointers().subspan(
        0, /*len=*/metadata_.InputPortCount());
  }
  // Gets pointers to the JIT ABI struct pointers for each register.
  // Write to the pointed-to memory to manually set a register.
  absl::Span<uint8_t* const> register_pointers() const {
    // Previous register values got swapped over to the inputs.
    // Registers follow the input-ports in the input vector.
    return input_arg_set().get_element_pointers().subspan(
        metadata_.InputPortCount());
  }
  // Gets the pointers to the JIT ABI output pointers for each output port.
  absl::Span<uint8_t const* const> output_port_pointers() const {
    switch (sample_time_) {
      case OutputPortSampleTime::kAtLastPosEdgeClock:
        return output_arg_set().get_element_pointers().subspan(
            0,
            /*len=*/metadata_.OutputPortCount());
      case OutputPortSampleTime::kAfterLastClock:
        return after_last_clock_output_set_->get_element_pointers().subspan(
            0, metadata_.OutputPortCount());
    }
    LOG(FATAL) << "unknown sample type.";
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

  OutputPortSampleTime sample_time() const { return sample_time_; }

 protected:
  BlockJitContinuation(const BlockJit::InterfaceMetadata& metadata,
                       BlockJit* jit, const JittedFunctionBase& jit_func,
                       BlockEvaluator::OutputPortSampleTime sample_time);

 private:
  void SwapRegisters() { arg_set_index_ = arg_set_index_ == 0 ? 1 : 0; }

  const BlockJit::InterfaceMetadata& metadata_;
  BlockJit* block_jit_;

  // At what time in the clock cycle are output ports sampled.
  OutputPortSampleTime sample_time_;

  // Backing buffers for the argument sets. There are two of the register
  // buffers to enable efficient ping-ponging between them. The output register
  // buffer becomes the input on the next cycle and vice versa.
  JitBuffer input_port_buffers_;
  JitBuffer output_port_buffers_;
  std::array<JitBuffer, 2> register_buffers_;
  JitBuffer extra_register_write_buffers_;

  // JitArgumentSets used to pass inputs/outputs to the jitted block. These
  // argument sets do not own the buffers but rather hold pointers into the
  // `*_buffer` JitBuffer fields. Two argument sets are used to enable copy-free
  // passing of the output registers of run to the input registers of the next
  // run. The output register buffers of input_sets_[0] alias the input register
  // buffers of output_sets_[1] and vice versa.
  std::array<JitArgumentSet, 2> input_sets_;
  std::array<JitArgumentSet, 2> output_sets_;

  // Which of the two JitArgumentSets in input_sets_/output_sets should be
  // used. This index alternates between 0 and 1.
  int64_t arg_set_index_ = 0;

  // Buffers for the output ports that are sampled after the rising edge of the
  // clock (OutputPortSampleTime::kAfterLastClock) where newly computed register
  // output values are propagated to output port. Note this includes space for
  // the registers though their values are ignored and they are never read. Only
  // available if OutputPortSampleTime::kAfterLastClock sample time.
  std::unique_ptr<JitArgumentSetOwnedBuffer> after_last_clock_output_set_;

  const JitArgumentSet& input_arg_set() const {
    return input_sets_[arg_set_index_];
  }

  const JitArgumentSet& output_arg_set() const {
    return output_sets_[arg_set_index_];
  }
  JitArgumentSet& output_arg_set() { return output_sets_[arg_set_index_]; }

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
      const absl::flat_hash_map<std::string, Value>& initial_registers,
      BlockEvaluator::OutputPortSampleTime sample_time) const override;

 private:
  bool supports_observer_;
};

inline constexpr JitBlockEvaluator kJitBlockEvaluator(false);
inline constexpr JitBlockEvaluator kObservableJitBlockEvaluator(true);

}  // namespace xls

#endif  // XLS_JIT_BLOCK_JIT_H_
