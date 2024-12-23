// Copyright 2024 The XLS Authors
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
#ifndef XLS_DSLX_BYTECODE_PROC_HIERARCHY_INTERPRETER_H_
#define XLS_DSLX_BYTECODE_PROC_HIERARCHY_INTERPRETER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// The execution state that a proc may be left in after a call to
// ProcInstance::Run.
enum class ProcExecutionState : uint8_t {
  // The proc tick completed.
  kCompleted,
  // The proc tick was blocked on a blocking receive.
  kBlockedOnReceive,
};

// Data structure holding the result of a single call to ProcInstance::Run.
struct ProcRunResult {
  ProcExecutionState execution_state;

  // If tick state is kBlockedOnReceive this field holds the name and usage
  // location of the blocked channel.
  std::optional<BlockedChannelInfo> blocked_channel_info;

  // Whether any progress was made (at least one instruction was executed).
  bool progress_made;
};

// A ProcInstance is an instantiation of a Proc.
// ProcInstance : Proc :: Object : Class, roughly.
class ProcInstance {
 public:
  ProcInstance(Proc* proc, std::unique_ptr<BytecodeInterpreter> interpreter,
               std::unique_ptr<BytecodeFunction> next_fn,
               std::vector<InterpValue> proc_members, InterpValue initial_state,
               const TypeInfo* type_info)
      : proc_(proc),
        interpreter_(std::move(interpreter)),
        next_fn_(std::move(next_fn)),
        proc_members_(std::move(proc_members)),
        state_(std::move(initial_state)),
        type_info_(type_info) {}

  // Executes a single "tick" of the ProcInstance. If a tick completes, then the
  // state is updated.
  absl::StatusOr<ProcRunResult> Run();
  Proc* proc() const { return proc_; }
  const BytecodeInterpreterOptions& options() const {
    return interpreter_->options();
  }
  const BytecodeInterpreter& interpreter() const { return *interpreter_; }

 private:
  Proc* proc_;
  std::unique_ptr<BytecodeInterpreter> interpreter_;
  std::unique_ptr<BytecodeFunction> next_fn_;
  std::vector<InterpValue> proc_members_;
  InterpValue state_;
  const TypeInfo* type_info_;
};

// An interpreter which evaluates a hierarchy of procs elaborated from the top
// proc. The abstraction includes interpreters for each proc instantiation and
// InterpValueChannels for each channel instantiation.
class ProcHierarchyInterpreter {
 public:
  static absl::StatusOr<std::unique_ptr<ProcHierarchyInterpreter>> Create(
      ImportData* import_data, TypeInfo* type_info, Proc* top_proc,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions());

  // Execute at most a single iteration of every proc in the hierarchy. Upon
  // return, all procs are either blocked on a receive or have completed a
  // tick. A proc may be blocked and resumed multiple times in a single
  // invocation of `Tick`.
  absl::Status Tick();

  // Tick the proc network until the specified output interface channels have
  // produced at least a specified number of outputs as indicated by
  // `output_counts`. `output_counts` must only contain output channels and need
  // not contain all output channels. Returns the number of ticks executed
  // before the conditions were met. If the maximum number of tick specified by
  // the BytecodeInterpreterOptions used to construct the interpreter is
  // exceeded then an error is returned.
  absl::StatusOr<int64_t> TickUntilOutput(
      absl::Span<const std::pair<std::string, int64_t>> output_counts);

  // Return various properties of the channel arguments of the top proc
  // interface.
  InterpValueChannel& GetInterfaceChannel(int64_t index) {
    InterpValue::ChannelReference channel_reference =
        interface_args_[index].GetChannelReference().value();
    return channel_manager_.GetChannel(
        channel_reference.GetChannelId().value());
  }
  std::string_view GetInterfaceChannelName(int64_t index) const {
    return interface_channels_[index].name;
  }
  ChannelDirection GetInterfaceChannelDirection(int64_t index) const {
    return interface_channels_[index].direction;
  }
  const Type* GetInterfaceChannelPayloadType(int64_t index) const {
    return interface_channels_[index].payload_type;
  }

  // Return the number of channels on the top proc interface.
  int64_t GetInterfaceSize() const { return interface_channels_.size(); }

  absl::Span<const InterpValue> InterfaceArgs() const {
    return interface_args_;
  }

  absl::Span<ProcInstance> proc_instances();
  InterpValueChannelManager& channel_manager() { return channel_manager_; }

  absl::Status AddInterfaceChannel(std::string_view name, const Type* arg_type,
                                   const Type* payload_type);
  void AddProcInstance(ProcInstance&& proc_instance);

 private:
  InterpValueChannelManager channel_manager_;
  std::vector<ProcInstance> proc_instances_;
  std::vector<InterpValue> interface_args_;

  // The interface is defined by the config arguments of the top-level proc.
  struct InterfaceChannel {
    std::string name;
    ChannelDirection direction;
    const Type* payload_type;
    InterpValueChannel* channel;
  };
  std::vector<InterfaceChannel> interface_channels_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_INTERPRETER_H_
