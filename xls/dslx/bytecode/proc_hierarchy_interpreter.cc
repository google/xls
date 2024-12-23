// Copyright 2021 The XLS Authors
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

#include "xls/dslx/bytecode/proc_hierarchy_interpreter.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

// Specialization of BytecodeInterpreter for executing Proc `config` functions.
// These are special b/c they define a tree of ProcInstances that we need to
// collect at the end so we can "tick" them. Only this class, unlike
// BytecodeInterpreter, can process `spawn` nodes.
class ProcConfigBytecodeInterpreter : public BytecodeInterpreter {
 public:
  ~ProcConfigBytecodeInterpreter() override = default;

  // Implementation of Spawn handling common to both ElaborateProcHierarchy and
  // EvalSpawn. `next_args` should not include Proc members or the obligatory
  // Token; they're added to the arg list internally.
  static absl::Status EvalSpawn(
      ImportData* import_data, ProcIdFactory* proc_id_factory,
      const std::optional<ProcId>& caller_proc_id, const TypeInfo* type_info,
      const std::optional<ParametricEnv>& caller_bindings,
      const std::optional<ParametricEnv>& callee_bindings,
      std::optional<Bytecode::SpawnFunctions> spawn_functions, Proc* proc,
      absl::Span<const InterpValue> config_args,
      ProcHierarchyInterpreter* hierarchy_interpreter,
      const BytecodeInterpreterOptions& options = BytecodeInterpreterOptions());

 private:
  ProcConfigBytecodeInterpreter(ImportData* import_data,
                                ProcIdFactory* proc_id_factory,
                                const std::optional<ProcId>& proc_id,
                                ProcHierarchyInterpreter* hierarchy_interpreter,
                                const BytecodeInterpreterOptions& options);

  absl::Status EvalSpawn(const Bytecode& bytecode) override;

  ProcHierarchyInterpreter* hierarchy_interpreter_;
  ProcIdFactory* proc_id_factory_;
};

ProcConfigBytecodeInterpreter::ProcConfigBytecodeInterpreter(
    ImportData* import_data, ProcIdFactory* proc_id_factory,
    const std::optional<ProcId>& proc_id,
    ProcHierarchyInterpreter* hierarchy_interpreter,
    const BytecodeInterpreterOptions& options)
    : BytecodeInterpreter(import_data, proc_id,
                          /*channel_manager=*/nullptr, options),
      hierarchy_interpreter_(hierarchy_interpreter),
      proc_id_factory_(proc_id_factory) {}

absl::Status ProcConfigBytecodeInterpreter::EvalSpawn(
    const Bytecode& bytecode) {
  Frame& frame = frames().back();
  XLS_RET_CHECK(proc_id().has_value());
  XLS_ASSIGN_OR_RETURN(const Bytecode::SpawnData* spawn_data,
                       bytecode.spawn_data());
  XLS_ASSIGN_OR_RETURN(
      std::vector<InterpValue> config_args,
      PopArgsRightToLeft(spawn_data->spawn_functions().config->args().size()));
  return EvalSpawn(import_data(), proc_id_factory_,
                   /*caller_proc_id=*/proc_id(), frame.type_info(),
                   spawn_data->caller_bindings(), spawn_data->callee_bindings(),
                   spawn_data->spawn_functions(), spawn_data->proc(),
                   config_args, hierarchy_interpreter_, options());
}

/* static */ absl::Status ProcConfigBytecodeInterpreter::EvalSpawn(
    ImportData* import_data, ProcIdFactory* proc_id_factory,
    const std::optional<ProcId>& caller_proc_id, const TypeInfo* type_info,
    const std::optional<ParametricEnv>& caller_bindings,
    const std::optional<ParametricEnv>& callee_bindings,
    std::optional<Bytecode::SpawnFunctions> spawn_functions, Proc* proc,
    absl::Span<const InterpValue> config_args,
    ProcHierarchyInterpreter* hierarchy_interpreter,
    const BytecodeInterpreterOptions& options) {
  const TypeInfo* parent_ti = type_info;

  const ParametricEnv& actual_caller_bindings =
      caller_bindings.has_value() ? caller_bindings.value() : ParametricEnv();

  auto get_parametric_type_info =
      [type_info, actual_caller_bindings](
          const Invocation* invoc) -> absl::StatusOr<TypeInfo*> {
    std::optional<TypeInfo*> maybe_type_info =
        type_info->GetInvocationTypeInfo(invoc, actual_caller_bindings);
    if (!maybe_type_info.has_value()) {
      return absl::InternalError(absl::StrFormat(
          "ProcConfigBytecodeInterpreter::EvalSpawn; could not find type info "
          "for invocation `%s` caller_bindings: %s",
          invoc->ToString(), actual_caller_bindings.ToString()));
    }
    return maybe_type_info.value();
  };

  // We need to get a new TI if there's a spawn, i.e., this isn't a top-level
  // proc instantiation, to avoid constexpr values from colliding between
  // different proc instantiations.
  if (spawn_functions.has_value()) {
    // We're guaranteed that these have values if the proc is parametric (the
    // root proc can't be parametric).
    XLS_ASSIGN_OR_RETURN(type_info,
                         get_parametric_type_info(spawn_functions->config));
  }

  auto channel_instance_allocator = [&]() -> int64_t {
    return hierarchy_interpreter->channel_manager().AllocateChannel();
  };
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> config_bf,
      BytecodeEmitter::EmitProcConfig(
          import_data, type_info, proc->config(), callee_bindings,
          channel_instance_allocator,
          BytecodeEmitterOptions{.format_preference =
                                     options.format_preference()}));

  ProcId callee_proc_id = proc_id_factory->CreateProcId(caller_proc_id, proc);
  ProcConfigBytecodeInterpreter cbi(import_data, proc_id_factory,
                                    callee_proc_id, hierarchy_interpreter,
                                    options);
  XLS_RETURN_IF_ERROR(cbi.InitFrame(config_bf.get(), config_args, type_info));
  XLS_RETURN_IF_ERROR(cbi.Run());
  XLS_RET_CHECK_EQ(cbi.stack().size(), 1);
  InterpValue proc_member_tuple = cbi.stack().PeekOrDie();
  XLS_RET_CHECK(proc_member_tuple.IsTuple());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* proc_members,
                       proc_member_tuple.GetValues());

  InterpValue initial_state(InterpValue::MakeToken());
  if (spawn_functions.has_value()) {
    XLS_ASSIGN_OR_RETURN(initial_state, parent_ti->GetConstExpr(
                                            spawn_functions->next->args()[0]));
  } else {
    // If this is the top-level proc, then we can get its initial state from the
    // ModuleMember typechecking, since A) top-level procs can't be
    // parameterized and B) typechecking will eagerly constexpr evaluate init
    // functions.
    XLS_ASSIGN_OR_RETURN(initial_state,
                         parent_ti->GetConstExpr(proc->init().body()));
  }

  std::vector<NameDef*> member_defs;
  member_defs.reserve(proc->members().size());
  for (const ProcMember* param : proc->members()) {
    member_defs.push_back(param->name_def());
  }

  if (spawn_functions.has_value()) {
    XLS_ASSIGN_OR_RETURN(type_info,
                         get_parametric_type_info(spawn_functions->next));
  }

  std::vector<InterpValue> full_next_args = *proc_members;
  full_next_args.push_back(initial_state);

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> next_bf,
      BytecodeEmitter::EmitProcNext(
          import_data, type_info, proc->next(), callee_bindings, member_defs,
          BytecodeEmitterOptions{.format_preference =
                                     options.format_preference()}));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeInterpreter> next_interpreter,
      CreateUnique(import_data, callee_proc_id, next_bf.get(), full_next_args,
                   &hierarchy_interpreter->channel_manager(), options));
  hierarchy_interpreter->AddProcInstance(
      ProcInstance{proc, std::move(next_interpreter), std::move(next_bf),
                   *proc_members, initial_state, type_info});
  return absl::OkStatus();
}

}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<ProcHierarchyInterpreter>>
ProcHierarchyInterpreter::Create(ImportData* import_data, TypeInfo* type_info,
                                 Proc* top_proc,
                                 const BytecodeInterpreterOptions& options) {
  ProcIdFactory proc_id_factory;
  auto hierarchy_interpreter = std::make_unique<ProcHierarchyInterpreter>();

  // Allocate the channels for the top-level config interface.
  for (int64_t index = 0; index < top_proc->config().params().size(); ++index) {
    dslx::Param* param = top_proc->config().params().at(index);
    std::optional<Type*> param_type = type_info->GetItem(param);
    XLS_RET_CHECK(param_type.has_value());
    // Currently, only channels are supported as parameters to the config
    // function of a proc.
    dslx::ChannelType* channel_type =
        dynamic_cast<dslx::ChannelType*>(param_type.value());
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }

    XLS_RETURN_IF_ERROR(hierarchy_interpreter->AddInterfaceChannel(
        param->identifier(), param_type.value(),
        &channel_type->payload_type()));
  }

  XLS_RETURN_IF_ERROR(ProcConfigBytecodeInterpreter::EvalSpawn(
      import_data, &proc_id_factory,
      /*caller_proc_id=*/std::nullopt, type_info,
      /*caller_bindings=*/std::nullopt,
      /*callee_bindings=*/std::nullopt,
      /*spawn_functions=*/std::nullopt, top_proc,
      /*config_args=*/hierarchy_interpreter->InterfaceArgs(),
      hierarchy_interpreter.get(), options));
  return std::move(hierarchy_interpreter);
}

absl::StatusOr<ProcRunResult> ProcInstance::Run() {
  bool progress_made = false;
  absl::Status result_status = interpreter_->Run(&progress_made);

  if (result_status.ok()) {
    // The proc completed a tick. Update the state.
    state_ = interpreter_->stack().PeekOrDie();
    std::vector<InterpValue> full_next_args = proc_members_;
    full_next_args.push_back(state_);
    XLS_RETURN_IF_ERROR(
        interpreter_->InitFrame(next_fn_.get(), full_next_args, type_info_));
    return ProcRunResult{.execution_state = ProcExecutionState::kCompleted,
                         .blocked_channel_info = std::nullopt,
                         .progress_made = progress_made};
  }

  if (result_status.code() == absl::StatusCode::kUnavailable) {
    // Empty recv channel. Just return Ok and we'll try again next time.
    return ProcRunResult{
        .execution_state = ProcExecutionState::kBlockedOnReceive,
        .blocked_channel_info = interpreter_->blocked_channel_info(),
        .progress_made = progress_made};
  }

  return result_status;
}

absl::Span<ProcInstance> ProcHierarchyInterpreter::proc_instances() {
  return absl::MakeSpan(proc_instances_);
}

absl::Status ProcHierarchyInterpreter::AddInterfaceChannel(
    std::string_view name, const Type* arg_type, const Type* payload_type) {
  const ChannelType* channel_type = dynamic_cast<const ChannelType*>(arg_type);
  if (channel_type == nullptr) {
    return absl::UnimplementedError(absl::StrFormat(
        "DSLX interpreter does not support arrays of channels on the interface "
        "of the top-level proc. Channel type of argument `%s`: %s",
        name, arg_type->ToString()));
  }
  auto channel_instance_allocator = [&]() -> int64_t {
    return channel_manager_.AllocateChannel();
  };

  XLS_ASSIGN_OR_RETURN(
      InterpValue channel_reference,
      CreateChannelReference(channel_type->direction(), channel_type,
                             channel_instance_allocator));
  interface_args_.push_back(channel_reference);
  interface_channels_.push_back(InterfaceChannel{
      .name = std::string{name},
      .direction = channel_type->direction(),
      .payload_type = &channel_type->payload_type(),
      .channel = &channel_manager_.GetChannel(channel_manager_.size() - 1)});
  return absl::OkStatus();
}

void ProcHierarchyInterpreter::AddProcInstance(ProcInstance&& proc_instance) {
  proc_instances_.push_back(std::move(proc_instance));
}

absl::Status ProcHierarchyInterpreter::Tick() {
  std::deque<ProcInstance*> ready_list;
  for (auto& p : proc_instances()) {
    ready_list.push_back(&p);
  }

  std::deque<ProcInstance*> next_ready_list;
  bool progress_made;
  do {
    progress_made = false;
    while (!ready_list.empty()) {
      ProcInstance* p = ready_list.front();
      ready_list.pop_front();

      XLS_ASSIGN_OR_RETURN(ProcRunResult run_result, p->Run());
      if (run_result.execution_state == ProcExecutionState::kBlockedOnReceive) {
        next_ready_list.push_back(p);
        progress_made |= run_result.progress_made;
      }
    }
    ready_list = std::move(next_ready_list);
    next_ready_list.clear();
  } while (progress_made);

  return absl::OkStatus();
}

absl::StatusOr<int64_t> ProcHierarchyInterpreter::TickUntilOutput(
    absl::Span<const std::pair<std::string, int64_t>> output_counts) {
  std::vector<InterpValueChannel*> output_channels;
  std::vector<int64_t> expected_output_counts;
  for (const auto& [arg_name, count] : output_counts) {
    bool found = false;
    for (int64_t i = 0; i < interface_channels_.size(); ++i) {
      const InterfaceChannel& channel = interface_channels_[i];
      if (channel.direction != ChannelDirection::kOut) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Channel `%s` is not an output channel", arg_name));
      }
      if (channel.name == arg_name) {
        output_channels.push_back(channel.channel);
        expected_output_counts.push_back(count);
        found = true;
        break;
      }
    }
    if (!found) {
      return absl::InvalidArgumentError(
          absl::StrFormat("No such interface argument `%s`", arg_name));
    }
  }

  auto is_done = [&]() {
    for (int64_t i = 0; i < output_channels.size(); ++i) {
      if (output_channels[i]->GetSize() < expected_output_counts[i]) {
        return false;
      }
    }
    return true;
  };

  const BytecodeInterpreterOptions& options =
      proc_instances().front().options();
  int64_t tick_count = 0;
  while (!is_done()) {
    bool progress_made = false;
    if (options.max_ticks().has_value() &&
        tick_count > options.max_ticks().value()) {
      return absl::DeadlineExceededError(
          absl::StrFormat("Exceeded limit of %d proc ticks before terminating",
                          options.max_ticks().value()));
    }
    std::vector<std::string> blocked_channels;
    for (auto& p : proc_instances()) {
      XLS_ASSIGN_OR_RETURN(ProcRunResult run_result, p.Run());
      if (run_result.execution_state == ProcExecutionState::kBlockedOnReceive) {
        XLS_RET_CHECK(run_result.blocked_channel_info.has_value());
        BlockedChannelInfo channel_info =
            run_result.blocked_channel_info.value();
        blocked_channels.push_back(absl::StrFormat(
            "%s: proc `%s` is blocked on receive on channel `%s`",
            channel_info.span.ToString(p.interpreter().file_table()),
            p.proc()->identifier(), channel_info.name));
      }
      progress_made |= run_result.progress_made;
    }

    if (!progress_made) {
      return absl::DeadlineExceededError(absl::StrFormat(
          "Procs are deadlocked:\n%s", absl::StrJoin(blocked_channels, "\n")));
    }
    ++tick_count;
  }
  return tick_count;
}

}  // namespace xls::dslx
