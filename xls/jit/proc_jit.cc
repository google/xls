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

#include "xls/jit/proc_jit.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/jit_runtime.h"

namespace xls {

absl::StatusOr<std::unique_ptr<ProcJit>> ProcJit::Create(
    Proc* proc, JitChannelQueueManager* queue_mgr, RecvFnT recv_fn,
    SendFnT send_fn, int64_t opt_level) {
  auto jit = absl::WrapUnique(new ProcJit(proc));
  XLS_ASSIGN_OR_RETURN(jit->orc_jit_,
                       OrcJit::Create(opt_level, /*emit_object_code=*/false));
  jit->ir_runtime_ = std::make_unique<JitRuntime>(
      jit->orc_jit_->GetDataLayout(), &jit->orc_jit_->GetTypeConverter());
  XLS_ASSIGN_OR_RETURN(
      jit->jitted_function_base_,
      BuildProcFunction(proc, queue_mgr, recv_fn, send_fn, jit->GetOrcJit()));

  // Pre-allocate input, output, and temporary buffers.
  for (const Param* param : proc->params()) {
    jit->input_buffers_.push_back(std::vector<uint8_t>(
        jit->orc_jit_->GetTypeConverter().GetTypeByteSize(param->GetType())));
    jit->output_buffers_.push_back(std::vector<uint8_t>(
        jit->orc_jit_->GetTypeConverter().GetTypeByteSize(param->GetType())));

    jit->input_ptrs_.push_back(jit->input_buffers_.back().data());
    jit->output_ptrs_.push_back(jit->output_buffers_.back().data());
  }
  jit->temp_buffer_.resize(jit->jitted_function_base_.temp_buffer_size);

  return jit;
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> ProcJit::ConvertStateToView(
    absl::Span<const Value> state_value, bool initialize_with_value) {
  std::vector<std::vector<uint8_t>> state_buffers;

  for (int64_t i = 0; i < proc()->GetStateElementCount(); ++i) {
    Type* state_type = proc_->GetStateElementType(i);

    if (!ValueConformsToType(state_value[i], state_type)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected state argument %s (%d) to be of type %s, is: %s",
          proc_->GetStateParam(i)->GetName(), i, state_type->ToString(),
          state_value[i].ToString()));
    }

    state_buffers.push_back(std::vector<uint8_t>(
        orc_jit_->GetTypeConverter().GetTypeByteSize(state_type)));

    if (initialize_with_value) {
      ir_runtime_->BlitValueToBuffer(state_value[i], state_type,
                                     absl::MakeSpan(state_buffers.back()));
    }
  }

  return state_buffers;
}

std::vector<Value> ProcJit::ConvertStateViewToValue(
    absl::Span<uint8_t const* const> state_buffers) {
  std::vector<Value> state_values;
  for (int64_t i = 0; i < proc()->GetStateElementCount(); ++i) {
    Type* state_type = proc_->GetStateElementType(i);
    state_values.push_back(
        ir_runtime_->UnpackBuffer(state_buffers[i], state_type));
  }

  return state_values;
}

absl::Status ProcJit::RunWithViews(absl::Span<const uint8_t* const> state,
                                   absl::Span<uint8_t* const> next_state,
                                   void* user_data) {
  InterpreterEvents events;

  // The JITed function requires an input (and output)for each parameter,
  // including the token. Only the state parameter values are passed in so
  // create a new input (and output) buffer array with a dummy token value at
  // the beginning.
  std::vector<const uint8_t*> inputs;
  inputs.push_back(nullptr);
  inputs.insert(inputs.end(), state.begin(), state.end());

  std::vector<uint8_t*> outputs;
  outputs.push_back(nullptr);
  outputs.insert(outputs.end(), next_state.begin(), next_state.end());

  jitted_function_base_.function(inputs.data(), outputs.data(),
                                 temp_buffer_.data(), &events, user_data,
                                 runtime());

  return absl::OkStatus();
}

absl::StatusOr<InterpreterResult<std::vector<Value>>> ProcJit::Run(
    absl::Span<const Value> state, void* user_data) {
  int64_t state_element_count = proc()->GetStateElementCount();
  if (state.size() != state_element_count) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Arg list to '%s' has the wrong size: %d vs expected %d.",
        proc_->name(), state.size(), state_element_count));
  }

  std::vector<Type*> param_types;
  for (const Param* param : proc_->params()) {
    param_types.push_back(param->GetType());
  }

  std::vector<Value> args;
  args.push_back(Value::Token());
  args.insert(args.end(), state.begin(), state.end());
  XLS_RETURN_IF_ERROR(
      ir_runtime_->PackArgs(args, param_types, absl::MakeSpan(input_ptrs_)));

  InterpreterEvents events;

  jitted_function_base_.function(input_ptrs_.data(), output_ptrs_.data(),
                                 temp_buffer_.data(), &events, user_data,
                                 runtime());

  std::vector<Value> next_state;
  for (int64_t i = 0; i < proc_->GetStateElementCount(); ++i) {
    Value result = ir_runtime_->UnpackBuffer(output_ptrs_[i + 1],
                                             proc_->GetStateElementType(i));
    next_state.push_back(result);
  }
  return InterpreterResult<std::vector<Value>>{std::move(next_state),
                                               std::move(events)};
}

}  // namespace xls
