// Copyright 2020 The XLS Authors
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

#include "xls/jit/function_jit.h"

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/jit_runtime.h"

namespace xls {

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::Create(
    Function* xls_function, int64_t opt_level) {
  return CreateInternal(xls_function, opt_level, /*emit_object_code=*/false);
}

absl::StatusOr<JitObjectCode> FunctionJit::CreateObjectCode(
    Function* xls_function, int64_t opt_level) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<FunctionJit> jit,
      CreateInternal(xls_function, opt_level, /*emit_object_code=*/true));
  return JitObjectCode{
      .function_name = std::string{jit->GetJittedFunctionName()},
      .object_code = jit->orc_jit_->GetObjectCode(),
      .parameter_buffer_sizes = jit->jitted_function_base_.input_buffer_sizes,
      .return_buffer_size = jit->jitted_function_base_.output_buffer_sizes[0],
      .temp_buffer_size = jit->GetTempBufferSize(),
  };
}

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::CreateInternal(
    Function* xls_function, int64_t opt_level, bool emit_object_code) {
  auto jit = absl::WrapUnique(new FunctionJit(xls_function));
  XLS_ASSIGN_OR_RETURN(jit->orc_jit_,
                       OrcJit::Create(opt_level, emit_object_code));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       OrcJit::CreateDataLayout());
  jit->jit_runtime_ = std::make_unique<JitRuntime>(data_layout);
  XLS_ASSIGN_OR_RETURN(jit->jitted_function_base_,
                       BuildFunction(xls_function, *jit->orc_jit_));

  // Pre-allocate argument, result, and temporary buffers.
  for (int64_t i = 0; i < xls_function->params().size(); ++i) {
    jit->arg_buffers_.push_back(std::vector<uint8_t>(jit->GetArgTypeSize(i)));
    jit->arg_buffer_ptrs_.push_back(jit->arg_buffers_.back().data());
  }
  jit->result_buffer_.resize(jit->GetReturnTypeSize());
  jit->temp_buffer_.resize(jit->GetTempBufferSize());

  return jit;
}

absl::StatusOr<InterpreterResult<Value>> FunctionJit::Run(
    absl::Span<const Value> args) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Arg list to '%s' has the wrong size: %d vs expected %d.",
        xls_function_->name(), args.size(), xls_function_->params().size()));
  }

  for (int i = 0; i < params.size(); i++) {
    if (!ValueConformsToType(args[i], params[i]->GetType())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Got argument %s for parameter %d which is not of type %s",
          args[i].ToString(), i, params[i]->GetType()->ToString()));
    }
  }

  std::vector<Type*> param_types;
  for (const Param* param : xls_function_->params()) {
    param_types.push_back(param->GetType());
  }

  // Allocate argument buffers and copy in arg Values.
  XLS_RETURN_IF_ERROR(jit_runtime_->PackArgs(args, param_types,
                                             absl::MakeSpan(arg_buffer_ptrs_)));

  InterpreterEvents events;
  InvokeJitFunction(arg_buffer_ptrs_, result_buffer_.data(), &events);
  Value result = jit_runtime_->UnpackBuffer(
      result_buffer_.data(), xls_function_->return_value()->GetType());

  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

absl::StatusOr<InterpreterResult<Value>> FunctionJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args);
}

absl::Status FunctionJit::RunWithViews(absl::Span<uint8_t* const> args,
                                       absl::Span<uint8_t> result_buffer,
                                       InterpreterEvents* events) {
  absl::Span<Param* const> params = xls_function_->params();
  if (args.size() != params.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Arg list has the wrong size: %d vs expected %d.",
                        args.size(), xls_function_->params().size()));
  }

  if (result_buffer.size() < GetReturnTypeSize()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Result buffer too small - must be at least %d bytes!",
                     GetReturnTypeSize()));
  }

  InvokeJitFunction(args, result_buffer.data(), events);
  return absl::OkStatus();
}

void FunctionJit::InvokeJitFunction(absl::Span<uint8_t* const> arg_buffers,
                                    uint8_t* output_buffer,
                                    InterpreterEvents* events) {
  uint8_t* output_buffers[1] = {output_buffer};
  jitted_function_base_.function(
      arg_buffers.data(), output_buffers, temp_buffer_.data(), events,
      /*user_data=*/nullptr, runtime(), /*continuation_point=*/0);
}

}  // namespace xls
