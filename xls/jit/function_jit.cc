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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Support/Error.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/aot_compiler.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"
#include "xls/jit/orc_jit.h"

namespace xls {

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::Create(
    Function* xls_function, int64_t opt_level, bool include_observer_callbacks,
    JitObserver* jit_observer) {
  return CreateInternal(xls_function, opt_level, include_observer_callbacks,
                        jit_observer);
}

// Returns an object containing an AOT-compiled version of the specified XLS
// function.
/* static */ absl::StatusOr<std::unique_ptr<FunctionJit>>
FunctionJit::CreateFromAot(Function* xls_function,
                           const AotEntrypointProto& entrypoint,
                           std::string_view data_layout,
                           JitFunctionType function_unpacked,
                           std::optional<JitFunctionType> function_packed) {
  XLS_ASSIGN_OR_RETURN(
      JittedFunctionBase jfb,
      JittedFunctionBase::BuildFromAot(xls_function, entrypoint,
                                       function_unpacked, function_packed));
  llvm::Expected<llvm::DataLayout> layout =
      llvm::DataLayout::parse(data_layout);
  XLS_RET_CHECK(layout) << "Unable to parse '" << data_layout
                        << "' to an llvm data-layout.";
  // OrcJit is simply the arena that holds the JITed code. Since we are already
  // compiled theres no need to create and initialize it.
  // TODO(allight): Ideally we wouldn't even need to link in the llvm stuff if
  // we go down this path, that's a larger refactor however and just carrying
  // around some extra .so's isn't a huge deal.
  return std::unique_ptr<FunctionJit>(new FunctionJit(
      xls_function, std::unique_ptr<OrcJit>(nullptr), std::move(jfb),
      /*has_observer_callbacks=*/false, std::make_unique<JitRuntime>(*layout)));
}

absl::StatusOr<JitObjectCode> FunctionJit::CreateObjectCode(
    Function* xls_function, int64_t opt_level, bool include_msan,
    JitObserver* observer) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<AotCompiler> comp,
                       AotCompiler::Create(include_msan, opt_level, observer));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout, comp->CreateDataLayout());
  XLS_ASSIGN_OR_RETURN(JittedFunctionBase jfb,
                       JittedFunctionBase::Build(xls_function, *comp));
  XLS_ASSIGN_OR_RETURN(auto obj_code, std::move(comp)->GetObjectCode());
  return JitObjectCode{.object_code = std::move(obj_code),
                       .entrypoints =
                           {
                               FunctionEntrypoint{
                                   .function = xls_function,
                                   .jit_info = std::move(jfb),
                               },
                           },
                       .data_layout = data_layout};
}

absl::StatusOr<std::unique_ptr<FunctionJit>> FunctionJit::CreateInternal(
    Function* xls_function, int64_t opt_level, bool has_observer_callbacks,
    JitObserver* jit_observer) {
  XLS_ASSIGN_OR_RETURN(
      auto orc_jit,
      OrcJit::Create(opt_level, has_observer_callbacks, jit_observer));
  XLS_ASSIGN_OR_RETURN(llvm::DataLayout data_layout,
                       orc_jit->CreateDataLayout());
  XLS_ASSIGN_OR_RETURN(auto function_base,
                       JittedFunctionBase::Build(xls_function, *orc_jit));

  return std::unique_ptr<FunctionJit>(new FunctionJit(
      xls_function, std::move(orc_jit), std::move(function_base),
      has_observer_callbacks, std::make_unique<JitRuntime>(data_layout)));
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
  XLS_RETURN_IF_ERROR(
      jit_runtime_->PackArgs(args, param_types, arg_buffers_.pointers()));

  // Call observers with function params.
  if (CurrentRuntimeObserver() != nullptr) {
    for (int64_t i = 0; i < function()->params().size(); ++i) {
      CurrentRuntimeObserver()->RecordNodeValue(
          static_cast<int64_t>(
              reinterpret_cast<intptr_t>(function()->params()[i])),
          arg_buffers_.pointers()[i]);
    }
  }
  InterpreterEvents events;
  jitted_function_base_.RunJittedFunction(
      arg_buffers_, result_buffers_, temp_buffer_, &events,
      /*instance_context=*/&callbacks_, /*jit_runtime=*/runtime(),
      /*continuation_point=*/0);
  Value result = jit_runtime_->UnpackBuffer(
      result_buffers_.pointers()[0], xls_function_->return_value()->GetType());

  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

absl::StatusOr<InterpreterResult<Value>> FunctionJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*xls_function_, kwargs));
  return Run(positional_args);
}

template <bool kForceZeroCopy>
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

  InvokeUnalignedJitFunction<kForceZeroCopy>(args, result_buffer.data(),
                                             events);
  return absl::OkStatus();
}

template absl::Status FunctionJit::RunWithViews</*kForceZeroCopy=*/true>(
    absl::Span<uint8_t* const> args, absl::Span<uint8_t> result_buffer,
    InterpreterEvents* events);
template absl::Status FunctionJit::RunWithViews</*kForceZeroCopy=*/false>(
    absl::Span<uint8_t* const> args, absl::Span<uint8_t> result_buffer,
    InterpreterEvents* events);

template <bool kForceZeroCopy>
void FunctionJit::InvokeUnalignedJitFunction(
    absl::Span<const uint8_t* const> arg_buffers, uint8_t* output_buffer,
    InterpreterEvents* events) {
  uint8_t* output_buffers[1] = {output_buffer};
  jitted_function_base_.RunUnalignedJittedFunction<kForceZeroCopy>(
      arg_buffers.data(), output_buffers, temp_buffer_.get(), events,
      /*instance_context=*/&callbacks_, runtime(), /*continuation=*/0);
}

template void FunctionJit::InvokeUnalignedJitFunction</*kForceZeroCopy=*/false>(
    absl::Span<const uint8_t* const> arg_buffers, uint8_t* output_buffer,
    InterpreterEvents* events);
template void FunctionJit::InvokeUnalignedJitFunction</*kForceZeroCopy=*/true>(
    absl::Span<const uint8_t* const> arg_buffers, uint8_t* output_buffer,
    InterpreterEvents* events);

}  // namespace xls
