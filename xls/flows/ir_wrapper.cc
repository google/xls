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

#include "xls/flows/ir_wrapper.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/passes/standard_pipeline.h"

namespace xls {
using dslx::Module;
using dslx::TypecheckedModule;

absl::StatusOr<JitChannelQueueWrapper> JitChannelQueueWrapper::Create(
    JitChannelQueue* queue, ProcJit* jit) {
  JitChannelQueueWrapper wrapper;

  wrapper.jit_ = jit;
  wrapper.queue_ = queue;

  int64_t id = queue->channel_id();
  XLS_ASSIGN_OR_RETURN(Channel * ch, jit->proc()->package()->GetChannel(id));
  wrapper.type_ = ch->type();

  int64_t buffer_size = jit->type_converter()->GetTypeByteSize(wrapper.type_);
  wrapper.buffer_.resize(buffer_size);

  return wrapper;
}

absl::Status JitChannelQueueWrapper::Enqueue(const Value& v) {
  jit_->runtime()->BlitValueToBuffer(v, type_, absl::MakeSpan(buffer_));

  queue_->Send(buffer_.data(), buffer_.size());

  return absl::OkStatus();
}

absl::StatusOr<Value> JitChannelQueueWrapper::Dequeue() {
  XLS_RET_CHECK(!queue_->Empty());

  queue_->Recv(buffer_.data(), buffer_.size());

  return jit_->runtime()->UnpackBuffer(buffer_.data(), type_);
}

absl::Status JitChannelQueueWrapper::EnqueueWithUint64(uint64_t v) {
  if (!type_->IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Queue id=%d has non-Bits-typed type: %s",
                        queue_->channel_id(), type_->ToString()));
  }

  if (Bits::MinBitCountUnsigned(v) > type_->AsBitsOrDie()->bit_count()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Value %d for queue id=%d does not fit in type: %s", v,
                        queue_->channel_id(), type_->ToString()));
  }

  Value xls_v(UBits(v, type_->AsBitsOrDie()->bit_count()));

  return Enqueue(xls_v);
}

absl::StatusOr<uint64_t> JitChannelQueueWrapper::DequeueWithUint64() {
  XLS_ASSIGN_OR_RETURN(Value xls_v, Dequeue());

  if (!xls_v.IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Queue id=%d has non-Bits-typed type: %s",
                        queue_->channel_id(), type_->ToString()));
  }

  return xls_v.bits().ToUint64();
}

absl::Status JitChannelQueueWrapper::Enqueue(absl::Span<uint8_t> buffer) {
  queue_->Send(buffer.data(), buffer.size());
  return absl::OkStatus();
}

absl::Status JitChannelQueueWrapper::Dequeue(absl::Span<uint8_t> buffer) {
  queue_->Recv(buffer.data(), buffer.size());
  return absl::OkStatus();
}

absl::StatusOr<DslxModuleAndPath> DslxModuleAndPath::Create(
    absl::string_view module_name, absl::string_view file_path) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<dslx::Module> module,
                       dslx::ParseModuleFromFileAtPath(file_path, module_name));

  return Create(std::move(module), file_path);
}

absl::StatusOr<DslxModuleAndPath> DslxModuleAndPath::Create(
    std::unique_ptr<dslx::Module> module, absl::string_view file_path) {
  DslxModuleAndPath module_and_path;

  module_and_path.TakeDslxModule(std::move(module));
  module_and_path.SetFilePath(file_path);

  return module_and_path;
}

absl::StatusOr<dslx::Module*> IrWrapper::GetDslxModule(
    absl::string_view name) const {
  XLS_RET_CHECK(top_module_ != nullptr);

  if (top_module_->name() == name) {
    return top_module_;
  }

  for (Module* m : other_modules_) {
    XLS_RET_CHECK(m != nullptr);
    if (m->name() == name) {
      return m;
    }
  }

  std::vector<std::string_view> valid_module_names;
  valid_module_names.push_back(top_module_->name());
  for (Module* m : other_modules_) {
    valid_module_names.push_back(m->name());
  }

  return absl::NotFoundError(
      absl::StrFormat("Could not find module with name %s, valid modules [%s]",
                      name, absl::StrJoin(valid_module_names, ", ")));
}

absl::StatusOr<Package*> IrWrapper::GetIrPackage() const {
  XLS_RET_CHECK(package_ != nullptr);
  return package_.get();
}

absl::StatusOr<IrWrapper> IrWrapper::Create(
    absl::string_view ir_package_name, DslxModuleAndPath top_module,
    std::vector<DslxModuleAndPath> import_modules) {
  IrWrapper ir_wrapper(ir_package_name);

  // Compile DSLX
  for (DslxModuleAndPath& module_and_path : import_modules) {
    XLS_RET_CHECK(module_and_path.GetDslxModule() != nullptr);

    XLS_ASSIGN_OR_RETURN(TypecheckedModule module_typechecked,
                         TypecheckModule(module_and_path.GiveUpDslxModule(),
                                         module_and_path.GetFilePath(),
                                         &ir_wrapper.import_data_));

    ir_wrapper.other_modules_.push_back(module_typechecked.module);

    XLS_VLOG_LINES(3, ir_wrapper.other_modules_.back()->ToString());
  }

  XLS_RET_CHECK(top_module.GetDslxModule() != nullptr);
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule top_typechecked,
      TypecheckModule(top_module.GiveUpDslxModule(), top_module.GetFilePath(),
                      &ir_wrapper.import_data_));
  ir_wrapper.top_module_ = top_typechecked.module;
  XLS_VLOG_LINES(3, ir_wrapper.top_module_->ToString());

  // Convert into IR
  const dslx::ConvertOptions convert_options = {
      .emit_positions = true, .emit_fail_as_assert = true, .verify_ir = true};

  XLS_RET_CHECK_OK(dslx::ConvertModuleIntoPackage(
      ir_wrapper.top_module_, &ir_wrapper.import_data_, convert_options,
      /*traverse_tests=*/false, ir_wrapper.package_.get()));

  XLS_VLOG(3) << "IrWrapper Package (pre-opt):";
  XLS_VLOG_LINES(3, ir_wrapper.package_->DumpIr());

  // Optimize IR using default options
  XLS_RETURN_IF_ERROR(
      RunStandardPassPipeline(ir_wrapper.package_.get()).status());

  return std::move(ir_wrapper);
}

absl::StatusOr<IrWrapper> IrWrapper::Create(
    absl::string_view ir_package_name, std::unique_ptr<Module> top_module,
    absl::string_view top_module_path, std::unique_ptr<Module> other_module,
    absl::string_view other_module_path) {
  XLS_ASSIGN_OR_RETURN(
      DslxModuleAndPath top_module_and_path,
      DslxModuleAndPath::Create(std::move(top_module), top_module_path));

  XLS_ASSIGN_OR_RETURN(
      DslxModuleAndPath other_module_and_path,
      DslxModuleAndPath::Create(std::move(other_module), other_module_path));

  std::vector<DslxModuleAndPath> other_module_and_path_vec;
  other_module_and_path_vec.push_back(std::move(other_module_and_path));

  return Create(ir_package_name, std::move(top_module_and_path),
                std::move(other_module_and_path_vec));
}

absl::StatusOr<IrWrapper> IrWrapper::Create(
    absl::string_view ir_package_name, std::unique_ptr<Module> top_module,
    absl::string_view top_module_path,
    absl::Span<std::unique_ptr<Module>> other_modules,
    absl::Span<absl::string_view> other_modules_path) {
  XLS_ASSIGN_OR_RETURN(
      DslxModuleAndPath top_module_and_path,
      DslxModuleAndPath::Create(std::move(top_module), top_module_path));

  std::vector<DslxModuleAndPath> other_modules_and_paths_vec;

  XLS_RET_CHECK(other_modules.size() == other_modules_path.size());
  for (int64_t i = 0; i < other_modules.size(); ++i) {
    XLS_RET_CHECK(other_modules[i] != nullptr);

    XLS_ASSIGN_OR_RETURN(DslxModuleAndPath other_module_and_path,
                         DslxModuleAndPath::Create(std::move(other_modules[i]),
                                                   other_modules_path[i]));

    other_modules_and_paths_vec.push_back(std::move(other_module_and_path));
  }

  return Create(ir_package_name, std::move(top_module_and_path),
                std::move(other_modules_and_paths_vec));
}

absl::StatusOr<Function*> IrWrapper::GetIrFunction(
    absl::string_view name) const {
  XLS_RET_CHECK(top_module_ != nullptr);

  XLS_ASSIGN_OR_RETURN(std::string mangled_name,
                       dslx::MangleDslxName(top_module_->name(), name,
                                            dslx::CallingConvention::kTypical));

  return package_->GetFunction(mangled_name);
}

absl::StatusOr<Proc*> IrWrapper::GetIrProc(absl::string_view name) const {
  XLS_RET_CHECK(top_module_ != nullptr);

  XLS_ASSIGN_OR_RETURN(std::string mangled_name,
                       dslx::MangleDslxName(top_module_->name(), name,
                                            dslx::CallingConvention::kTypical));

  return package_->GetProc(mangled_name);
}

absl::StatusOr<FunctionJit*> IrWrapper::GetAndMaybeCreateFunctionJit(
    absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(Function * f, GetIrFunction(name));

  if (pre_compiled_function_jit_.contains(f)) {
    return pre_compiled_function_jit_.at(f).get();
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<FunctionJit> jit,
                       FunctionJit::Create(f));
  pre_compiled_function_jit_[f] = std::move(jit);

  return pre_compiled_function_jit_[f].get();
}

namespace {

// Recv/Send functions for the JIT proc creation.
bool RecvFn(JitChannelQueue* queue, Receive* recv_ptr, uint8_t* data_ptr,
            int64_t data_sz, void* user_data) {
  return queue->Recv(data_ptr, data_sz);
}

void SendFn(JitChannelQueue* queue, Send* send_ptr, uint8_t* data_ptr,
            int64_t data_sz, void* user_data) {
  queue->Send(data_ptr, data_sz);
}

}  // namespace

absl::StatusOr<ProcJit*> IrWrapper::GetAndMaybeCreateProcJit(
    absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(Proc * p, GetIrProc(name));

  if (pre_compiled_proc_jit_.contains(p)) {
    return pre_compiled_proc_jit_.at(p).get();
  }

  if (jit_channel_manager_ == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        jit_channel_manager_,
        JitChannelQueueManager::CreateThreadUnsafe(package_.get()));
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ProcJit> jit,
      ProcJit::Create(p, jit_channel_manager_.get(), RecvFn, SendFn));
  pre_compiled_proc_jit_[p] = std::move(jit);

  return pre_compiled_proc_jit_[p].get();
}

absl::StatusOr<JitChannelQueue*> IrWrapper::GetJitChannelQueue(
    absl::string_view name) const {
  XLS_ASSIGN_OR_RETURN(Channel * channel, package_->GetChannel(name));
  return jit_channel_manager_->GetQueueById(channel->id());
}

absl::StatusOr<JitChannelQueueWrapper> IrWrapper::CreateJitChannelQueueWrapper(
    absl::string_view name, ProcJit* jit) const {
  XLS_ASSIGN_OR_RETURN(JitChannelQueue * queue, GetJitChannelQueue(name));

  return JitChannelQueueWrapper::Create(queue, jit);
}

}  // namespace xls
