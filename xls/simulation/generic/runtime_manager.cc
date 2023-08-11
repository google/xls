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

#include "xls/simulation/generic/runtime_manager.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/package.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/simulation/generic/managed_serial_proc_runtime.h"

namespace xls::simulation::generic {

absl::StatusOr<std::unique_ptr<RuntimeManager>> RuntimeManager::Create(
    Package* package, bool use_jit) {
  absl::StatusOr<std::unique_ptr<SerialProcRuntime>> result;
  if (use_jit) {
    XLS_ASSIGN_OR_RETURN(result, CreateJitSerialProcRuntime(package));
  } else {
    XLS_ASSIGN_OR_RETURN(result, CreateInterpreterSerialProcRuntime(package));
  }
  return absl::WrapUnique(new RuntimeManager(std::move(result).value()));
}

RuntimeManager::RuntimeManager(std::unique_ptr<SerialProcRuntime> runtime) {
  auto base_ptr = runtime.release();
  auto derived_ptr = static_cast<ManagedSerialProcRuntime*>(base_ptr);
  this->runtime_ = std::unique_ptr<ManagedSerialProcRuntime>(derived_ptr);
  this->status_for_singlevalue_manager_ = std::make_unique<RuntimeStatus>();
  this->status_ = this->status_for_singlevalue_manager_.get();
}

bool RuntimeManager::HasDeadlock() const { return this->status_->deadlock_; }

absl::Status RuntimeManager::Reset() {
  this->runtime_->ResetState();
  this->status_->deadlock_ = false;
  return absl::OkStatus();
}

ManagedSerialProcRuntime& RuntimeManager::runtime() { return *this->runtime_; }

RuntimeStatus& RuntimeManager::status() { return *this->status_; }

absl::Status RuntimeManager::Update() {
  auto status = runtime_->SerialProcRuntime::Tick();
  this->status_->deadlock_ = !status.ok();
  return status;
}

}  // namespace xls::simulation::generic
