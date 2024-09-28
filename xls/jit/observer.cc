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

#include "xls/jit/observer.h"

#include <cstdint>
#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/jit/jit_runtime.h"

namespace xls {

CompoundJitObserver::CompoundJitObserver(
    absl::Span<JitObserver* const> observers)
    : observers_(observers.begin(), observers.end()) {}

JitObserverRequests CompoundJitObserver::GetNotificationOptions() const {
  return JitObserverRequests{
      .unoptimized_module = absl::c_any_of(
          observers_,
          [](auto* o) {
            return o->GetNotificationOptions().unoptimized_module;
          }),
      .optimized_module = absl::c_any_of(
          observers_,
          [](auto* o) { return o->GetNotificationOptions().optimized_module; }),
      .assembly_code_str =
          absl::c_any_of(observers_,
                         [](auto* o) {
                           return o->GetNotificationOptions().assembly_code_str;
                         }),
  };
}
void CompoundJitObserver::UnoptimizedModule(const llvm::Module* module) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().unoptimized_module) {
      o->UnoptimizedModule(module);
    }
  }
}
void CompoundJitObserver::OptimizedModule(const llvm::Module* module) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().optimized_module) {
      o->OptimizedModule(module);
    }
  }
}
void CompoundJitObserver::AssemblyCodeString(const llvm::Module* module,
                                             std::string_view asm_code) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().assembly_code_str) {
      o->AssemblyCodeString(module, asm_code);
    }
  }
}

void CompoundJitObserver::AddObserver(JitObserver* o) {
  observers_.push_back(o);
}

void RuntimeEvaluationObserver::RecordNodeValue(int64_t node_ptr,
                                                const uint8_t* data) {
  Node* node = to_node_(node_ptr);
  this->NodeEvaluated(node, runtime_->UnpackBuffer(data, node->GetType()));
}

}  // namespace xls
