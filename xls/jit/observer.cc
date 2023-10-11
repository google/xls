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

#include <string_view>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/Module.h"

namespace xls {

CompoundObserver::CompoundObserver(absl::Span<JitObserver* const> observers)
    : observers_(observers.begin(), observers.end()) {}

JitObserverRequests CompoundObserver::GetNotificationOptions() const {
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
void CompoundObserver::UnoptimizedModule(const llvm::Module* module) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().unoptimized_module) {
      o->UnoptimizedModule(module);
    }
  }
}
void CompoundObserver::OptimizedModule(const llvm::Module* module) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().optimized_module) {
      o->OptimizedModule(module);
    }
  }
}
void CompoundObserver::AssemblyCodeString(const llvm::Module* module,
                                          std::string_view asm_code) {
  for (auto* o : observers_) {
    if (o->GetNotificationOptions().assembly_code_str) {
      o->AssemblyCodeString(module, asm_code);
    }
  }
}

void CompoundObserver::AddObserver(JitObserver* o) { observers_.push_back(o); }
}  // namespace xls
