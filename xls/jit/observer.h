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

#ifndef XLS_JIT_OBSERVER_H_
#define XLS_JIT_OBSERVER_H_

#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "llvm/include/llvm/IR/Module.h"

namespace xls {

// All the things an observer can handle. Setting these flags tells users that
// they do not need to call the observer methods. They should be considered
// purely advisory however.
struct JitObserverRequests {
  // Do we want to get called for unoptimized IR code being created.
  bool unoptimized_module = false;
  // Do we want to get called for optimized IR code being created.
  bool optimized_module = false;
  // Do we want to get called with optimized asm code.
  bool assembly_code_str = false;
};

// Basic observer for JIT compilation events
class JitObserver {
 public:
  virtual ~JitObserver() = default;
  // What events we want to get notified about.
  virtual JitObserverRequests GetNotificationOptions() const = 0;
  // Called when a LLVM module has been created and is ready for JITing
  virtual void UnoptimizedModule(const llvm::Module* module) {}
  // Called when a LLVM module has been created and is ready for JITing
  virtual void OptimizedModule(const llvm::Module* module) {}
  // Called when a LLVM module has been compiled with the module code.
  virtual void AssemblyCodeString(const llvm::Module* module,
                                  std::string_view asm_code) {}
};

// A compound observer that lets one trigger multiple observers at once.
class CompoundObserver final : public JitObserver {
 public:
  explicit CompoundObserver(absl::Span<JitObserver* const> observers);

  JitObserverRequests GetNotificationOptions() const final;
  void UnoptimizedModule(const llvm::Module* module) final;
  void OptimizedModule(const llvm::Module* module) final;
  void AssemblyCodeString(const llvm::Module* module,
                          std::string_view asm_code) final;

  void AddObserver(JitObserver* o);

 private:
  std::vector<JitObserver*> observers_;
};

}  // namespace xls

#endif  // XLS_JIT_OBSERVER_H_
