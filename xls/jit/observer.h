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

#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_runtime.h"

// Forward declare llvm::Module so we don't need to link it.
namespace llvm {
class Module;
}  // namespace llvm

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
class CompoundJitObserver final : public JitObserver {
 public:
  explicit CompoundJitObserver(absl::Span<JitObserver* const> observers);

  JitObserverRequests GetNotificationOptions() const final;
  void UnoptimizedModule(const llvm::Module* module) final;
  void OptimizedModule(const llvm::Module* module) final;
  void AssemblyCodeString(const llvm::Module* module,
                          std::string_view asm_code) final;

  void AddObserver(JitObserver* o);

 private:
  std::vector<JitObserver*> observers_;
};

// An observer that is given a node-pointer (frozen at the time of jit) and the
// jit-encoded value that node takes.
class RuntimeObserver {
 public:
  virtual ~RuntimeObserver() = default;
  virtual void RecordNodeValue(int64_t node_ptr, const uint8_t* data) = 0;
};

// A translator that lets one easily convert from a jit runtime observer to the
// interpreter observer.
class RuntimeEvaluationObserver : public RuntimeObserver,
                                  public EvaluationObserver {
 public:
  RuntimeEvaluationObserver(std::function<Node*(int64_t)> to_node,
                            JitRuntime* runtime)
      : to_node_(std::move(to_node)), runtime_(runtime) {}
  void RecordNodeValue(int64_t node_ptr, const uint8_t* data) final;

 private:
  std::function<Node*(int64_t)> to_node_;
  JitRuntime* runtime_;
};

class RuntimeEvaluationObserverAdapter final
    : public RuntimeEvaluationObserver {
 public:
  RuntimeEvaluationObserverAdapter(EvaluationObserver* obs,
                                   const std::function<Node*(int64_t)>& to_node,
                                   JitRuntime* runtime)
      : RuntimeEvaluationObserver(to_node, runtime), real_(obs) {}

  std::optional<RuntimeObserver*> AsRawObserver() final { return this; }

  void NodeEvaluated(Node* n, const Value& v) final {
    real_->NodeEvaluated(n, v);
  }

 private:
  EvaluationObserver* real_;
};

}  // namespace xls

#endif  // XLS_JIT_OBSERVER_H_
