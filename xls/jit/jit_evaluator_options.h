// Copyright 2025 The XLS Authors
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

#ifndef XLS_JIT_JIT_EVALUATOR_OPTIONS_H_
#define XLS_JIT_JIT_EVALUATOR_OPTIONS_H_

#include <cstdint>
#include <string>

#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

namespace xls {

// Evaluator options specific to the JIT.
class JitEvaluatorOptions {
 public:
  JitEvaluatorOptions() = default;

  // The LLVM optimization level.
  JitEvaluatorOptions& set_opt_level(int64_t value) {
    opt_level_ = value;
    return *this;
  }
  int64_t opt_level() const { return opt_level_; }

  // Additional text to append to symbol names to ensure no collisions
  JitEvaluatorOptions& set_symbol_salt(std::string value) {
    symbol_salt_ = std::move(value);
    return *this;
  }
  const std::string& symbol_salt() const { return symbol_salt_; }

  // Whether to include emit observer callbacks in the jitted code.
  JitEvaluatorOptions& set_include_observer_callbacks(bool value) {
    include_observer_callbacks_ = value;
    return *this;
  }
  bool include_observer_callbacks() const {
    return include_observer_callbacks_;
  }

  // Whether to include msan calls in the jitted code. This *must* match
  // the configuration of the binary the jitted code is included in.
  JitEvaluatorOptions& set_include_msan(bool value) {
    include_msan_ = value;
    return *this;
  }
  bool include_msan() const { return include_msan_; }

  JitEvaluatorOptions& set_jit_observer(JitObserver* value) {
    jit_observer_ = value;
    return *this;
  }
  JitObserver* jit_observer() const { return jit_observer_; }

 private:
  int64_t opt_level_ = LlvmCompiler::kDefaultOptLevel;
  std::string symbol_salt_;
  bool include_observer_callbacks_ = false;
  bool include_msan_ = false;
  JitObserver* jit_observer_ = nullptr;
};

}  // namespace xls

#endif  // XLS_JIT_JIT_EVALUATOR_OPTIONS_H_
