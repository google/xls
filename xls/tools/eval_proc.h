// Copyright 2026 The XLS Authors
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

#ifndef XLS_TOOLS_EVAL_PROC_H_
#define XLS_TOOLS_EVAL_PROC_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/jit/llvm_compiler.h"

namespace xls {

struct SerialProcRuntimeOptions {
  bool use_jit = false;
  EvaluatorOptions evaluator_options;
  int64_t llvm_opt_level = LlvmCompiler::kDefaultOptLevel;
};

// Creates an interpreter- or JIT-backed `SerialProcRuntime` for `package`,
// supporting both proc-scoped and package-scoped channel elaborations.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateSerialProcRuntime(
    Package* package, const SerialProcRuntimeOptions& options = {});

// Creates an interpreter- or JIT-backed `SerialProcRuntime` rooted at `top`.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateSerialProcRuntime(
    Proc* top, const SerialProcRuntimeOptions& options = {});

// Generates up to `total_ticks` random input values for each receive-only
// boundary channel in `package`.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
GenerateRandomInputsForPackage(Package* package, int64_t total_ticks,
                               absl::BitGenRef rng);

// Attaches random inputs to all receive-only boundary channels on `runtime`:
// single-value channels are populated once, and streaming channels attach a
// generator cycling through `pool_size` pre-sampled random values.
absl::Status AttachRandomInputGenerators(SerialProcRuntime& runtime,
                                         absl::BitGenRef rng,
                                         int64_t pool_size = 64);

// Drains all pending elements from send-only streaming boundary channels on
// `runtime`.
void DrainOutputChannels(SerialProcRuntime& runtime);

}  // namespace xls

#endif  // XLS_TOOLS_EVAL_PROC_H_
