// Copyright 2021 The XLS Authors
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

#ifndef XLS_CODEGEN_CODEGEN_PASS_H_
#define XLS_CODEGEN_CODEGEN_PASS_H_

#include "absl/types/optional.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Defines the pass types for passes involved in lowering and optimizing prior
// to codegen.

// Options passed to each pass.
struct CodegenPassOptions : public PassOptions {
  // Options to use for codegen.
  CodegenOptions codegen_options;

  // Optional schedule. If given, a feedforward pipeline is generated based on
  // the schedule.
  absl::optional<PipelineSchedule> schedule;
};

// Data structure operated on by codegen passes. Contains the IR and associated
// metadata which may be used and mutated by passes.
struct CodegenPassUnit {
  CodegenPassUnit(Package* p, Block* b) : package(p), block(b) {}

  // The package containing IR to lower.
  Package* package;

  // The top-level block to generate a Verilog module for.
  Block* block;

  // The signature is generated (and potentially mutated) during the codegen
  // process.
  // TODO(https://github.com/google/xls/issues/410): 2021/04/27 Consider adding
  // a "block" contruct which corresponds to a verilog module. This block could
  // hold its own signature. This would help prevent the signature from getting
  // out-of-sync with the IR.
  absl::optional<ModuleSignature> signature;

  // These methods are required by CompoundPassBase.
  std::string DumpIr() const;
  const std::string& name() const { return block->name(); }
};

using CodegenPass = PassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenCompoundPass =
    CompoundPassBase<CodegenPassUnit, CodegenPassOptions, PassResults>;
using CodegenInvariantChecker = CodegenCompoundPass::InvariantChecker;

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_PASS_H_
