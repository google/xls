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

#ifndef XLS_CODEGEN_SIGNATURE_GENERATOR_H_
#define XLS_CODEGEN_SIGNATURE_GENERATOR_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/ir/function_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

// Generate a ModuleSignature corresponding to a Verilog module generated from
// the given function/proc with the given options/schedule.
// TODO(meheff) 2021/04/27 Replace FunctionBase with block. This function should
// be called only after conversion of a function/proc into a block.
absl::StatusOr<ModuleSignature> GenerateSignature(
    const CodegenOptions& options, FunctionBase* func_base,
    absl::optional<PipelineSchedule> schedule = absl::nullopt);

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_SIGNATURE_GENERATOR_H_
