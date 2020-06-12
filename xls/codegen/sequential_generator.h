// Copyright 2020 Google LLC
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

#ifndef THIRD_PARTY_XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
#define THIRD_PARTY_XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_

#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/vast.h"
#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Generate a pipeline module that implements the loop's body.
xabsl::StatusOr<std::unique_ptr<ModuleGeneratorResult>>
GenerateLoopBodyPipeline(CountedFor* loop, bool use_system_verilog,
                         SchedulingOptions& scheduling_options,
                         const DelayEstimator& = GetStandardDelayEstimator());

// Emits the given function as a verilog module which reuses the same hardware
// over time to executed loop iterations.
xabsl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func);

}  // namespace verilog
}  // namespace xls

#endif  // THIRD_PARTY_XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
