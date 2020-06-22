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

#ifndef XLS_PASSES_STANDARD_PIPELINE_H_
#define XLS_PASSES_STANDARD_PIPELINE_H_

#include "xls/common/status/statusor.h"
#include "xls/passes/passes.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

// CreateStandardPassPipeline connects together the various optimization
// and analysis passes in the order of execution.
std::unique_ptr<CompoundPass> CreateStandardPassPipeline();

// Creates and runs the standard pipeline on the given package with default
// options.
xabsl::StatusOr<bool> RunStandardPassPipeline(Package* package);

// Creates a pipeline for constructing a schedule for a feedforward pipeline.
std::unique_ptr<SchedulingCompoundPass> CreateStandardSchedulingPassPipeline();

}  // namespace xls

#endif  // XLS_PASSES_STANDARD_PIPELINE_H_
