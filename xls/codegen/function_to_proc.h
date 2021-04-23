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

#ifndef XLS_CODEGEN_FUNCTION_TO_PROC_H_
#define XLS_CODEGEN_FUNCTION_TO_PROC_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {

// Converts a function into a combinational (stateless and registerless) proc of
// the given name (the function name is ignored). Function arguments become
// receive nodes over port channels, the function return value becomes a send
// over a port channel. Returns a pointer to the proc.
absl::StatusOr<Proc*> FunctionToProc(Function* f, absl::string_view proc_name);

// Converts a function in a pipelined (stateless) proc. The pipeline is
// constructed using the given schedule. Registers are inserted between each
// stage. Inputs and outputs are not flopped.
absl::StatusOr<Proc*> FunctionToPipelinedProc(const PipelineSchedule& schedule,
                                              Function* f,
                                              absl::string_view proc_name);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_FUNCTION_TO_PROC_H_
