// Copyright 2022 The XLS Authors
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

#ifndef XLS_INTERPRETER_INTERPRETER_PROC_RUNTIME_H_
#define XLS_INTERPRETER_INTERPRETER_PROC_RUNTIME_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/package.h"

namespace xls {

// Create a SerialProcRuntime composed of ProcInterpreters. Supports old-style
// procs.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
CreateInterpreterSerialProcRuntime(
    Package* package, const EvaluatorOptions& options = EvaluatorOptions());

// Create a SerialProcRuntime composed of ProcInterpreters. Constructed from the
// elaboration of the given proc. Supports new-style procs.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>>
CreateInterpreterSerialProcRuntime(
    Proc* top, const EvaluatorOptions& options = EvaluatorOptions());

}  // namespace xls

#endif  // XLS_INTERPRETER_INTERPRETER_PROC_RUNTIME_H_
