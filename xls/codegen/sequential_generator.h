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
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"

namespace xls {
namespace verilog {

// Emits the given function as a verilog module which reuses the same hardware
// over time to executed loop iterations.
xabsl::StatusOr<ModuleGeneratorResult> ToSequentialModuleText(Function* func);

}  // namespace verilog
}  // namespace xls

#endif  // THIRD_PARTY_XLS_CODEGEN_SEQUENTIAL_GENERATOR_H_
