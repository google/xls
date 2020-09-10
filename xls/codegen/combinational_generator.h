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

#ifndef XLS_CODEGEN_COMBINATIONAL_GENERATOR_H_
#define XLS_CODEGEN_COMBINATIONAL_GENERATOR_H_

#include <string>

#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/name_to_bit_count.h"
#include "xls/codegen/vast.h"
#include "xls/ir/function.h"

namespace xls {
namespace verilog {

// Emits the given function as a combinational Verilog module. If
// use_system_verilog is true the generated module will be SystemVerilog
// otherwise it will be Verilog.
absl::StatusOr<ModuleGeneratorResult> ToCombinationalModuleText(
    Function* func, bool use_system_verilog = true);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_COMBINATIONAL_GENERATOR_H_
