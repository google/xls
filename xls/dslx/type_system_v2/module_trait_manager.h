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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_MODULE_TRAIT_MANAGER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_MODULE_TRAIT_MANAGER_H_

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/trait_deriver.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"

namespace xls::dslx {

// An object that manages the derivation and resolution of trait functions for a
// particular module. Derived trait function implementations do not reside in
// the AST; they are accessed via the `ModuleTraitManager` of the `StructDef`
// owner module during type inference, and accessed via invocation data in
// `TypeInfo` afterwards.
class ModuleTraitManager {
 public:
  virtual ~ModuleTraitManager() = default;

  // Obtains the derived trait function of the given name for the given struct.
  // This generates the function if it has not been requested before.
  virtual absl::StatusOr<std::optional<Function*>> GetTraitFunction(
      StructDef& struct_def, const StructType& concrete_struct_type,
      std::optional<const ParametricContext*> parametric_struct_context,
      std::string_view function_name) = 0;
};

std::unique_ptr<ModuleTraitManager> CreateModuleTraitManager(
    Module& module, ImportData& import_data, InferenceTable& table,
    std::optional<TraitDeriver*> trait_deriver, TypeSystemTracer& tracer);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_MODULE_TRAIT_MANAGER_H_
