// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPECHECK_MODULE_V2_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPECHECK_MODULE_V2_H_

#include <filesystem>
#include <memory>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/semantics_analysis.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// The top-level type checking function (counterpart to `TypecheckModuleV1`)
// used for modules that have the `kTypeInferenceVersion2` annotation.
absl::StatusOr<std::unique_ptr<ModuleInfo>> TypecheckModuleV2(
    std::unique_ptr<Module> module, std::filesystem::path path,
    ImportData* import_data, WarningCollector* warnings,
    std::unique_ptr<SemanticsAnalysis> semantics_analysis);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPECHECK_MODULE_V2_H_
