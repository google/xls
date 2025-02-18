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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_TO_TYPE_INFO_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_TO_TYPE_INFO_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Converts the given `InferenceTable` into a `TypeInfo`, by concretizing the
// types associated with all nodes in the table. This is the final step of type
// inference.
absl::StatusOr<TypeInfo*> InferenceTableToTypeInfo(
    InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    std::unique_ptr<Module> builtins_module);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_TO_TYPE_INFO_H_
