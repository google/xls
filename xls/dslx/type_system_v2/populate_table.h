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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_POPULATE_TABLE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_POPULATE_TABLE_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Populates the inference table with type variables and annotations.
absl::Status PopulateTable(InferenceTable* table, Module* module,
                           ImportData* import_data, WarningCollector* warnings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_POPULATE_TABLE_H_
