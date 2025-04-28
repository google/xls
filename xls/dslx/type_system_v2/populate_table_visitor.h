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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_POPULATE_TABLE_VISITOR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_POPULATE_TABLE_VISITOR_H_

#include <memory>

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// Populates an InferenceTable starting at a given AST node.
class PopulateTableVisitor {
 public:
  virtual ~PopulateTableVisitor() = default;

  // Populate the InferenceTable with the data from the module.
  virtual absl::Status PopulateFromModule(const Module* module) = 0;
  virtual absl::Status PopulateFromInvocation(const Invocation* invocation) = 0;
};

// Creates a PopulateTableVisitor for the given module and table.
std::unique_ptr<PopulateTableVisitor> CreatePopulateTableVisitor(
    Module* module, InferenceTable* table, ImportData* import_data,
    TypecheckModuleFn typecheck_imported_module);

}  // namespace xls::dslx

#endif
