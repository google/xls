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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_CONSTANT_COLLECTOR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_CONSTANT_COLLECTOR_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// An object that collects constant values (as well as compile-time unrolled
// constructs) into `TypeInfo` during the process of generating the `TypeInfo`
// for a module. This is a late step in the type checking of a particular node,
// performed after its concrete type has been determined and validated.
class ConstantCollector {
 public:
  virtual ~ConstantCollector() = default;

  // Collects any constexpr values from the given `node`, whose concrete type
  // has been determined to be `type`, and stores them in `ti`.
  virtual absl::Status CollectConstants(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, const Type& type, TypeInfo* ti) = 0;
};

// Creates a `ConstantCollector` instance bound to the given dependencies, which
// is intended to be reused during the entire type checking procedure for a
// module (though currently the collector is stateless).
std::unique_ptr<ConstantCollector> CreateConstantCollector(
    InferenceTable& table, Module& module, ImportData& import_data,
    WarningCollector& warning_collector, const FileTable& file_table,
    InferenceTableConverter& converter, Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    TypeSystemTracer& tracer);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_CONSTANT_COLLECTOR_H_
