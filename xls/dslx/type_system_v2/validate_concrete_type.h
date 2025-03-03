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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_CONCRETE_TYPE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_CONCRETE_TYPE_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Checks if the given concrete type ultimately makes sense for the given node,
// based on the intrinsic properties of the node, like being an add operation or
// containing an embedded literal.
absl::Status ValidateConcreteType(const AstNode* node, const Type* type,
                                  const TypeInfo& ti,
                                  const TypeAnnotation* annotation,
                                  WarningCollector& warning_collector,
                                  const FileTable& file_table);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_VALIDATE_CONCRETE_TYPE_H_
