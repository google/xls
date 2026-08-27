// Copyright 2026 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_TYPE_TO_TYPE_ANNOTATION_H_
#define XLS_DSLX_FRONTEND_TYPE_TO_TYPE_ANNOTATION_H_

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

class TypeInfo;

// Creates a type annotation for the given type in the module.
absl::StatusOr<TypeAnnotation*> CreateTypeAnnotation(
    Module& new_module, const Type& type, const Span& span,
    const Module* source_module = nullptr, const TypeInfo* type_info = nullptr);
}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_TYPE_TO_TYPE_ANNOTATION_H_
