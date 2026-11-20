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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_DECLARATION_RESOLUTION_CACHE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_DECLARATION_RESOLUTION_CACHE_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

struct SumConstructorRef {
  SumRef sum_ref;
  const SumVariant* variant;
};

// A successfully resolved alias suffix, independent of the reference that led
// to it. Retain the parameter source and original construction so replaying
// the suffix preserves both diagnostics and inference evidence.
struct ResolvedTypeAlias {
  TypeDefinition definition;
  const TypeRefTypeAnnotation* first_annotation = nullptr;
  const TypeRefTypeAnnotation* parameter_annotation = nullptr;
  const TypeVariableTypeAnnotation* last_type_variable = nullptr;
  std::optional<const Expr*> construction_origin;
};

// Owned by ImportData for the lifetime of its AST corpus. Cache only successful
// concrete declaration resolutions: errors and unresolved generic references
// may become resolvable later. AST clones have new keys and must resolve again;
// these entries contain syntax pointers, not evaluated parametric values.
struct DeclarationResolutionCache {
  absl::flat_hash_map<const ColonRef*, SumConstructorRef> constructors;
  absl::flat_hash_map<const TypeAlias*, ResolvedTypeAlias> aliases;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_DECLARATION_RESOLUTION_CACHE_H_
