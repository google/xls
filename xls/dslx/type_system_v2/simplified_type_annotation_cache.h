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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_CACHE_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_CACHE_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// This class caches simplified type annotations for certain kinds of AST nodes
// if they are likely to be referenced repeatedly inside type resolution and
// unification. This can speed up this process in certain use cases, for example
// when many instances of a struct is created.
class SimplifiedTypeAnnotationCache {
 public:
  // Caches a fabricated bits-like type annotation for the given
  // `parametric_context` and `node`. Only certain kinds of nodes will be
  // cached, otherwise this function has no effect.
  bool MaybeAddBitsLikeTypeAnnotation(
      Module& module,
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node, bool is_signed, int64_t size);

  // Gets the cached type annotation for the given `parametric_context` and
  // `node`, if one exists.
  std::optional<const TypeAnnotation*> GetSimplifiedypeAnnotation(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<const StructMemberNode*, const TypeAlias*> node);

 private:
  absl::flat_hash_map<
      std::pair<const AstNode*, std::optional<const ParametricContext*>>,
      const TypeAnnotation*>
      cache_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_CACHE_H_
