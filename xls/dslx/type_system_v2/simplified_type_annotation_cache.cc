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

#include "xls/dslx/type_system_v2/simplified_type_annotation_cache.h"

#include <cstdint>
#include <optional>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"

namespace xls::dslx {

bool SimplifiedTypeAnnotationCache::MaybeAddBitsLikeTypeAnnotation(
    Module& module, std::optional<const ParametricContext*> parametric_context,
    const AstNode* node, bool is_signed, int64_t size) {
  if (node->kind() == AstNodeKind::kStructMember) {
    // Currently we only support caching non-parametric struct member types,
    // because they are known while we process its StructDef. We do not have
    // a way to get a callback when a member type of a parametric struct is
    // first unified.
    if (!parametric_context) {
      cache_[{node, std::nullopt}] = SignednessAndSizeToAnnotation(
          module, {TypeInferenceFlag::kNone, is_signed, size},
          *node->GetSpan());
      return true;
    }
  } else if (node->kind() == AstNodeKind::kTypeAlias) {
    cache_[{node, parametric_context}] = SignednessAndSizeToAnnotation(
        module, {TypeInferenceFlag::kNone, is_signed, size}, *node->GetSpan());
    return true;
  }
  return false;
}

std::optional<const TypeAnnotation*>
SimplifiedTypeAnnotationCache::GetSimplifiedypeAnnotation(
    std::optional<const ParametricContext*> parametric_context,
    std::variant<const StructMemberNode*, const TypeAlias*> node) {
  auto cached = cache_.find({ToAstNode(node), parametric_context});
  if (cached != cache_.end()) {
    return cached->second;
  }
  return std::nullopt;
}

}  // namespace xls::dslx
