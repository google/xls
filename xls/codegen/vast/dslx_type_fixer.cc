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

#include "xls/codegen/vast/dslx_type_fixer.h"

#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls {
namespace dslx {

class DslxTypeFixerImpl : public DslxTypeFixer {
 public:
  CloneReplacer GetReplacer(const TypeInfo* ti) final {
    return [this, ti](const AstNode* node, Module* target_module,
                      const absl::flat_hash_map<const AstNode*, AstNode*>&)
               -> absl::StatusOr<std::optional<AstNode*>> {
      std::optional<const Expr*> unwrapped = UnwrapDeadCast(ti, node);
      if (unwrapped.has_value()) {
        XLS_ASSIGN_OR_RETURN(AstNode * clone_of_unwrapped,
                             CloneAst(*unwrapped, GetReplacer(ti)));
        Expr* result = down_cast<Expr*>(clone_of_unwrapped);
        result->set_in_parens(false);
        return result;
      }
      return std::nullopt;
    };
  }

 private:
  std::optional<const Expr*> UnwrapDeadCast(const TypeInfo* ti,
                                            const AstNode* node) {
    const std::optional<Type*> casted = ti->GetItem(node);
    std::optional<const Expr*> unwrapped;

    // Walk through the layers of a potential cast onion. Note that there are
    // usually at most 2-3 layers.
    while (node->kind() == AstNodeKind::kCast) {
      const Expr* expr = down_cast<const Cast*>(node)->expr();
      const std::optional<Type*> uncasted = ti->GetItem(expr);

      // See if we are OK to drop all layers of casts up to here.
      if (**casted == **uncasted) {
        unwrapped = expr;
      } else {
        std::optional<BitsLikeProperties> casted_bits = GetBitsLike(**casted);
        std::optional<BitsLikeProperties> uncasted_bits =
            GetBitsLike(**uncasted);
        if (casted_bits.has_value() && uncasted_bits.has_value() &&
            *casted_bits == *uncasted_bits) {
          unwrapped = expr;
        }
      }

      node = expr;
    }

    return unwrapped;
  }
};

}  // namespace dslx

std::unique_ptr<DslxTypeFixer> CreateDslxTypeFixer() {
  return std::make_unique<dslx::DslxTypeFixerImpl>();
}

}  // namespace xls
