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

#include "xls/dslx/type_system/warn_on_defined_but_unused.h"

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {

absl::Status WarnOnDefinedButUnused(Function& f, DeduceCtx* ctx) {
  // We say we want types in case there are references in e.g. expressions
  // within type annotations, say dimensions.
  XLS_ASSIGN_OR_RETURN(std::vector<AstNode*> nodes,
                       CollectUnder(f.body(), /*want_types=*/true));

  // Note: we use pointer sets instead of using a btree with Span as the
  // comparator because we want to avoid the case where nodes have the same span
  // (e.g. via mistakes in span formation).
  absl::flat_hash_set<const NameDef*> all_defs;
  absl::flat_hash_set<const NameDef*> referenced_defs;

  // Helper that unwraps an AnyNameDef and adds it to the referenced_def set if
  // it is not a BuiltInNameDef.
  auto reference_any_name_def = [&](const AnyNameDef& any_name_def) {
    if (std::holds_alternative<const NameDef*>(any_name_def)) {
      referenced_defs.insert(std::get<const NameDef*>(any_name_def));
    }
  };

  // For every node in the body, see if it's a name definition or reference of
  // some form, and handle it appropriately.
  for (const AstNode* node : nodes) {
    if (const NameDef* name_def = dynamic_cast<const NameDef*>(node)) {
      all_defs.insert(name_def);
    } else if (const NameRef* name_ref = dynamic_cast<const NameRef*>(node)) {
      reference_any_name_def(name_ref->name_def());
    } else if (const TypeRef* type_ref = dynamic_cast<const TypeRef*>(node)) {
      reference_any_name_def(
          TypeDefinitionGetNameDef(type_ref->type_definition()));
    } else {
      continue;  // Not relevant.
    }
  }

  // Figure out which of the definitions were unreferenced.
  absl::flat_hash_set<const NameDef*> unreferenced_defs;
  for (const NameDef* def : all_defs) {
    if (referenced_defs.contains(def)) {
      continue;
    }
    unreferenced_defs.insert(def);
  }

  // Sort them for reporting stability.
  std::vector<const NameDef*> to_warn(unreferenced_defs.begin(),
                                      unreferenced_defs.end());
  std::sort(
      to_warn.begin(), to_warn.end(), [](const NameDef* a, const NameDef* b) {
        return a->span() < b->span() ||
               (a->span() == b->span() && a->identifier() < b->identifier());
      });

  // Warn on all the appropriate NameDefs that went unreferenced.
  for (const NameDef* n : to_warn) {
    if (absl::StartsWith(n->identifier(), "_")) {
      // Users can silence unused warnings by prefixing an identifier with an
      // underscore to make it more well documented; e.g.
      //  let (one, _two, three) = ...;  // _two can go unused
      continue;
    }
    std::optional<const Type*> type = ctx->type_info()->GetItem(n);
    XLS_RET_CHECK(type.has_value()) << absl::StreamFormat(
        "NameDef `%s` %p @ %s parent kind `%v` had no associated type "
        "information in type info %p",
        n->ToString(), n, n->span().ToString(), n->parent()->kind(),
        ctx->type_info());
    // For now tokens are implicitly joined at the end of a proc `next()`, so we
    // don't warn on these.
    if (type.value()->IsToken()) {
      continue;
    }
    // TODO(leary): 2023-08-10 Struct instantiations currently bypass type
    // aliases, so we don't have precise information here. I believe we need to
    // hold a TypeRef in the StructInstance AST node.
    if (n->parent()->kind() == AstNodeKind::kTypeAlias) {
      continue;
    }
    ctx->warnings()->Add(
        n->span(), WarningKind::kUnusedDefinition,
        absl::StrFormat(
            "Definition of `%s` (type `%s`) is not used in function `%s`",
            n->identifier(), type.value()->ToString(), f.identifier()));
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
