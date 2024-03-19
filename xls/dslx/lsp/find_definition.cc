// Copyright 2023 The XLS Authors
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

#include "xls/dslx/lsp/find_definition.h"

#include <optional>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

std::optional<Span> FindDefinition(const Module& m, const Pos& selected) {
  std::vector<const AstNode*> intercepting = m.FindIntercepting(selected);
  XLS_VLOG(3) << "Found " << intercepting.size()
              << " nodes intercepting selected position: " << selected;

  std::vector<const NameDef*> defs;
  for (const AstNode* node : intercepting) {
    XLS_VLOG(5) << "Intercepting node kind: " << node->kind() << " @ "
                << node->GetSpan().value();
    if (auto* name_ref = dynamic_cast<const NameRef*>(node);
        name_ref != nullptr) {
      XLS_VLOG(3) << "Intercepting name ref: `" << name_ref->ToString() << "`";
      std::variant<const NameDef*, BuiltinNameDef*> name_def =
          name_ref->name_def();
      if (std::holds_alternative<const NameDef*>(name_def)) {
        defs.push_back(std::get<const NameDef*>(name_def));
      }
    } else if (auto* type_ref = dynamic_cast<const TypeRef*>(node)) {
      XLS_VLOG(3) << "Intercepting type ref: `" << type_ref->ToString() << "`";
      AnyNameDef type_definer =
          TypeDefinitionGetNameDef(type_ref->type_definition());
      if (std::holds_alternative<const NameDef*>(type_definer)) {
        defs.push_back(std::get<const NameDef*>(type_definer));
      }
    }
  }

  if (defs.size() == 1) {
    return defs.at(0)->GetSpan();
  }
  if (defs.size() > 1) {
    LOG(WARNING) << "Multiple name references found intercepting at position: "
                 << selected;
  }

  return std::nullopt;
}

}  // namespace xls::dslx
