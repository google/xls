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

namespace xls::dslx {

std::optional<Span> FindDefinition(const Module& m, const Pos& selected) {
  std::vector<const AstNode*> intercepting = m.FindIntercepting(selected);

  std::vector<const NameRef*> name_refs;
  for (const AstNode* node : intercepting) {
    if (auto* name_ref = dynamic_cast<const NameRef*>(node);
        name_ref != nullptr) {
      name_refs.push_back(name_ref);
    }
  }

  if (name_refs.size() == 1) {
    std::variant<const NameDef*, BuiltinNameDef*> name_def =
        name_refs.at(0)->name_def();
    if (std::holds_alternative<const NameDef*>(name_def)) {
      return std::get<const NameDef*>(name_def)->span();
    }
  } else if (name_refs.size() > 1) {
    XLS_LOG(WARNING)
        << "Multiple name references found intercepting at position: "
        << selected;
  }

  return std::nullopt;
}

}  // namespace xls::dslx
