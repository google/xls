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
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

const NameDef* GetNameDef(
    const std::variant<Module*, EnumDef*, BuiltinNameDef*,
                       ArrayTypeAnnotation*>& colon_ref_subject,
    std::string_view attr) {
  return absl::visit(
      Visitor{
          [&](Module* m) -> const NameDef* {
            std::optional<ModuleMember*> member = m->FindMemberWithName(attr);
            CHECK(member.has_value());
            return ModuleMemberGetNameDef(*member.value());
          },
          [&](EnumDef* e) -> const NameDef* { return e->GetNameDef(attr); },
          [](BuiltinNameDef*) -> const NameDef* { return nullptr; },
          [](ArrayTypeAnnotation*) -> const NameDef* { return nullptr; },
      },
      colon_ref_subject);
}

}  // namespace

std::optional<Span> FindDefinition(const Module& m, const Pos& selected,
                                   const TypeInfo& type_info,
                                   ImportData& import_data) {
  std::vector<const AstNode*> intercepting = m.FindIntercepting(selected);
  VLOG(3) << "Found " << intercepting.size()
          << " nodes intercepting selected position: " << selected;

  struct Reference {
    // The span where the reference occurs.
    Span from;
    // The name definition being referred to.
    const NameDef* to;
  };

  std::vector<Reference> defs;
  for (const AstNode* node : intercepting) {
    VLOG(5) << "Intercepting node kind: " << node->kind() << " @ "
            << node->GetSpan().value() << " :: `" << node->ToString() << "`";
    if (auto* colon_ref = dynamic_cast<const ColonRef*>(node);
        colon_ref != nullptr) {
      VLOG(3) << "Intercepting colon ref: `" << colon_ref->ToString() << "`";
      auto node_or = ResolveColonRefSubjectAfterTypeChecking(
          &import_data, &type_info, colon_ref);
      if (!node_or.ok()) {
        return std::nullopt;
      }
      const NameDef* name_def = GetNameDef(node_or.value(), colon_ref->attr());
      if (name_def != nullptr) {
        defs.push_back(Reference{colon_ref->span(), name_def});
      }
    } else if (auto* name_ref = dynamic_cast<const NameRef*>(node);
               name_ref != nullptr) {
      VLOG(3) << "Intercepting name ref: `" << name_ref->ToString() << "`";
      std::variant<const NameDef*, BuiltinNameDef*> name_def =
          name_ref->name_def();
      if (std::holds_alternative<const NameDef*>(name_def)) {
        defs.push_back(
            Reference{name_ref->span(), std::get<const NameDef*>(name_def)});
      }
    } else if (auto* type_ref = dynamic_cast<const TypeRef*>(node)) {
      VLOG(3) << "Intercepting type ref: `" << type_ref->ToString() << "`";
      AnyNameDef type_definer =
          TypeDefinitionGetNameDef(type_ref->type_definition());
      if (std::holds_alternative<const NameDef*>(type_definer)) {
        defs.push_back(Reference{type_ref->span(),
                                 std::get<const NameDef*>(type_definer)});
      }
    }
  }

  if (defs.size() == 1) {
    return defs.at(0).to->GetSpan();
  }
  if (defs.size() > 1) {
    // Find the reference that is "most containing" (i.e. outer-most).
    //
    // Consider the case of a colon-ref:
    //
    // `foo::bar`
    //    ^ position here
    //
    // It will intercept both the `foo` name reference and the `foo::bar` colon
    // reference and we want to present the colon-referenced definition (and
    // that's the outer-most / most containing).
    size_t most_containing = 0;
    for (size_t i = 1; i < defs.size(); ++i) {
      if (defs[i].from.Contains(defs[most_containing].from.start())) {
        most_containing = i;
      }
    }
    const Reference& reference = defs.at(most_containing);
    VLOG(3) << "Most containing; reference is to: `" << reference.to->ToString()
            << "` @ " << reference.to->span();
    return reference.to->GetSpan();
  }

  return std::nullopt;
}

}  // namespace xls::dslx
