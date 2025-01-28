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

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

const NameDef* GetNameDef(
    const std::variant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*,
                       Impl*>& colon_ref_subject,
    std::string_view attr) {
  return absl::visit(
      Visitor{
          [&](Module* m) -> const NameDef* {
            std::optional<ModuleMember*> member = m->FindMemberWithName(attr);
            CHECK(member.has_value());
            const ModuleMember& mm = *member.value();
            std::vector<NameDef*> name_defs = ModuleMemberGetNameDefs(mm);
            VLOG(5) << absl::StreamFormat(
                "module: `%s` attr: `%s` name_defs: [%s]", m->name(), attr,
                absl::StrJoin(name_defs, ", ",
                              [](std::string* out, const NameDef* name_def) {
                                absl::StrAppendFormat(out, "`%s`",
                                                      name_def->identifier());
                              }));
            if (name_defs.empty()) {
              return nullptr;
            }
            if (name_defs.size() == 1) {
              return name_defs.at(0);
            }
            // Constructs like `use` statement can define multiple names at
            // module scope, and we can only refer to one of them by colon-ref.
            for (const NameDef* name_def : name_defs) {
              if (name_def->identifier() == attr) {
                return name_def;
              }
            }
            return nullptr;
          },
          [&](EnumDef* e) -> const NameDef* { return e->GetNameDef(attr); },
          [](Impl* s) -> const NameDef* { return nullptr; },
          [](BuiltinNameDef*) -> const NameDef* { return nullptr; },
          [](ArrayTypeAnnotation*) -> const NameDef* { return nullptr; },
      },
      colon_ref_subject);
}

}  // namespace

std::optional<const NameDef*> FindDefinition(const Module& m,
                                             const Pos& selected,
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
            << node->GetSpan().value().ToString(import_data.file_table())
            << " :: `" << node->ToString() << "`";

    if (auto* colon_ref = dynamic_cast<const ColonRef*>(node);
        colon_ref != nullptr) {
      VLOG(3) << "Intercepting node is ColonRef: `" << colon_ref->ToString()
              << "`";

      using Resolved = std::variant<Module*, EnumDef*, BuiltinNameDef*,
                                    ArrayTypeAnnotation*, Impl*>;
      absl::StatusOr<Resolved> node = ResolveColonRefSubjectAfterTypeChecking(
          &import_data, &type_info, colon_ref);
      if (!node.ok()) {
        VLOG(3) << "Could not resolve ColonRef subject; status: "
                << node.status();
        return std::nullopt;
      }

      VLOG(3) << absl::StreamFormat("Resolving NameDef for attribute: `%s`",
                                    colon_ref->attr());
      const NameDef* name_def = GetNameDef(node.value(), colon_ref->attr());

      // Since this is another file, we cannot do an enclosure test, and we
      // return immediately.
      return name_def;
    }

    if (auto* name_ref = dynamic_cast<const NameRef*>(node);
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
    } else if (auto* name_def = dynamic_cast<const NameDef*>(node)) {
      defs.push_back(Reference{name_def->span(), name_def});
    }
  }

  VLOG(3) << "Found " << defs.size() << " definitions intercepting position "
          << selected;

  if (defs.size() == 1) {
    return defs.at(0).to;
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
            << "` @ "
            << reference.to->span().ToString(import_data.file_table());
    return reference.to;
  }

  return std::nullopt;
}

}  // namespace xls::dslx
