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

#include "xls/dslx/type_system_v2/import_utils.h"

#include <optional>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {

absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }

  // Collect parametrics and instantiator by walking through any type
  // aliases before getting the struct or proc definition.
  std::vector<ExprOrType> parametrics = type_ref_annotation->parametrics();
  std::optional<const StructInstanceBase*> instantiator =
      type_ref_annotation->instantiator();
  TypeDefinition maybe_alias =
      type_ref_annotation->type_ref()->type_definition();

  while (std::holds_alternative<TypeAlias*>(maybe_alias) &&
         dynamic_cast<TypeRefTypeAnnotation*>(
             &std::get<TypeAlias*>(maybe_alias)->type_annotation())) {
    type_ref_annotation = dynamic_cast<TypeRefTypeAnnotation*>(
        &std::get<TypeAlias*>(maybe_alias)->type_annotation());
    if (!parametrics.empty() && !type_ref_annotation->parametrics().empty()) {
      return TypeInferenceErrorStatus(
          annotation->span(), /* type= */ nullptr,
          absl::StrFormat(
              "Parametric values defined multiple times for annotation: `%s`",
              annotation->ToString()),
          import_data.file_table());
    }

    parametrics =
        parametrics.empty() ? type_ref_annotation->parametrics() : parametrics;
    instantiator = instantiator.has_value()
                       ? instantiator
                       : type_ref_annotation->instantiator();
    maybe_alias = type_ref_annotation->type_ref()->type_definition();
  }

  XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                       GetStructOrProcDef(type_ref_annotation, import_data));
  if (!def.has_value()) {
    return std::nullopt;
  }
  return StructOrProcRef{
      .def = *def, .parametrics = parametrics, .instantiator = instantiator};
}

// Forward declaration for internal resolution helper.
absl::StatusOr<std::optional<const StructDefBase*>>
ResolveColonRefToStructDefBase(ColonRef* colon_ref,
                               const ImportData& import_data);

template <typename T>
absl::StatusOr<std::optional<const StructDefBase*>> ResolveToStructBase(
    T node, const ImportData& import_data) {
  return absl::visit(
      Visitor{[&](TypeAlias* alias)
                  -> absl::StatusOr<std::optional<const StructDefBase*>> {
                return GetStructOrProcDef(&alias->type_annotation(),
                                          import_data);
              },
              [](StructDef* struct_def)
                  -> absl::StatusOr<std::optional<const StructDefBase*>> {
                return struct_def;
              },
              [](ProcDef* proc_def)
                  -> absl::StatusOr<std::optional<const StructDefBase*>> {
                return proc_def;
              },
              [&](ColonRef* colon_ref)
                  -> absl::StatusOr<std::optional<const StructDefBase*>> {
                return ResolveColonRefToStructDefBase(colon_ref, import_data);
              },
              [](UseTreeEntry*)
                  -> absl::StatusOr<std::optional<const StructDefBase*>> {
                // TODO(https://github.com/google/xls/issues/352): 2025-01-23
                // Resolve possible Struct or Proc definition through the extern
                // UseTreeEntry.
                return std::nullopt;
              },
              [](auto*) -> absl::StatusOr<std::optional<const StructDefBase*>> {
                return std::nullopt;
              }},
      node);
}

absl::StatusOr<std::optional<const StructDefBase*>>
ResolveColonRefToStructDefBase(ColonRef* colon_ref,
                               const ImportData& import_data) {
  std::optional<ImportSubject> subject = colon_ref->ResolveImportSubject();
  if (subject.has_value() && std::holds_alternative<Import*>(*subject)) {
    Import* import = std::get<Import*>(*subject);
    XLS_ASSIGN_OR_RETURN(ModuleInfo * import_module,
                         import_data.Get(ImportTokens(import->subject())));
    std::optional<ModuleMember*> member =
        import_module->module().FindMemberWithName(colon_ref->attr());
    CHECK(member.has_value());
    return ResolveToStructBase(**member, import_data);
  }
  if (std::holds_alternative<TypeRefTypeAnnotation*>(colon_ref->subject())) {
    return GetStructOrProcDef(
        std::get<TypeRefTypeAnnotation*>(colon_ref->subject()), import_data);
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<const StructDefBase*>> GetStructOrProcDef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  const auto* type_ref_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(annotation);
  if (type_ref_annotation == nullptr) {
    return std::nullopt;
  }
  const TypeDefinition& def =
      type_ref_annotation->type_ref()->type_definition();
  return ResolveToStructBase(def, import_data);
}

}  // namespace xls::dslx
