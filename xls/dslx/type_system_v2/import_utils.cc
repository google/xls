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
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

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

template <typename T>
absl::StatusOr<std::optional<StructOrProcRef>> ResolveToStructOrProcRef(
    T node, const ImportData& import_data) {
  return absl::visit(
      Visitor{
          [&](TypeAlias* alias)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return GetStructOrProcRef(&alias->type_annotation(), import_data);
          },
          [&](TypeRefTypeAnnotation* type_annotation)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return GetStructOrProcRef(type_annotation, import_data);
          },
          [](StructDef* struct_def)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return StructOrProcRef{.def = struct_def};
          },
          [](ProcDef* proc_def)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return StructOrProcRef{.def = proc_def};
          },
          [&](NameRef* name_ref)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return ResolveToStructOrProcRef(name_ref->name_def(), import_data);
          },
          [&](const NameDef* name_def)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            if (auto* struct_def =
                    dynamic_cast<StructDefBase*>(name_def->definer())) {
              return StructOrProcRef{.def = struct_def};
            }
            if (auto* type_alias =
                    dynamic_cast<TypeAlias*>(name_def->definer())) {
              return GetStructOrProcRef(&type_alias->type_annotation(),
                                        import_data);
            }
            return std::nullopt;
          },
          [&](ColonRef* colon_ref)
              -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return GetStructOrProcRef(colon_ref, import_data);
          },
          [](UseTreeEntry*) -> absl::StatusOr<std::optional<StructOrProcRef>> {
            // TODO(https://github.com/google/xls/issues/352): 2025-01-23
            // Resolve possible Struct or Proc definition through the extern
            // UseTreeEntry.
            return std::nullopt;
          },
          [](auto*) -> absl::StatusOr<std::optional<StructOrProcRef>> {
            return std::nullopt;
          }},
      node);
}

absl::StatusOr<std::optional<ModuleInfo*>> GetImportedModuleInfo(
    const ColonRef* colon_ref, const ImportData& import_data) {
  std::optional<ImportSubject> subject = colon_ref->ResolveImportSubject();
  if (subject.has_value() && std::holds_alternative<Import*>(*subject)) {
    Import* import = std::get<Import*>(*subject);
    return import_data.Get(ImportTokens(import->subject()));
  }
  return std::nullopt;
}

absl::StatusOr<ModuleMember> GetPublicModuleMember(
    const Module& module, const ColonRef* node, const FileTable& file_table) {
  std::optional<const ModuleMember*> member =
      module.FindMemberWithName(node->attr());
  if (!member.has_value()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to a module member %s that "
                        "doesn't exist",
                        node->ToString()),
        file_table);
  }
  if (!IsPublic(**member)) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr,
        absl::StrFormat("Attempted to refer to a module member %s that "
                        "is not public",
                        node->ToString()),
        file_table);
  }
  return **member;
}

absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const ColonRef* colon_ref, const ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> import_module,
                       GetImportedModuleInfo(colon_ref, import_data));
  if (import_module.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        ModuleMember member,
        GetPublicModuleMember((*import_module)->module(), colon_ref,
                              import_data.file_table()));
    return ResolveToStructOrProcRef(member, import_data);
  }
  return ResolveToStructOrProcRef(colon_ref->subject(), import_data);
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
  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> ref,
                       ResolveToStructOrProcRef(def, import_data));
  if (!ref.has_value()) {
    return std::nullopt;
  }
  return ref->def;
}

template <typename T>
absl::StatusOr<std::optional<const EnumDef*>> ResolveToEnumDef(
    T node, const ImportData& import_data) {
  return absl::visit(
      Visitor{
          [&](TypeAlias* alias)
              -> absl::StatusOr<std::optional<const EnumDef*>> {
            const TypeAnnotation* type_annotation = &alias->type_annotation();
            if (const TypeRefTypeAnnotation* type_ref_type_annotation =
                    dynamic_cast<const TypeRefTypeAnnotation*>(
                        type_annotation)) {
              return ResolveToEnumDef(
                  type_ref_type_annotation->type_ref()->type_definition(),
                  import_data);
            }
            return std::nullopt;
          },
          [&](ColonRef* colon_ref)
              -> absl::StatusOr<std::optional<const EnumDef*>> {
            XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> import_module,
                                 GetImportedModuleInfo(colon_ref, import_data));
            if (import_module.has_value()) {
              XLS_ASSIGN_OR_RETURN(
                  ModuleMember member,
                  GetPublicModuleMember((*import_module)->module(), colon_ref,
                                        import_data.file_table()));
              return ResolveToEnumDef(member, import_data);
            }
            return std::nullopt;
          },
          [](EnumDef* enum_def)
              -> absl::StatusOr<std::optional<const EnumDef*>> {
            return enum_def;
          },
          [](auto* n) -> absl::StatusOr<std::optional<const EnumDef*>> {
            return std::nullopt;
          }},
      node);
}

absl::StatusOr<std::optional<const EnumDef*>> GetEnumDef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  if (const auto* type_ref_type_annotation =
          dynamic_cast<const TypeRefTypeAnnotation*>(annotation)) {
    return ResolveToEnumDef(
        type_ref_type_annotation->type_ref()->type_definition(), import_data);
  }
  return std::nullopt;
}

bool IsImport(const ColonRef* colon_ref) {
  if (colon_ref->ResolveImportSubject().has_value()) {
    return true;
  }
  if (std::holds_alternative<ColonRef*>(colon_ref->subject())) {
    return IsImport(std::get<ColonRef*>(colon_ref->subject()));
  }
  return false;
}

}  // namespace xls::dslx
