// Copyright 2021 The XLS Authors
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
#include "xls/dslx/ast_utils.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {
namespace {

// Has to be an enum, given the context we're in: looking for _values_.
absl::StatusOr<EnumDef*> ResolveTypeDefToEnum(ImportData* import_data,
                                              const TypeInfo* type_info,
                                              TypeDef* type_def) {
  TypeDefinition td = type_def;
  while (absl::holds_alternative<TypeDef*>(td)) {
    TypeDef* type_def = absl::get<TypeDef*>(td);
    TypeAnnotation* type = type_def->type_annotation();
    TypeRefTypeAnnotation* type_ref_type =
        dynamic_cast<TypeRefTypeAnnotation*>(type);
    // TODO(rspringer): We'll need to collect parametrics from type_ref_type to
    // support parametric TypeDefs.
    XLS_RET_CHECK(type_ref_type != nullptr);
    td = type_ref_type->type_ref()->type_definition();
  }

  if (absl::holds_alternative<ColonRef*>(td)) {
    ColonRef* colon_ref = absl::get<ColonRef*>(td);
    XLS_ASSIGN_OR_RETURN(auto subject, ResolveColonRefSubject(
                                           import_data, type_info, colon_ref));
    XLS_RET_CHECK(absl::holds_alternative<Module*>(subject));
    Module* module = absl::get<Module*>(subject);
    XLS_ASSIGN_OR_RETURN(td, module->GetTypeDefinition(colon_ref->attr()));

    if (absl::holds_alternative<TypeDef*>(td)) {
      // We need to get the right type info for the enum's containing module. We
      // can get the top-level module since [currently?] enums can't be
      // parameterized.
      type_info = import_data->GetRootTypeInfo(module).value();
      return ResolveTypeDefToEnum(import_data, type_info,
                                  absl::get<TypeDef*>(td));
    }
  }

  if (!absl::holds_alternative<EnumDef*>(td)) {
    return absl::InternalError(
        "ResolveTypeDefToEnum() can only be called when the TypeDef "
        "directory or indirectly refers to an EnumDef.");
  }

  return absl::get<EnumDef*>(td);
}

}  // namespace

bool IsBuiltinFn(Expr* callee) {
  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }

  return absl::holds_alternative<BuiltinNameDef*>(name_ref->name_def());
}

absl::StatusOr<std::string> GetBuiltinName(Expr* callee) {
  if (!IsBuiltinFn(callee)) {
    return absl::InvalidArgumentError("Callee is not a builtin function.");
  }

  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  return name_ref->identifier();
}

absl::StatusOr<Function*> ResolveFunction(Expr* callee, TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetFunctionOrError(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  absl::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  absl::optional<const ImportedInfo*> imported_info =
      type_info->GetImported(*import);
  return imported_info.value()->module->GetFunctionOrError(colon_ref->attr());
}

absl::StatusOr<Proc*> ResolveProc(Expr* callee, TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetProcOrError(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  absl::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  absl::optional<const ImportedInfo*> imported_info =
      type_info->GetImported(*import);
  return imported_info.value()->module->GetProcOrError(colon_ref->attr());
}

absl::StatusOr<absl::variant<Module*, EnumDef*>> ResolveColonRefSubject(
    ImportData* import_data, const TypeInfo* type_info,
    const ColonRef* colon_ref) {
  if (absl::holds_alternative<NameRef*>(colon_ref->subject())) {
    // Inside a ColonRef, the LHS can't be a BuiltinNameDef.
    NameRef* name_ref = absl::get<NameRef*>(colon_ref->subject());
    NameDef* name_def = absl::get<NameDef*>(name_ref->name_def());
    if (Import* import = dynamic_cast<Import*>(name_def->definer());
        import != nullptr) {
      absl::optional<const ImportedInfo*> imported =
          type_info->GetImported(import);
      if (!imported.has_value()) {
        return absl::InternalError(absl::StrCat(
            "Could not find Module for Import: ", import->ToString()));
      }
      return imported.value()->module;
    }

    // If the LHS isn't an Import, then it has to be an EnumDef (possibly via a
    // TypeDef).
    if (EnumDef* enum_def = dynamic_cast<EnumDef*>(name_def->definer());
        enum_def != nullptr) {
      return enum_def;
    }

    TypeDef* type_def = dynamic_cast<TypeDef*>(name_def->definer());
    XLS_RET_CHECK(type_def != nullptr);

    if (type_def->owner() != type_info->module()) {
      // We need to get the right type info for the enum's containing module. We
      // can get the top-level module since [currently?] enums can't be
      // parameterized (and we know this must be an enum, per the above).
      type_info = import_data->GetRootTypeInfo(type_def->owner()).value();
    }
    return ResolveTypeDefToEnum(import_data, type_info, type_def);
  }

  XLS_RET_CHECK(absl::holds_alternative<ColonRef*>(colon_ref->subject()));
  ColonRef* subject = absl::get<ColonRef*>(colon_ref->subject());
  XLS_ASSIGN_OR_RETURN(auto resolved_subject,
                       ResolveColonRefSubject(import_data, type_info, subject));
  // Has to be a module, since it's a ColonRef inside a ColonRef.
  XLS_RET_CHECK(absl::holds_alternative<Module*>(resolved_subject));
  Module* module = absl::get<Module*>(resolved_subject);

  // And the subject has to be a type, namely an enum, since the ColonRef must
  // be of the form: <MODULE>::SOMETHING::SOMETHING_ELSE. Keep in mind, though,
  // that we might have to traverse an EnumDef.
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       module->GetTypeDefinition(subject->attr()));
  if (absl::holds_alternative<TypeDef*>(td)) {
    return ResolveTypeDefToEnum(import_data, type_info,
                                absl::get<TypeDef*>(td));
  }

  return absl::get<EnumDef*>(td);
}

}  // namespace xls::dslx
