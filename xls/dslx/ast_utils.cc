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

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/token_utils.h"

namespace xls::dslx {
namespace {

// Has to be an enum or builtin-type name, given the context we're in: looking
// for _values_ hanging off, e.g. in service of a `::` ref.
absl::StatusOr<std::variant<EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>>
ResolveTypeDefToDirectColonRefSubject(ImportData* import_data,
                                      const TypeInfo* type_info,
                                      TypeDef* type_def) {
  XLS_VLOG(5) << "ResolveTypeDefToDirectColonRefSubject; type_def: `"
              << type_def->ToString() << "`";

  TypeDefinition td = type_def;
  while (std::holds_alternative<TypeDef*>(td)) {
    TypeDef* type_def = std::get<TypeDef*>(td);
    XLS_VLOG(5) << "TypeDef: `" << type_def->ToString() << "`";
    TypeAnnotation* type = type_def->type_annotation();
    XLS_VLOG(5) << "TypeAnnotation: `" << type->ToString() << "`";

    if (auto* bti = dynamic_cast<BuiltinTypeAnnotation*>(type);
        bti != nullptr) {
      return bti->builtin_name_def();
    }
    if (auto* ata = dynamic_cast<ArrayTypeAnnotation*>(type); ata != nullptr) {
      return ata;
    }

    TypeRefTypeAnnotation* type_ref_type =
        dynamic_cast<TypeRefTypeAnnotation*>(type);
    // TODO(rspringer): We'll need to collect parametrics from type_ref_type to
    // support parametric TypeDefs.
    XLS_RET_CHECK(type_ref_type != nullptr)
        << type->ToString() << " :: " << type->GetNodeTypeName();
    XLS_VLOG(5) << "TypeRefTypeAnnotation: `" << type_ref_type->ToString()
                << "`";

    td = type_ref_type->type_ref()->type_definition();
  }

  if (std::holds_alternative<ColonRef*>(td)) {
    ColonRef* colon_ref = std::get<ColonRef*>(td);
    XLS_ASSIGN_OR_RETURN(auto subject, ResolveColonRefSubject(
                                           import_data, type_info, colon_ref));
    XLS_RET_CHECK(std::holds_alternative<Module*>(subject));
    Module* module = std::get<Module*>(subject);
    XLS_ASSIGN_OR_RETURN(td, module->GetTypeDefinition(colon_ref->attr()));

    if (std::holds_alternative<TypeDef*>(td)) {
      // We need to get the right type info for the enum's containing module. We
      // can get the top-level module since [currently?] enums can't be
      // parameterized.
      type_info = import_data->GetRootTypeInfo(module).value();
      return ResolveTypeDefToDirectColonRefSubject(import_data, type_info,
                                                   std::get<TypeDef*>(td));
    }
  }

  if (!std::holds_alternative<EnumDef*>(td)) {
    return absl::InternalError(
        "ResolveTypeDefToDirectColonRefSubject() can only be called when the "
        "TypeDef "
        "directory or indirectly refers to an EnumDef.");
  }

  return std::get<EnumDef*>(td);
}

void FlattenToSetInternal(const AstNode* node,
                          absl::flat_hash_set<const AstNode*>* the_set) {
  the_set->insert(node);
  for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
    FlattenToSetInternal(child, the_set);
  }
}

}  // namespace

bool IsBuiltinFn(Expr* callee) {
  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  if (name_ref == nullptr) {
    return false;
  }

  return std::holds_alternative<BuiltinNameDef*>(name_ref->name_def());
}

absl::StatusOr<std::string> GetBuiltinName(Expr* callee) {
  if (!IsBuiltinFn(callee)) {
    return absl::InvalidArgumentError("Callee is not a builtin function.");
  }

  NameRef* name_ref = dynamic_cast<NameRef*>(callee);
  return name_ref->identifier();
}

absl::StatusOr<Function*> ResolveFunction(Expr* callee,
                                          const TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetMemberOrError<Function>(
        name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  std::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  std::optional<const ImportedInfo*> imported_info =
      type_info->GetImported(*import);
  return imported_info.value()->module->GetMemberOrError<Function>(
      colon_ref->attr());
}

absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetMemberOrError<Proc>(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  std::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  std::optional<const ImportedInfo*> imported_info =
      type_info->GetImported(*import);
  return imported_info.value()->module->GetMemberOrError<Proc>(
      colon_ref->attr());
}

// When a ColonRef's subject is a NameRef, this resolves the entity referred to
// by that ColonRef. In a valid program that can only be a limited set of
// things, which is reflected in the return type provided.
//
// e.g.
//
//    A::B
//    ^
//    \- subject name_ref
//
// Args:
//  name_ref: The subject in the colon ref.
//
// Returns the entity the subject name_ref is referring to.
static absl::StatusOr<
    std::variant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>>
ResolveColonRefNameRefSubject(NameRef* name_ref, ImportData* import_data,
                              const TypeInfo* type_info) {
  XLS_VLOG(5) << "ResolveColonRefNameRefSubject for `" << name_ref->ToString()
              << "`";

  std::variant<const NameDef*, BuiltinNameDef*> any_name_def =
      name_ref->name_def();
  if (std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return std::get<BuiltinNameDef*>(any_name_def);
  }

  const NameDef* name_def = std::get<const NameDef*>(any_name_def);
  AstNode* definer = name_def->definer();
  XLS_VLOG(5) << " ResolveColonRefNameRefSubject definer: `"
              << definer->ToString()
              << "` type: " << definer->GetNodeTypeName();

  if (Import* import = dynamic_cast<Import*>(definer); import != nullptr) {
    std::optional<const ImportedInfo*> imported =
        type_info->GetImported(import);
    if (!imported.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find Module for Import: ", import->ToString()));
    }
    return imported.value()->module;
  }

  // If the LHS isn't an Import, then it has to be an EnumDef (possibly via a
  // TypeDef).
  if (EnumDef* enum_def = dynamic_cast<EnumDef*>(definer);
      enum_def != nullptr) {
    return enum_def;
  }

  TypeDef* type_def = dynamic_cast<TypeDef*>(definer);
  XLS_RET_CHECK(type_def != nullptr);

  if (type_def->owner() != type_info->module()) {
    // We need to get the right type info for the enum's containing module. We
    // can get the top-level module since [currently?] enums can't be
    // parameterized (and we know this must be an enum, per the above).
    type_info = import_data->GetRootTypeInfo(type_def->owner()).value();
  }
  XLS_ASSIGN_OR_RETURN(auto resolved, ResolveTypeDefToDirectColonRefSubject(
                                          import_data, type_info, type_def));
  return WidenVariant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>(
      resolved);
}

absl::StatusOr<
    std::variant<Module*, EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*>>
ResolveColonRefSubject(ImportData* import_data, const TypeInfo* type_info,
                       const ColonRef* colon_ref) {
  XLS_VLOG(5) << "ResolveColonRefSubject for " << colon_ref->ToString();

  if (std::holds_alternative<NameRef*>(colon_ref->subject())) {
    NameRef* name_ref = std::get<NameRef*>(colon_ref->subject());
    return ResolveColonRefNameRefSubject(name_ref, import_data, type_info);
  }

  XLS_RET_CHECK(std::holds_alternative<ColonRef*>(colon_ref->subject()));
  ColonRef* subject = std::get<ColonRef*>(colon_ref->subject());
  XLS_ASSIGN_OR_RETURN(auto resolved_subject,
                       ResolveColonRefSubject(import_data, type_info, subject));
  // Has to be a module, since it's a ColonRef inside a ColonRef.
  XLS_RET_CHECK(std::holds_alternative<Module*>(resolved_subject));
  Module* module = std::get<Module*>(resolved_subject);

  // And the subject has to be a type, namely an enum, since the ColonRef must
  // be of the form: <MODULE>::SOMETHING::SOMETHING_ELSE. Keep in mind, though,
  // that we might have to traverse an EnumDef.
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       module->GetTypeDefinition(subject->attr()));
  if (std::holds_alternative<TypeDef*>(td)) {
    XLS_ASSIGN_OR_RETURN(auto resolved,
                         ResolveTypeDefToDirectColonRefSubject(
                             import_data, type_info, std::get<TypeDef*>(td)));
    return WidenVariant<Module*, EnumDef*, BuiltinNameDef*,
                        ArrayTypeAnnotation*>(resolved);
  }

  return std::get<EnumDef*>(td);
}

absl::Status VerifyParentage(const Module* module) {
  for (const ModuleMember member : module->top()) {
    if (std::holds_alternative<Function*>(member)) {
      return VerifyParentage(std::get<Function*>(member));
    }
    if (std::holds_alternative<Proc*>(member)) {
      return VerifyParentage(std::get<Proc*>(member));
    }
    if (std::holds_alternative<TestFunction*>(member)) {
      return VerifyParentage(std::get<TestFunction*>(member));
    }
    if (std::holds_alternative<TestProc*>(member)) {
      return VerifyParentage(std::get<TestProc*>(member));
    }
    if (std::holds_alternative<QuickCheck*>(member)) {
      return VerifyParentage(std::get<QuickCheck*>(member));
    }
    if (std::holds_alternative<TypeDef*>(member)) {
      return VerifyParentage(std::get<TypeDef*>(member));
    }
    if (std::holds_alternative<StructDef*>(member)) {
      return VerifyParentage(std::get<StructDef*>(member));
    }
    if (std::holds_alternative<ConstantDef*>(member)) {
      return VerifyParentage(std::get<ConstantDef*>(member));
    }
    if (std::holds_alternative<EnumDef*>(member)) {
      return VerifyParentage(std::get<EnumDef*>(member));
    }
    if (std::holds_alternative<Import*>(member)) {
      return VerifyParentage(std::get<Import*>(member));
    }
  }

  return absl::OkStatus();
}

absl::Status VerifyParentage(const AstNode* root) {
  if (const Module* module = dynamic_cast<const Module*>(root);
      module != nullptr) {
    return VerifyParentage(module);
  }

  for (const auto* child : root->GetChildren(/*want_types=*/true)) {
    XLS_RETURN_IF_ERROR(VerifyParentage(child));
    if (child->parent() != root) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Child \"%s\" (%s) of node \"%s\" (%s) had "
          "node \"%s\" (%s) as its parent.",
          child->ToString(), child->GetNodeTypeName(), root->ToString(),
          root->GetNodeTypeName(), child->parent()->ToString(),
          child->parent()->GetNodeTypeName()));
    }
  }

  return absl::OkStatus();
}

absl::flat_hash_set<const AstNode*> FlattenToSet(const AstNode* node) {
  absl::flat_hash_set<const AstNode*> the_set;
  FlattenToSetInternal(node, &the_set);
  return the_set;
}

absl::StatusOr<InterpValue> GetBuiltinNameDefColonAttr(
    const BuiltinNameDef* builtin_name_def, std::string_view attr) {
  // We only support MAX on builtin types at the moment -- this is checked
  // during typechecking.
  if (attr != "MAX") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid attribute of builtin name %s: %s",
                        builtin_name_def->identifier(), attr));
  }
  const auto& sized_type_keywords = GetSizedTypeKeywordsMetadata();
  auto it = sized_type_keywords.find(builtin_name_def->identifier());
  // We should have checked this was a valid type keyword in typechecking.
  XLS_RET_CHECK(it != sized_type_keywords.end());
  auto [is_signed, width] = it->second;
  return InterpValue::MakeMaxValue(is_signed, width);
}

absl::StatusOr<InterpValue> GetArrayTypeColonAttr(
    const ArrayTypeAnnotation* array_type, uint64_t constexpr_dim,
    std::string_view attr) {
  auto* builtin_type =
      dynamic_cast<BuiltinTypeAnnotation*>(array_type->element_type());
  if (builtin_type == nullptr) {
    return absl::InvalidArgumentError(
        "Can only take '::' attributes of uN/sN/bits array types.");
  }
  bool is_signed;
  switch (builtin_type->builtin_type()) {
    case BuiltinType::kUN:
    case BuiltinType::kBits:
      is_signed = false;
      break;
    case BuiltinType::kSN:
      is_signed = true;
      break;
    default:
      return absl::InvalidArgumentError(
          "Can only take '::' attributes of uN/sN/bits array types.");
  }

  if (attr != "MAX") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid attribute of builtin array type: %s", attr));
  }
  return InterpValue::MakeMaxValue(is_signed, constexpr_dim);
}

}  // namespace xls::dslx
