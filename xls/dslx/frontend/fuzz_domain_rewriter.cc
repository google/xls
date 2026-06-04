// Copyright 2026 The XLS Authors
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

#include "xls/dslx/frontend/fuzz_domain_rewriter.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

namespace {

// Extracts the value of the `fuzz_domain` attribute from a struct definition,
// if present.
std::optional<std::string> GetFuzzDomainName(const StructDef* struct_def) {
  if (struct_def == nullptr) {
    return std::nullopt;
  }
  for (const Attribute* attr : struct_def->attributes()) {
    if (attr->attribute_kind() == AttributeKind::kFuzzDomain) {
      return std::get<AttributeData::StringLiteralArgument>(attr->args()[0])
          .text;
    }
  }
  return std::nullopt;
}

// Derives the type annotation for a domain member based on the original struct
// member's type, recursively populating nested domain structs if necessary.
absl::StatusOr<TypeAnnotation*> DeriveDomainMemberTypeAnnotation(
    Module& module, ImportData& import_data, const StructMemberNode* member,
    absl::FunctionRef<absl::Status(StructDef*)> populate_domain_fn) {
  TypeAnnotation* member_type = member->type();
  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_ref,
                       GetStructOrProcRef(member_type, import_data));
  if (!struct_ref.has_value()) {
    return nullptr;
  }
  if (struct_ref->def->kind() != AstNodeKind::kStructDef) {
    return nullptr;
  }

  const StructDef* nested_struct =
      absl::down_cast<const StructDef*>(struct_ref->def);
  std::optional<std::string> domain_name = GetFuzzDomainName(nested_struct);
  if (!domain_name.has_value()) {
    return nullptr;
  }

  Module* nested_module = nested_struct->owner();
  XLS_ASSIGN_OR_RETURN(
      StructDef * nested_domain_struct,
      nested_module->GetMemberOrError<StructDef>(*domain_name));

  XLS_RETURN_IF_ERROR(populate_domain_fn(nested_domain_struct));

  if (!member_type->IsAnnotation<const TypeRefTypeAnnotation>()) {
    return nullptr;
  }

  auto* type_ref_type =
      member_type->AsAnnotation<const TypeRefTypeAnnotation>();
  TypeRef* type_ref = type_ref_type->type_ref();
  TypeDefinition def = type_ref->type_definition();
  if (std::holds_alternative<ColonRef*>(def)) {
    const ColonRef* colon_ref = std::get<ColonRef*>(def);
    ColonRef* domain_colon_ref = module.Make<ColonRef>(
        colon_ref->span(), colon_ref->subject(), *domain_name);
    return module.Make<TypeRefTypeAnnotation>(
        member->span(),
        module.Make<TypeRef>(member->span(), TypeDefinition(domain_colon_ref)),
        std::vector<ExprOrType>(), std::nullopt);
  }

  if (std::holds_alternative<StructDef*>(def)) {
    return module.Make<TypeRefTypeAnnotation>(
        member->span(),
        module.Make<TypeRef>(member->span(),
                             TypeDefinition(nested_domain_struct)),
        std::vector<ExprOrType>(), std::nullopt);
  }

  // TODO(davidplass): also handle TypeAlias

  return nullptr;
}

// Populates an empty fuzz domain struct with members derived from its
// corresponding original struct.
absl::Status PopulateDomainStruct(StructDef* struct_def, Module& module,
                                  ImportData& import_data) {
  if (struct_def == nullptr || !struct_def->is_domain_struct() ||
      !struct_def->members().empty()) {
    return absl::OkStatus();
  }

  StructDef* original =
      dynamic_cast<StructDef*>(struct_def->name_def()->definer());
  XLS_RET_CHECK(original != nullptr);

  for (const StructMemberNode* member : original->members()) {
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * domain_member_type,
                         DeriveDomainMemberTypeAnnotation(
                             module, import_data, member,
                             [&module, &import_data](StructDef* sd) {
                               return PopulateDomainStruct(sd, module,
                                                           import_data);
                             }));

    if (domain_member_type == nullptr) {
      domain_member_type = module.Make<TupleTypeAnnotation>(
          member->span(), std::vector<TypeAnnotation*>());
    }

    NameDef* member_name_def = module.Make<NameDef>(
        member->span(), member->name(), /*definer=*/nullptr);
    StructMemberNode* domain_member =
        module.Make<StructMemberNode>(member->span(), member_name_def,
                                      member->colon_span(), domain_member_type);
    member_name_def->set_definer(domain_member);

    struct_def->AddMember(domain_member);
  }
  return absl::OkStatus();
}

}  // namespace

// Walks the module's top-level members and populates any empty fuzz domain
// structs.
absl::Status RewriteDomainStructs(Module& module, ImportData& import_data) {
  for (const ModuleMember& member : module.top()) {
    if (std::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = std::get<StructDef*>(member);
      if (struct_def->is_domain_struct() && struct_def->members().empty()) {
        XLS_RETURN_IF_ERROR(
            PopulateDomainStruct(struct_def, module, import_data));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xls::dslx
