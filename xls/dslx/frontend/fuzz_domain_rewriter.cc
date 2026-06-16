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
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

namespace {

// Extracts the value of the `fuzz_domain` attribute from a struct definition,
// if present.
std::optional<std::string> GetFuzzDomainName(const StructDef* struct_def) {
  std::optional<Attribute*> attr =
      GetAttribute(struct_def, AttributeKind::kFuzzDomain);
  if (attr.has_value() &&
      (*attr)->attribute_kind() == AttributeKind::kFuzzDomain) {
    return std::get<AttributeData::StringLiteralArgument>((*attr)->args()[0])
        .text;
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
    // Not a struct type, so no domain struct is needed.
    return nullptr;
  }
  XLS_RET_CHECK(struct_ref->def->kind() == AstNodeKind::kStructDef);

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

// If the struct_def is an empty domain struct, populates it with members
// derived from its corresponding original struct.
absl::Status MaybePopulateDomainStruct(StructDef* struct_def, Module& module,
                                       ImportData& import_data) {
  if (!struct_def->is_domain_struct() || !struct_def->members().empty()) {
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
                               return MaybePopulateDomainStruct(sd, module,
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

// For instances of struct domains, populates "missing members" with the
// "arbitrary" domain (i.e., empty tuple.)
class StructInstanceCompleter : public AstNodeVisitorWithDefault {
 public:
  StructInstanceCompleter(Module& module, const ImportData& import_data)
      : module_(module), import_data_(import_data) {}

  absl::Status DefaultHandler(const AstNode* node) override {
    for (auto* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleFuzzTestFunction(const FuzzTestFunction* node) override {
    if (node->domains().has_value()) {
      is_inside_fuzz_test_domain_ = true;
      XLS_RETURN_IF_ERROR((*node->domains())->Accept(this));
      is_inside_fuzz_test_domain_ = false;
    }
    // Don't need to recurse into the function body, as the struct completion
    // doesn't apply there.
    return absl::OkStatus();
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    auto* mutable_node = const_cast<StructInstance*>(node);
    auto struct_or_proc_ref_status =
        GetStructOrProcRef(node->struct_ref(), import_data_);
    if (!struct_or_proc_ref_status.ok()) {
      return absl::OkStatus();
    }

    std::optional<StructOrProcRef> struct_or_proc_ref =
        *struct_or_proc_ref_status;
    if (!struct_or_proc_ref.has_value() ||
        struct_or_proc_ref->def->kind() != AstNodeKind::kStructDef) {
      return absl::OkStatus();
    }

    const StructDef* struct_def =
        absl::down_cast<const StructDef*>(struct_or_proc_ref->def);
    bool is_domain_struct = struct_def->is_domain_struct();
    if (!is_domain_struct) {
      is_domain_struct = is_inside_fuzz_test_domain_;
    }
    if (!is_domain_struct) {
      return absl::OkStatus();
    }

    for (const StructMemberNode* member : struct_def->members()) {
      std::string member_name = member->name();
      if (node->GetExpr(member_name).status().code() ==
          absl::StatusCode::kNotFound) {
        // This struct field hasn't been added to this instance yet, so add it
        // with an empty tuple as its value to indicate "arbitrary" domain.
        Expr* unit_expr =
            module_.Make<XlsTuple>(node->span(), std::vector<Expr*>(),
                                   /*has_trailing_comma=*/false);
        XLS_RETURN_IF_ERROR(mutable_node->AddMember(member_name, unit_expr));
      }
    }
    return absl::OkStatus();
  }

 private:
  // True if `node` is inside a fuzz_test function's domain
  // expression.
  bool is_inside_fuzz_test_domain_ = false;

  Module& module_;
  const ImportData& import_data_;
};

}  // namespace

absl::Status RewriteDomainStructs(Module& module, ImportData& import_data) {
  for (const ModuleMember& member : module.top()) {
    if (std::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = std::get<StructDef*>(member);
      XLS_RETURN_IF_ERROR(
          MaybePopulateDomainStruct(struct_def, module, import_data));
    }
  }

  StructInstanceCompleter completer(module, import_data);
  return WalkPostOrder(&module, &completer, /*want_types=*/false);
}

}  // namespace xls::dslx
