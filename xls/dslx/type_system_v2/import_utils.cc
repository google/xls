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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

// Recursive visitor that unwraps a possible chain of type aliasing leading to a
// struct, impl-style proc, or enum def. After visiting a subtree, the result
// can be obtained by calling `GetStructOrProcRef` or `GetEnumDef`.
class TypeRefUnwrapper : public AstNodeVisitorWithDefault {
 public:
  explicit TypeRefUnwrapper(const ImportData& import_data)
      : import_data_(import_data) {}

  absl::Status HandleColonRef(const ColonRef* colon_ref) override {
    XLS_ASSIGN_OR_RETURN(std::optional<ModuleInfo*> import_module,
                         GetImportedModuleInfo(colon_ref, import_data_));
    if (import_module.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          ModuleMember member,
          GetPublicModuleMember((*import_module)->module(), colon_ref,
                                import_data_.file_table()));
      return ToAstNode(member)->Accept(this);
    }
    // The entire `ColonRef` is not a struct reference otherwise. We don't want
    // to produce anything for a `ColonRef` to a member of a struct.
    return absl::OkStatus();
  }

  absl::Status HandleTypeAlias(const TypeAlias* alias) override {
    return alias->type_annotation().Accept(this);
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* annotation) override {
    if (!annotation->parametrics().empty() && !parametrics_.empty()) {
      return TypeInferenceErrorStatus(
          annotation->span(), /* type= */ nullptr,
          absl::StrFormat(
              "Parametric values defined multiple times for annotation: `%s`",
              annotation->ToString()),
          import_data_.file_table());
    }

    if (!type_ref_type_annotation_.has_value()) {
      type_ref_type_annotation_ = annotation;
    }
    if (!annotation->parametrics().empty()) {
      parametrics_ = annotation->parametrics();
    }
    if (!instantiator_.has_value()) {
      instantiator_ = annotation->instantiator();
    }
    return ToAstNode(annotation->type_ref()->type_definition())->Accept(this);
  }

  absl::Status HandleProcDef(const ProcDef* def) override {
    type_def_ = const_cast<ProcDef*>(def);
    return absl::OkStatus();
  }

  absl::Status HandleStructDef(const StructDef* def) override {
    type_def_ = const_cast<StructDef*>(def);
    return absl::OkStatus();
  }

  absl::Status HandleEnumDef(const EnumDef* def) override {
    type_def_ = const_cast<EnumDef*>(def);
    return absl::OkStatus();
  }

  absl::Status HandleNameDef(const NameDef* name_def) override {
    return name_def->definer()->Accept(this);
  }

  absl::Status HandleNameRef(const NameRef* name_ref) override {
    return ToAstNode(name_ref->name_def())->Accept(this);
  }

  std::optional<StructOrProcRef> GetStructOrProcRef() {
    if (!type_def_.has_value() ||
        (!std::holds_alternative<StructDef*>(*type_def_) &&
         !std::holds_alternative<ProcDef*>(*type_def_))) {
      return std::nullopt;
    }
    return StructOrProcRef{
        .def = down_cast<StructDefBase*>(ToAstNode(*type_def_)),
        .parametrics = parametrics_,
        .instantiator = instantiator_,
        .type_ref_type_annotation = type_ref_type_annotation_};
  }

  std::optional<const EnumDef*> GetEnumDef() {
    return type_def_.has_value() && std::holds_alternative<EnumDef*>(*type_def_)
               ? std::make_optional(std::get<EnumDef*>(*type_def_))
               : std::nullopt;
  }

 private:
  const ImportData& import_data_;

  // These fields get populated as we visit nodes.
  std::vector<ExprOrType> parametrics_;
  std::optional<TypeDefinition> type_def_;
  std::optional<const TypeRefTypeAnnotation*> type_ref_type_annotation_;
  std::optional<const StructInstanceBase*> instantiator_;
};

bool IsAbstractStructOrProcRef(const StructOrProcRef& ref) {
  return GetRequiredParametricBindings(ref.def->parametric_bindings()).size() >
         ref.parametrics.size();
}

}  // namespace

absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  if (!annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
    return std::nullopt;
  }
  TypeRefUnwrapper unwrapper(import_data);
  XLS_RETURN_IF_ERROR(annotation->Accept(&unwrapper));
  return unwrapper.GetStructOrProcRef();
}

absl::StatusOr<std::optional<StructOrProcRef>> GetStructOrProcRefForSubject(
    const ColonRef* ref, const ImportData& import_data) {
  TypeRefUnwrapper unwrapper(import_data);
  XLS_RETURN_IF_ERROR(ToAstNode(ref->subject())->Accept(&unwrapper));
  return unwrapper.GetStructOrProcRef();
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
  TypeRefUnwrapper unwrapper(import_data);
  XLS_RETURN_IF_ERROR(colon_ref->Accept(&unwrapper));
  return unwrapper.GetStructOrProcRef();
}

absl::StatusOr<std::optional<const StructDefBase*>> GetStructOrProcDef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> ref,
                       GetStructOrProcRef(annotation, import_data));
  return ref.has_value() ? std::make_optional(ref->def) : std::nullopt;
}

absl::StatusOr<bool> IsReferenceToAbstractType(const AstNode* node,
                                               const ImportData& import_data,
                                               const InferenceTable& table) {
  std::optional<StructOrProcRef> ref;
  if (node->kind() == AstNodeKind::kColonRef &&
      IsColonRefWithTypeTarget(table, down_cast<const ColonRef*>(node))) {
    XLS_ASSIGN_OR_RETURN(
        ref, GetStructOrProcRef(down_cast<const ColonRef*>(node), import_data));
  } else if (node->kind() == AstNodeKind::kTypeAlias ||
             (node->kind() == AstNodeKind::kNameDef &&
              node->parent() != nullptr &&
              node->parent()->kind() == AstNodeKind::kTypeAlias)) {
    const TypeAlias* alias = node->kind() == AstNodeKind::kTypeAlias
                                 ? down_cast<const TypeAlias*>(node)
                                 : down_cast<const TypeAlias*>(node->parent());
    XLS_ASSIGN_OR_RETURN(
        ref, GetStructOrProcRef(&alias->type_annotation(), import_data));
  }
  return ref.has_value() && IsAbstractStructOrProcRef(*ref);
}

absl::StatusOr<std::optional<const EnumDef*>> GetEnumDef(
    const TypeAnnotation* annotation, const ImportData& import_data) {
  if (!annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
    return std::nullopt;
  }
  TypeRefUnwrapper unwrapper(import_data);
  XLS_RETURN_IF_ERROR(annotation->Accept(&unwrapper));
  return unwrapper.GetEnumDef();
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
