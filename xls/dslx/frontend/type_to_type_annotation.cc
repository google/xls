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

#include "xls/dslx/frontend/type_to_type_annotation.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

absl::StatusOr<std::string> FindImportAlias(const Module& module,
                                            const TypeInfo& type_info,
                                            const Module* target_module,
                                            std::string_view nominal_type) {
  for (const auto& [name, import_node] : module.GetImportByName()) {
    auto imported_info = type_info.GetImported(import_node);
    if (imported_info.has_value() &&
        imported_info.value()->module == target_module) {
      return name;
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Could not find import of type %s in module %s",
                      nominal_type, target_module->name()));
}

template <typename NominalDefT>
absl::StatusOr<TypeAnnotation*> CreateNominalTypeAnnotation(
    Module& new_module, const Module* source_module, const TypeInfo* type_info,
    const NominalDefT& nominal_type, const Span& span) {
  const Module* type_owner = nominal_type.owner();
  if (source_module == nullptr || type_owner == source_module) {
    XLS_ASSIGN_OR_RETURN(
        TypeDefinition type_def,
        new_module.GetTypeDefinition(nominal_type.identifier()));
    TypeRef* type_ref = new_module.Make<TypeRef>(span, type_def);
    return new_module.Make<TypeRefTypeAnnotation>(span, type_ref,
                                                  std::vector<ExprOrType>{});
  }

  if (type_info == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot resolve imported nominal type %s without type info",
        nominal_type.identifier()));
  }

  XLS_ASSIGN_OR_RETURN(std::string import_alias,
                       FindImportAlias(*source_module, *type_info, type_owner,
                                       nominal_type.identifier()));

  auto imports = new_module.GetImportByName();
  auto it = imports.find(import_alias);
  if (it == imports.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Could not find import %s in new module", import_alias));
  }
  Import* new_import = it->second;

  NameRef* subject =
      new_module.Make<NameRef>(span, import_alias, &new_import->name_def());
  ColonRef* colon_ref =
      new_module.Make<ColonRef>(span, subject, nominal_type.identifier());
  TypeRef* type_ref = new_module.Make<TypeRef>(span, colon_ref);
  return new_module.Make<TypeRefTypeAnnotation>(span, type_ref,
                                                std::vector<ExprOrType>{});
}

class TypeCloner : public TypeVisitorWithDefault {
 public:
  TypeCloner(Module& new_module, const Span& span,
             const Module* source_module = nullptr,
             const TypeInfo* type_info = nullptr)
      : new_module_(new_module),
        span_(span),
        source_module_(source_module),
        type_info_(type_info) {}

  absl::Status HandleBits(const BitsType& t) override {
    XLS_ASSIGN_OR_RETURN(int64_t width, t.size().GetAsInt64());
    XLS_ASSIGN_OR_RETURN(BuiltinType builtin_type,
                         GetBuiltinType(t.is_signed(), width));
    type_annotation_ = new_module_.Make<BuiltinTypeAnnotation>(
        span_, builtin_type,
        new_module_.GetOrCreateBuiltinNameDef(builtin_type));
    return absl::OkStatus();
  }

  absl::Status HandleChannel(const ChannelType& t) override {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * payload,
        CreateTypeAnnotation(new_module_, t.payload_type(), span_,
                             source_module_, type_info_));
    type_annotation_ = new_module_.Make<ChannelTypeAnnotation>(
        span_, t.direction(), payload, /*dims=*/std::nullopt);
    return absl::OkStatus();
  }

  absl::Status HandleToken(const TokenType& t) override {
    type_annotation_ = new_module_.Make<BuiltinTypeAnnotation>(
        span_, BuiltinType::kToken,
        new_module_.GetOrCreateBuiltinNameDef(BuiltinType::kToken));
    return absl::OkStatus();
  }

  absl::Status HandleTuple(const TupleType& t) override {
    std::vector<TypeAnnotation*> element_annotations;
    for (const auto& element_type : t.members()) {
      XLS_ASSIGN_OR_RETURN(
          TypeAnnotation * element_annot,
          CreateTypeAnnotation(new_module_, *element_type, span_,
                               source_module_, type_info_));
      element_annotations.push_back(element_annot);
    }
    type_annotation_ =
        new_module_.Make<TupleTypeAnnotation>(span_, element_annotations);
    return absl::OkStatus();
  }

  absl::Status HandleArray(const ArrayType& t) override {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * element_annot,
        CreateTypeAnnotation(new_module_, t.element_type(), span_,
                             source_module_, type_info_));
    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    Number* dim_expr = new_module_.Make<Number>(
        span_, absl::StrCat(size), NumberKind::kOther, /*type=*/nullptr);
    type_annotation_ =
        new_module_.Make<ArrayTypeAnnotation>(span_, element_annot, dim_expr);
    return absl::OkStatus();
  }

  absl::Status HandleStruct(const StructType& t) override {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * ta,
        CreateNominalTypeAnnotation(new_module_, source_module_, type_info_,
                                    t.nominal_type(), span_));
    type_annotation_ = ta;
    return absl::OkStatus();
  }

  absl::Status HandleEnum(const EnumType& t) override {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * ta,
        CreateNominalTypeAnnotation(new_module_, source_module_, type_info_,
                                    t.nominal_type(), span_));
    type_annotation_ = ta;
    return absl::OkStatus();
  }

  TypeAnnotation* type_annotation() { return type_annotation_; }

 private:
  TypeAnnotation* type_annotation_ = nullptr;
  Module& new_module_;
  const Span& span_;
  const Module* source_module_;
  const TypeInfo* type_info_;
};
}  // namespace

absl::StatusOr<TypeAnnotation*> CreateTypeAnnotation(
    Module& new_module, const Type& type, const Span& span,
    const Module* source_module, const TypeInfo* type_info) {
  TypeCloner cloner(new_module, span, source_module, type_info);
  XLS_RETURN_IF_ERROR(type.Accept(cloner));
  return cloner.type_annotation();
}

}  // namespace xls::dslx
