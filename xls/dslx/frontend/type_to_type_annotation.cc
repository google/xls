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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {
class TypeCloner : public TypeVisitorWithDefault {
 public:
  TypeCloner(Module& new_module, const Span& span)
      : new_module_(new_module), span_(span) {}

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
        CreateTypeAnnotation(new_module_, t.payload_type(), span_));
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
          CreateTypeAnnotation(new_module_, *element_type, span_));
      element_annotations.push_back(element_annot);
    }
    type_annotation_ =
        new_module_.Make<TupleTypeAnnotation>(span_, element_annotations);
    return absl::OkStatus();
  }

  absl::Status HandleArray(const ArrayType& t) override {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * element_annot,
        CreateTypeAnnotation(new_module_, t.element_type(), span_));
    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    Number* dim_expr = new_module_.Make<Number>(
        span_, absl::StrCat(size), NumberKind::kOther, /*type=*/nullptr);
    type_annotation_ =
        new_module_.Make<ArrayTypeAnnotation>(span_, element_annot, dim_expr);
    return absl::OkStatus();
  }

  absl::Status HandleStruct(const StructType& t) override {
    std::string name = t.nominal_type().identifier();
    XLS_ASSIGN_OR_RETURN(TypeDefinition type_def,
                         new_module_.GetTypeDefinition(name));
    TypeRef* type_ref = new_module_.Make<TypeRef>(span_, type_def);
    type_annotation_ = new_module_.Make<TypeRefTypeAnnotation>(
        span_, type_ref, std::vector<ExprOrType>{});
    return absl::OkStatus();
  }

  absl::Status HandleEnum(const EnumType& t) override {
    std::string name = t.nominal_type().identifier();
    XLS_ASSIGN_OR_RETURN(TypeDefinition type_def,
                         new_module_.GetTypeDefinition(name));
    TypeRef* type_ref = new_module_.Make<TypeRef>(span_, type_def);
    type_annotation_ = new_module_.Make<TypeRefTypeAnnotation>(
        span_, type_ref, std::vector<ExprOrType>{});
    return absl::OkStatus();
  }

  TypeAnnotation* type_annotation() { return type_annotation_; }

 private:
  TypeAnnotation* type_annotation_ = nullptr;
  Module& new_module_;
  const Span& span_;
};
}  // namespace

absl::StatusOr<TypeAnnotation*> CreateTypeAnnotation(Module& new_module,
                                                     const Type& type,
                                                     const Span& span) {
  TypeCloner cloner(new_module, span);
  XLS_RETURN_IF_ERROR(type.Accept(cloner));
  return cloner.type_annotation();
}

}  // namespace xls::dslx
