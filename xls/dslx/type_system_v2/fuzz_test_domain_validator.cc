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

#include "xls/dslx/type_system_v2/fuzz_test_domain_validator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

absl::StatusOr<InterpValue> FuzzTestDomainValidator::Evaluate(
    const Expr* expr) {
  const ParametricEnv env = table_.GetParametricEnv(parametric_context_);
  return ConstexprEvaluator::EvaluateToValue(
      const_cast<ImportData*>(&import_data_), const_cast<TypeInfo*>(&ti_),
      &warning_collector_, env, const_cast<Expr*>(expr));
}

absl::StatusOr<const Expr*> FuzzTestDomainValidator::CompleteStructInstance(
    const StructInstance* node, Module& module) {
  XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_ref,
                       GetStructOrProcRef(node->struct_ref(), import_data_));
  XLS_RET_CHECK(struct_ref.has_value());
  const StructDefBase* struct_def = struct_ref->def;

  std::vector<std::pair<std::string, Expr*>> new_members;
  absl::flat_hash_map<std::string_view, Expr*> existing_members;
  for (const auto& [name, expr] : node->members()) {
    existing_members[name] = expr;
  }

  for (const StructMemberNode* formal_member : struct_def->members()) {
    std::string name = formal_member->name();
    auto it = existing_members.find(name);
    if (it != existing_members.end()) {
      XLS_ASSIGN_OR_RETURN(const Expr* completed_member,
                           CompleteExpr(it->second, module));
      new_members.push_back({name, const_cast<Expr*>(completed_member)});
    } else {
      XlsTuple* empty_tuple =
          module.Make<XlsTuple>(node->span(), std::vector<Expr*>(),
                                /*has_multiline_elements=*/false);
      new_members.push_back({name, empty_tuple});
    }
  }

  return module.Make<StructInstance>(
      node->span(), const_cast<TypeAnnotation*>(node->struct_ref()),
      std::move(new_members));
}

absl::StatusOr<const Expr*> FuzzTestDomainValidator::CompleteExpr(
    const Expr* expr, Module& module) {
  if (expr->kind() == AstNodeKind::kStructInstance) {
    return CompleteStructInstance(absl::down_cast<const StructInstance*>(expr),
                                  module);
  }
  if (expr->kind() == AstNodeKind::kXlsTuple) {
    const XlsTuple* tuple = absl::down_cast<const XlsTuple*>(expr);
    std::vector<Expr*> new_members;
    for (const Expr* member : tuple->members()) {
      XLS_ASSIGN_OR_RETURN(const Expr* completed_member,
                           CompleteExpr(member, module));
      new_members.push_back(const_cast<Expr*>(completed_member));
    }
    return module.Make<XlsTuple>(tuple->span(), std::move(new_members),
                                 tuple->has_trailing_comma());
  }
  return expr;
}

absl::Status FuzzTestDomainValidator::RegisterTypes(const Expr* original,
                                                    const Expr* completed,
                                                    TypeInfo& ti) {
  std::optional<const Type*> type = ti.GetItem(original);
  XLS_RET_CHECK(type.has_value());
  ti.SetItem(completed, **type);

  if (original->kind() == AstNodeKind::kStructInstance) {
    const StructInstance* orig_struct =
        absl::down_cast<const StructInstance*>(original);
    const StructInstance* comp_struct =
        absl::down_cast<const StructInstance*>(completed);

    std::optional<const Type*> ref_type = ti.GetItem(orig_struct->struct_ref());
    if (ref_type.has_value()) {
      ti.SetItem(comp_struct->struct_ref(), **ref_type);
    }

    absl::flat_hash_map<std::string_view, Expr*> orig_members;
    for (const auto& [name, expr] : orig_struct->members()) {
      orig_members[name] = expr;
    }

    for (const auto& [name, comp_expr] : comp_struct->members()) {
      auto it = orig_members.find(name);
      if (it != orig_members.end()) {
        XLS_RETURN_IF_ERROR(RegisterTypes(it->second, comp_expr, ti));
      } else {
        ti.SetItem(comp_expr, Type::MakeUnit());
      }
    }
  } else if (original->kind() == AstNodeKind::kXlsTuple) {
    const XlsTuple* orig_tuple = absl::down_cast<const XlsTuple*>(original);
    const XlsTuple* comp_tuple = absl::down_cast<const XlsTuple*>(completed);
    XLS_RET_CHECK_EQ(orig_tuple->members().size(),
                     comp_tuple->members().size());
    for (int i = 0; i < orig_tuple->members().size(); ++i) {
      XLS_RETURN_IF_ERROR(RegisterTypes(orig_tuple->members()[i],
                                        comp_tuple->members()[i], ti));
    }
  }
  return absl::OkStatus();
}

absl::Status FuzzTestDomainValidator::ValidateScalarDomainValue(
    const InterpValue& value, const Type* param_type, const Span& span,
    std::string_view param_str) {
  if (auto* bits_type = dynamic_cast<const BitsType*>(param_type)) {
    if (!value.IsBits()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::StrFormat("Expected bits domain value; got %s",
                          value.ToString()),
          file_table_);
    }
    XLS_ASSIGN_OR_RETURN(int64_t val_bits, value.GetBitCount());
    if (val_bits != bits_type->size().GetAsInt64().value()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Fuzz test domain bit count ($0) does not match "
                           "parameter bit count ($1).",
                           val_bits, bits_type->size().GetAsInt64().value()),
          file_table_);
    }
    if (value.IsSigned() != bits_type->is_signed()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Fuzz test domain signedness ($0) does not match "
                           "parameter signedness ($1).",
                           value.IsSigned() ? "signed" : "unsigned",
                           bits_type->is_signed() ? "signed" : "unsigned"),
          file_table_);
    }
    return absl::OkStatus();
  }
  if (param_type->IsEnum()) {
    if (!value.IsEnum()) {
      return TypeInferenceErrorStatus(
          span, param_type, "Expected enum domain value", file_table_);
    }
    const EnumType* enum_type = absl::down_cast<const EnumType*>(param_type);
    std::optional<InterpValue::EnumData> enum_data = value.GetEnumData();
    XLS_RET_CHECK(enum_data.has_value());
    if (enum_data->def != &enum_type->nominal_type()) {
      return TypeInferenceErrorStatus(span, param_type, "Enum type mismatch",
                                      file_table_);
    }
    return absl::OkStatus();
  }
  return TypeInferenceErrorStatus(
      span, param_type,
      absl::Substitute("Unsupported parameter type for scalar domain: $0",
                       param_type->ToString()),
      file_table_);
}

absl::Status FuzzTestDomainValidator::ValidateTupleDomainValue(
    const InterpValue& value, const TupleType* tuple_type, const Span& span,
    std::string_view param_str) {
  if (!value.IsTuple()) {
    return TypeInferenceErrorStatus(
        span, tuple_type,
        absl::Substitute("Fuzz test domain evaluates to type $0, but "
                         "parameter '$1' expects a tuple.",
                         TagToString(value.tag()), param_str),
        file_table_);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != tuple_type->size()) {
    return TypeInferenceErrorStatus(
        span, tuple_type,
        absl::Substitute("Fuzz test domain tuple size ($0) does not match "
                         "parameter '$1' tuple size ($2).",
                         elements->size(), param_str, tuple_type->size()),
        file_table_);
  }
  for (int i = 0; i < elements->size(); ++i) {
    const Type& member_type = tuple_type->GetMemberType(i);
    std::string element_ctx =
        absl::StrFormat("tuple element %d: %s", i, member_type.ToString());
    XLS_RETURN_IF_ERROR(ValidateFuzzTestDomainValue(
        elements->at(i), &member_type, span, element_ctx));
  }
  return absl::OkStatus();
}

absl::Status FuzzTestDomainValidator::ValidateStructDomainValue(
    const InterpValue& value, const StructType* struct_type, const Span& span,
    std::string_view param_str) {
  if (!value.IsTuple()) {
    return TypeInferenceErrorStatus(
        span, struct_type,
        absl::Substitute("Fuzz test domain evaluates to type $0, but "
                         "parameter '$1' expects a struct.",
                         TagToString(value.tag()), param_str),
        file_table_);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != struct_type->size()) {
    return TypeInferenceErrorStatus(
        span, struct_type,
        absl::Substitute("Fuzz test domain struct size ($0) does not match "
                         "parameter '$1' struct size ($2).",
                         elements->size(), param_str, struct_type->size()),
        file_table_);
  }
  for (int i = 0; i < elements->size(); ++i) {
    const Type& member_type = struct_type->GetMemberType(i);
    std::string field_ctx = absl::StrCat(struct_type->GetMemberName(i), ": ",
                                         member_type.ToString());
    XLS_RETURN_IF_ERROR(ValidateFuzzTestDomainValue(
        elements->at(i), &member_type, span, field_ctx));
  }
  return absl::OkStatus();
}

absl::Status FuzzTestDomainValidator::ValidateArrayDomainValue(
    const InterpValue& value, const ArrayType* array_type, const Span& span,
    std::string_view param_str) {
  if (!value.IsArray() || value.is_range()) {
    return TypeInferenceErrorStatus(
        span, array_type, "Expected array of domains for array parameter",
        file_table_);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != array_type->size().GetAsInt64().value()) {
    return TypeInferenceErrorStatus(
        span, array_type,
        absl::Substitute("Fuzz test domain array size ($0) does not match "
                         "parameter '$1' array size ($2).",
                         elements->size(), param_str,
                         array_type->size().GetAsInt64().value()),
        file_table_);
  }
  const Type& element_type = array_type->element_type();
  for (int i = 0; i < elements->size(); ++i) {
    std::string element_ctx =
        absl::StrFormat("array element %d: %s", i, element_type.ToString());
    XLS_RETURN_IF_ERROR(ValidateFuzzTestDomainValue(
        elements->at(i), &element_type, span, element_ctx));
  }
  return absl::OkStatus();
}

absl::Status FuzzTestDomainValidator::ValidateScalarRangeDomainValue(
    const InterpValue& value, const Type* param_type, const Span& span,
    std::string_view param_str) {
  std::optional<std::shared_ptr<RangeData>> range_data = value.GetRangeData();
  if (range_data.has_value()) {
    XLS_RETURN_IF_ERROR(ValidateScalarDomainValue((*range_data)->start,
                                                  param_type, span, param_str));
    XLS_RETURN_IF_ERROR(ValidateScalarDomainValue((*range_data)->end,
                                                  param_type, span, param_str));
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->empty()) {
    return absl::OkStatus();
  }
  // Only need to look at the first element, since all elements in a range
  // must be the same type.
  return ValidateScalarDomainValue(elements->at(0), param_type, span,
                                   param_str);
}

absl::Status FuzzTestDomainValidator::ValidateScalarElementOfDomainValue(
    const InterpValue& value, const Type* param_type, const Span& span,
    std::string_view param_str) {
  // Value must be an array (ElementOf domain).
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  for (const InterpValue& element : *elements) {
    // Validate each element in the array against the parameter type.
    XLS_RETURN_IF_ERROR(
        ValidateScalarDomainValue(element, param_type, span, param_str));
  }
  return absl::OkStatus();
}

absl::Status FuzzTestDomainValidator::ValidateFuzzTestDomainValue(
    const InterpValue& value, const Type* param_type, const Span& span,
    std::string_view param_str) {
  if (value.IsTuple()) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         value.GetValues());
    if (elements->empty()) {
      // Empty tuple is the Arbitrary domain.
      return absl::OkStatus();
    }
    if (!param_type->IsTuple() && !param_type->IsStruct()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          "Fuzz test domain implies a tuple type, but parameter is not a "
          "tuple.",
          file_table_);
    }
  }

  if (param_type->IsTuple()) {
    return ValidateTupleDomainValue(
        value, absl::down_cast<const TupleType*>(param_type), span, param_str);
  }
  if (param_type->IsStruct()) {
    return ValidateStructDomainValue(
        value, absl::down_cast<const StructType*>(param_type), span, param_str);
  }
  if (param_type->IsArray()) {
    return ValidateArrayDomainValue(
        value, absl::down_cast<const ArrayType*>(param_type), span, param_str);
  }
  if (value.is_range()) {
    return ValidateScalarRangeDomainValue(value, param_type, span, param_str);
  }
  if (!value.IsArray()) {
    return TypeInferenceErrorStatus(
        span, param_type,
        absl::Substitute("Expected range or set domain for scalar "
                         "parameter $0; got type $1",
                         param_str, TagToString(value.tag())),
        file_table_);
  }
  return ValidateScalarElementOfDomainValue(value, param_type, span, param_str);
}

absl::Status FuzzTestDomainValidator::Validate(const Expr* domain,
                                               const Type* param_type,
                                               std::string_view param_str) {
  Module* module = domain->owner();
  XLS_RET_CHECK(module != nullptr);
  XLS_ASSIGN_OR_RETURN(const Expr* completed_domain,
                       CompleteExpr(domain, *module));
  if (completed_domain != domain) {
    XLS_RETURN_IF_ERROR(
        RegisterTypes(domain, completed_domain, const_cast<TypeInfo&>(ti_)));
  }
  absl::StatusOr<InterpValue> value_or = Evaluate(completed_domain);
  if (!value_or.ok()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::StrFormat("Fuzz test domain must be a constexpr expression; "
                        "evaluation failed: %s",
                        value_or.status().message()),
        file_table_);
  }
  const InterpValue& value = *value_or;

  absl::Status status =
      ValidateFuzzTestDomainValue(value, param_type, domain->span(), param_str);
  if (!status.ok()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::Substitute("Fuzz test domain `$0` is not "
                         "compatible with parameter `$1`: $2",
                         domain->ToString(), param_str, status.message()),
        file_table_);
  }
  return absl::OkStatus();
}

}  // namespace xls::dslx
