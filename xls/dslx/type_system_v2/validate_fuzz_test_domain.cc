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

#include "xls/dslx/type_system_v2/validate_fuzz_test_domain.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

// Evaluates the given expression to an InterpValue.
absl::StatusOr<InterpValue> Evaluate(const Expr* expr, const TypeInfo& ti,
                                     WarningCollector& warning_collector,
                                     const ImportData& import_data) {
  ParametricEnv env;
  // We unfortunately need to const cast the import data and type info because
  // constexpr evaluator modifies them both in certain situations.
  return ConstexprEvaluator::EvaluateToValue(
      const_cast<ImportData*>(&import_data), const_cast<TypeInfo*>(&ti),
      &warning_collector, env, const_cast<Expr*>(expr));
}

absl::Status ValidateScalarDomainValue(const InterpValue& domain_value,
                                       const Type* param_type, const Span& span,
                                       std::string_view param_str,
                                       const FileTable& file_table) {
  if (auto* bits_type = dynamic_cast<const BitsType*>(param_type)) {
    if (!domain_value.IsBits()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::StrFormat("Expected bits domain value; got %s",
                          domain_value.ToString()),
          file_table);
    }
    XLS_ASSIGN_OR_RETURN(int64_t domain_bit_count, domain_value.GetBitCount());
    if (domain_bit_count != bits_type->size().GetAsInt64().value()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Fuzz test domain bit count ($0) does not match "
                           "parameter bit count ($1).",
                           domain_bit_count,
                           bits_type->size().GetAsInt64().value()),
          file_table);
    }
    if (domain_value.IsSigned() != bits_type->is_signed()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Fuzz test domain signedness ($0) does not match "
                           "parameter signedness ($1).",
                           domain_value.IsSigned() ? "signed" : "unsigned",
                           bits_type->is_signed() ? "signed" : "unsigned"),
          file_table);
    }
  } else if (param_type->IsEnum()) {
    if (!domain_value.IsEnum()) {
      return TypeInferenceErrorStatus(span, param_type,
                                      "Expected enum domain value", file_table);
    }
    const EnumType& enum_type = param_type->AsEnum();
    std::optional<InterpValue::EnumData> enum_data = domain_value.GetEnumData();
    XLS_RET_CHECK(enum_data.has_value());
    if (enum_data->def != &enum_type.nominal_type()) {
      return TypeInferenceErrorStatus(span, param_type, "Enum type mismatch",
                                      file_table);
    }
  } else {
    return TypeInferenceErrorStatus(
        span, param_type,
        absl::Substitute("Unsupported parameter type for scalar domain: $0",
                         param_type->ToString()),
        file_table);
  }
  return absl::OkStatus();
}

absl::Status ValidateDomainValue(const InterpValue& value,
                                 const Type* param_type, const Span& span,
                                 std::string_view param_str,
                                 const FileTable& file_table);

absl::Status ValidateTupleDomainValue(const InterpValue& value,
                                      const TupleType& param_type,
                                      const Span& span,
                                      std::string_view param_str,
                                      const FileTable& file_table) {
  if (!value.IsTuple()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Fuzz test domain evaluates to type $0, but "
                         "parameter '$1' expects a tuple.",
                         TagToString(value.tag()), param_str),
        file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != param_type.size()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Fuzz test domain tuple size ($0) does not match "
                         "parameter '$1' tuple size ($2).",
                         elements->size(), param_str, param_type.size()),
        file_table);
  }
  for (int i = 0; i < elements->size(); ++i) {
    const Type& param_member = param_type.GetMemberType(i);
    std::string element_ctx =
        absl::StrFormat("tuple element %d: %s", i, param_member.ToString());
    XLS_RETURN_IF_ERROR(ValidateDomainValue(elements->at(i), &param_member,
                                            span, element_ctx, file_table));
  }
  return absl::OkStatus();
}

absl::Status ValidateStructDomainValue(const InterpValue& value,
                                       const StructType& param_type,
                                       const Span& span,
                                       std::string_view param_str,
                                       const FileTable& file_table) {
  if (!value.IsTuple()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Fuzz test domain evaluates to type $0, but "
                         "parameter '$1' expects a struct.",
                         TagToString(value.tag()), param_str),
        file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != param_type.size()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Fuzz test domain struct size ($0) does not match "
                         "parameter '$1' struct size ($2).",
                         elements->size(), param_str, param_type.size()),
        file_table);
  }
  for (int i = 0; i < elements->size(); ++i) {
    const Type& param_member = param_type.GetMemberType(i);
    std::string field_ctx = absl::StrCat(param_type.GetMemberName(i), ": ",
                                         param_member.ToString());
    XLS_RETURN_IF_ERROR(ValidateDomainValue(elements->at(i), &param_member,
                                            span, field_ctx, file_table));
  }
  return absl::OkStatus();
}

absl::Status ValidateArrayDomainValue(const InterpValue& value,
                                      const ArrayType& param_type,
                                      const Span& span,
                                      std::string_view param_str,
                                      const FileTable& file_table) {
  if (!value.IsArray() || value.is_range()) {
    return TypeInferenceErrorStatus(
        span, &param_type, "Expected array of domains for array parameter",
        file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != param_type.size().GetAsInt64().value()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Fuzz test domain array size ($0) does not match "
                         "parameter '$1' array size ($2).",
                         elements->size(), param_str,
                         param_type.size().GetAsInt64().value()),
        file_table);
  }
  const Type& param_element_type = param_type.element_type();
  for (int i = 0; i < elements->size(); ++i) {
    std::string element_ctx = absl::StrFormat("array element %d: %s", i,
                                              param_element_type.ToString());
    XLS_RETURN_IF_ERROR(ValidateDomainValue(
        elements->at(i), &param_element_type, span, element_ctx, file_table));
  }
  return absl::OkStatus();
}

absl::Status ValidateScalarRangeDomainValue(const InterpValue& value,
                                            const Type* param_type,
                                            const Span& span,
                                            std::string_view param_str,
                                            const FileTable& file_table) {
  std::optional<std::shared_ptr<RangeData>> range_data = value.GetRangeData();
  if (range_data.has_value()) {
    XLS_RETURN_IF_ERROR(ValidateScalarDomainValue(
        (*range_data)->start, param_type, span, param_str, file_table));
    XLS_RETURN_IF_ERROR(ValidateScalarDomainValue(
        (*range_data)->end, param_type, span, param_str, file_table));
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->empty()) {
    return absl::OkStatus();
  }
  return ValidateScalarDomainValue(elements->at(0), param_type, span, param_str,
                                   file_table);
}

absl::Status ValidateScalarElementOfDomainValue(const InterpValue& value,
                                                const Type* param_type,
                                                const Span& span,
                                                std::string_view param_str,
                                                const FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  for (const InterpValue& element : *elements) {
    XLS_RETURN_IF_ERROR(ValidateScalarDomainValue(element, param_type, span,
                                                  param_str, file_table));
  }
  return absl::OkStatus();
}

absl::Status ValidateDomainValue(const InterpValue& domain_value,
                                 const Type* param_type, const Span& span,
                                 std::string_view param_str,
                                 const FileTable& file_table) {
  if (domain_value.IsTuple()) {
    auto values_or = domain_value.GetValues();
    if (values_or.ok() && values_or.value()->empty()) {
      // Empty tuple value always matches (represents omitted/unconstrained
      // domain).
      return absl::OkStatus();
    }
  }

  if (param_type->IsStruct()) {
    return ValidateStructDomainValue(domain_value, param_type->AsStruct(), span,
                                     param_str, file_table);
  }

  if (param_type->IsTuple()) {
    return ValidateTupleDomainValue(domain_value, param_type->AsTuple(), span,
                                    param_str, file_table);
  }

  if (param_type->IsArray()) {
    return ValidateArrayDomainValue(domain_value, param_type->AsArray(), span,
                                    param_str, file_table);
  }

  if (domain_value.is_range()) {
    return ValidateScalarRangeDomainValue(domain_value, param_type, span,
                                          param_str, file_table);
  }
  if (!domain_value.IsArray()) {
    return TypeInferenceErrorStatus(
        span, param_type,
        absl::Substitute("Expected range or array domain for scalar "
                         "parameter $0; got $1",
                         param_str, TagToString(domain_value.tag())),
        file_table);
  }
  return ValidateScalarElementOfDomainValue(domain_value, param_type, span,
                                            param_str, file_table);
}

}  // namespace

absl::Status ValidateFuzzTestDomain(const Expr* domain, const Type* param_type,
                                    std::string_view param_str,
                                    const InferenceTable& table,
                                    const TypeInfo& ti,
                                    WarningCollector& warning_collector,
                                    const ImportData& import_data,
                                    const FileTable& file_table) {
  std::optional<Type*> maybe_domain_type = ti.GetItem(domain);
  XLS_RET_CHECK(maybe_domain_type.has_value());
  const Type* domain_type = *maybe_domain_type;

  // Top-level structural checks using domain_type
  if (domain_type->IsStruct() && !param_type->IsStruct()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::Substitute("Fuzz test domain `$0` implies a struct type, but "
                         "parameter `$1` is of type $2.",
                         domain->ToString(), param_str, param_type->ToString()),
        file_table);
  }
  if (domain_type->IsTuple()) {
    const TupleType& tuple_type = domain_type->AsTuple();
    if (tuple_type.empty()) {
      // Empty tuple domain is allowed for any parameter type.
      return absl::OkStatus();
    }
    if (!param_type->IsTuple()) {
      return TypeInferenceErrorStatus(
          domain->span(), param_type,
          absl::Substitute("Fuzz test domain `$0` implies a tuple type, but "
                           "parameter `$1` is of type $2.",
                           domain->ToString(), param_str,
                           param_type->ToString()),
          file_table);
    }
  }

  if (!domain_type->IsArray() && !domain_type->IsTuple() &&
      !domain_type->IsStruct()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::Substitute("Unsupported fuzz test domain `$0` of type `$1`.",
                         domain->ToString(), domain_type->ToString()),
        file_table);
  }

  absl::StatusOr<InterpValue> domain_value_status =
      Evaluate(domain, ti, warning_collector, import_data);
  if (!domain_value_status.ok()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::StrFormat("Fuzz test domain must be a constexpr expression; "
                        "evaluation failed: %s",
                        domain_value_status.status().message()),
        file_table);
  }
  const InterpValue& domain_value = *domain_value_status;

  return ValidateDomainValue(domain_value, param_type, domain->span(),
                             param_str, file_table);
}

}  // namespace xls::dslx
