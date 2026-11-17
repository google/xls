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

#include "xls/dslx/domain_utils.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
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
    XLS_ASSIGN_OR_RETURN(int64_t bits_size, bits_type->size().GetAsInt64());
    if (domain_bit_count != bits_size) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Domain bit count ($0) does not match parameter bit "
                           "count ($1).",
                           domain_bit_count, bits_size),
          file_table);
    }
    if (domain_value.IsSigned() != bits_type->is_signed()) {
      return TypeInferenceErrorStatus(
          span, param_type,
          absl::Substitute("Domain signedness ($0) does not match parameter "
                           "signedness ($1).",
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
        absl::Substitute("Domain evaluates to type $0, but parameter '$1' "
                         "expects a tuple.",
                         TagToString(value.tag()), param_str),
        file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != param_type.size()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Domain tuple size ($0) does not match parameter '$1' "
                         "tuple size ($2).",
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
        absl::Substitute("Domain evaluates to type $0, but parameter '$1' "
                         "expects a struct.",
                         TagToString(value.tag()), param_str),
        file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  if (elements->size() != param_type.size()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Domain struct size ($0) does not match parameter "
                         "'$1' struct size ($2).",
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
  if ((!value.IsArray() && !value.IsTuple()) || value.is_range()) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        "Expected array or tuple of domains for array parameter", file_table);
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       value.GetValues());
  XLS_ASSIGN_OR_RETURN(int64_t param_array_size,
                       param_type.size().GetAsInt64());
  if (elements->size() > param_array_size) {
    return TypeInferenceErrorStatus(
        span, &param_type,
        absl::Substitute("Domain array size ($0) exceeds parameter '$1' array "
                         "size ($2).",
                         elements->size(), param_str, param_array_size),
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

// Returns true for true array types (excluding `xN[S][N]` bit-vector types,
// which are represented as an ArrayType over BitsConstructorType).
bool IsCompositeArray(const Type& type) {
  return type.IsArray() && !GetBitsLike(type).has_value();
}

absl::Status ValidateMaxPoints(int64_t max_points) {
  if (max_points <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("max_points must be positive; got %d", max_points));
  }
  return absl::OkStatus();
}

absl::Status ValidateArraySize(std::string_view param_name, int64_t array_size,
                               int64_t max_points) {
  if (array_size > max_points) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Array parameter `%s` has size %d, which exceeds the maximum domain "
        "size of %d",
        param_name, array_size, max_points));
  }
  return absl::OkStatus();
}

absl::Status ValidateScalarDomainElement(std::string_view param_name,
                                         const Type& target_type,
                                         const InterpValue& val) {
  if (std::optional<BitsLikeProperties> bits_props = GetBitsLike(target_type);
      bits_props.has_value()) {
    if (!val.IsBits()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected bits domain value for parameter `%s`; got `%s`", param_name,
          val.ToString()));
    }
    XLS_ASSIGN_OR_RETURN(int64_t val_bits, val.GetBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t target_bits, bits_props->size.GetAsInt64());
    if (val_bits != target_bits) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain bit count (%d) does not match parameter `%s` bit count (%d)",
          val_bits, param_name, target_bits));
    }
    XLS_ASSIGN_OR_RETURN(bool target_signed, bits_props->is_signed.GetAsBool());
    if (val.IsSigned() != target_signed) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain signedness (%s) does not match parameter `%s` signedness "
          "(%s)",
          val.IsSigned() ? "signed" : "unsigned", param_name,
          target_signed ? "signed" : "unsigned"));
    }
    return absl::OkStatus();
  }
  if (target_type.IsEnum()) {
    if (!val.IsEnum()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected enum domain value for parameter `%s`; got `%s`", param_name,
          val.ToString()));
    }
    const EnumType& enum_type = target_type.AsEnum();
    std::optional<InterpValue::EnumData> enum_data = val.GetEnumData();
    XLS_RET_CHECK(enum_data.has_value());
    if (enum_data->def != &enum_type.nominal_type()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Enum domain type mismatch for parameter `%s`; expected `%s`",
          param_name, enum_type.ToString()));
    }
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unsupported parameter type `%s` for scalar domain of parameter `%s`",
      target_type.ToString(), param_name));
}

absl::Status ExtendProduct(std::string_view context_name,
                           std::string_view param_name, int64_t max_points,
                           std::vector<std::vector<InterpValue>>& product,
                           absl::Span<const InterpValue> factor) {
  if (factor.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Parameter `%s` has an empty domain", param_name));
  }
  if (static_cast<int64_t>(product.size()) >
      max_points / static_cast<int64_t>(factor.size())) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input domain for `%s` exceeds maximum size of %d "
                        "points",
                        context_name, max_points));
  }
  if (factor.size() == 1) {
    for (std::vector<InterpValue>& prefix : product) {
      prefix.push_back(factor[0]);
    }
    return absl::OkStatus();
  }
  std::vector<std::vector<InterpValue>> next;
  next.reserve(product.size() * factor.size());
  for (const std::vector<InterpValue>& prefix : product) {
    for (const InterpValue& val : factor) {
      std::vector<InterpValue> combined = prefix;
      combined.push_back(val);
      next.push_back(std::move(combined));
    }
  }
  product = std::move(next);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<InterpValue>> WrapProductAsTuples(
    std::vector<std::vector<InterpValue>> product) {
  std::vector<InterpValue> domain;
  domain.reserve(product.size());
  for (std::vector<InterpValue>& elems : product) {
    domain.push_back(InterpValue::MakeTuple(std::move(elems)));
  }
  return domain;
}

absl::StatusOr<std::vector<InterpValue>> WrapProductAsArrays(
    std::vector<std::vector<InterpValue>> product) {
  std::vector<InterpValue> domain;
  domain.reserve(product.size());
  for (std::vector<InterpValue>& elems : product) {
    XLS_ASSIGN_OR_RETURN(InterpValue arr,
                         InterpValue::MakeArray(std::move(elems)));
    domain.push_back(std::move(arr));
  }
  return domain;
}

absl::StatusOr<std::vector<InterpValue>> EnumerateRangeBounds(
    std::string_view param_name, const Type& target_type,
    const InterpValue& start, const InterpValue& end, bool inclusive,
    int64_t max_points) {
  XLS_RETURN_IF_ERROR(
      ValidateScalarDomainElement(param_name, target_type, start));
  XLS_RETURN_IF_ERROR(
      ValidateScalarDomainElement(param_name, target_type, end));
  XLS_ASSIGN_OR_RETURN(InterpValue gt, start.Gt(end));
  if (gt.IsTrue() || (!inclusive && start.Eq(end))) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Parameter `%s` has an empty domain", param_name));
  }
  XLS_ASSIGN_OR_RETURN(InterpValue diff, end.Sub(start));
  diff = InterpValue::MakeUnsigned(diff.GetBitsOrDie());
  if (inclusive) {
    XLS_ASSIGN_OR_RETURN(diff, diff.IncrementZeroExtendIfOverflow());
  }
  if (!diff.FitsInUint64()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input domain for `%s` exceeds maximum size of %d "
                        "points",
                        param_name, max_points));
  }
  XLS_ASSIGN_OR_RETURN(uint64_t count, diff.GetBitValueUnsigned());
  if (count > static_cast<uint64_t>(max_points)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input domain for `%s` exceeds maximum size of %d "
                        "points",
                        param_name, max_points));
  }

  XLS_ASSIGN_OR_RETURN(int64_t bit_count, start.GetBitCount());
  InterpValue one = start.IsSigned() ? InterpValue::MakeSBits(bit_count, 1)
                                     : InterpValue::MakeUBits(bit_count, 1);
  std::vector<InterpValue> domain;
  domain.reserve(count);
  InterpValue cur = start;
  for (uint64_t i = 0; i < count; ++i) {
    domain.push_back(cur);
    if (i + 1 < count) {
      XLS_ASSIGN_OR_RETURN(cur, cur.Add(one));
    }
  }
  return domain;
}

absl::StatusOr<std::vector<InterpValue>> EnumerateValueDomain(
    std::string_view param_name, const Type& target_type,
    const InterpValue& val, int64_t max_points) {
  if (val.IsTuple()) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    if (elements->empty()) {
      return EnumerateArbitraryDomain(param_name, target_type, max_points,
                                      /*allow_composite=*/true);
    }
  }

  if (target_type.IsStruct()) {
    if (!val.IsTuple()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain evaluates to type `%s`, but parameter `%s` expects a struct",
          TagToString(val.tag()), param_name));
    }
    const StructType& struct_type = target_type.AsStruct();
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    if (elements->size() != struct_type.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain struct size (%d) does not match parameter `%s` struct size "
          "(%d)",
          elements->size(), param_name, struct_type.size()));
    }
    std::vector<std::vector<InterpValue>> product = {{}};
    for (size_t i = 0; i < elements->size(); ++i) {
      std::string field_ctx =
          absl::StrCat(param_name, ".", struct_type.GetMemberName(i));
      XLS_ASSIGN_OR_RETURN(
          std::vector<InterpValue> field_domain,
          EnumerateValueDomain(field_ctx, struct_type.GetMemberType(i),
                               elements->at(i), max_points));
      XLS_RETURN_IF_ERROR(ExtendProduct(param_name, field_ctx, max_points,
                                        product, field_domain));
    }
    return WrapProductAsTuples(std::move(product));
  }

  if (target_type.IsTuple()) {
    if (!val.IsTuple()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain evaluates to type `%s`, but parameter `%s` expects a tuple",
          TagToString(val.tag()), param_name));
    }
    const TupleType& tuple_type = target_type.AsTuple();
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    if (elements->size() != tuple_type.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain tuple size (%d) does not match parameter `%s` tuple size "
          "(%d)",
          elements->size(), param_name, tuple_type.size()));
    }
    std::vector<std::vector<InterpValue>> product = {{}};
    for (size_t i = 0; i < elements->size(); ++i) {
      std::string elem_ctx = absl::StrFormat("%s.%d", param_name, i);
      XLS_ASSIGN_OR_RETURN(
          std::vector<InterpValue> elem_domain,
          EnumerateValueDomain(elem_ctx, tuple_type.GetMemberType(i),
                               elements->at(i), max_points));
      XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                        product, elem_domain));
    }
    return WrapProductAsTuples(std::move(product));
  }

  if (IsCompositeArray(target_type)) {
    if ((!val.IsTuple() && !val.IsArray()) || val.is_range()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected array or tuple of domains for array parameter `%s`; got "
          "`%s`",
          param_name, val.ToString()));
    }
    const ArrayType& array_type = target_type.AsArray();
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type.size().GetAsInt64());
    if (static_cast<int64_t>(elements->size()) > array_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Domain array size (%d) exceeds parameter `%s` array size (%d)",
          elements->size(), param_name, array_size));
    }
    XLS_RETURN_IF_ERROR(ValidateArraySize(param_name, array_size, max_points));
    std::vector<std::vector<InterpValue>> product = {{}};
    for (int64_t i = 0; i < array_size; ++i) {
      std::string elem_ctx = absl::StrFormat("%s[%d]", param_name, i);
      std::vector<InterpValue> elem_domain;
      if (i < static_cast<int64_t>(elements->size())) {
        XLS_ASSIGN_OR_RETURN(
            elem_domain,
            EnumerateValueDomain(elem_ctx, array_type.element_type(),
                                 elements->at(i), max_points));
      } else {
        XLS_ASSIGN_OR_RETURN(
            elem_domain,
            EnumerateArbitraryDomain(elem_ctx, array_type.element_type(),
                                     max_points, /*allow_composite=*/true));
      }
      XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                        product, elem_domain));
    }
    return WrapProductAsArrays(std::move(product));
  }

  if (std::optional<std::shared_ptr<RangeData>> range_data = val.GetRangeData();
      range_data.has_value()) {
    const RangeData& range = **range_data;
    return EnumerateRangeBounds(param_name, target_type, range.start, range.end,
                                range.inclusive, max_points);
  }

  if (val.IsArray()) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         val.GetValues());
    if (elements->empty()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Parameter `%s` has an empty domain", param_name));
    }
    if (static_cast<int64_t>(elements->size()) > max_points) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input domain for `%s` exceeds maximum size of %d "
                          "points",
                          param_name, max_points));
    }
    for (const InterpValue& element : *elements) {
      XLS_RETURN_IF_ERROR(
          ValidateScalarDomainElement(param_name, target_type, element));
    }
    return *elements;
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported domain value `%s` for parameter `%s`",
                      val.ToString(), param_name));
}

}  // namespace

absl::Status ValidateDomain(const Expr* domain, const Type* param_type,
                            std::string_view param_str, const TypeInfo& ti,
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
        absl::Substitute("Domain `$0` implies a struct type, but parameter "
                         "`$1` is of type $2.",
                         domain->ToString(), param_str, param_type->ToString()),
        file_table);
  }
  if (param_type->IsArray() && !domain_type->IsTuple()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::Substitute("Domain for array parameter `$0` must be a tuple; got "
                         "type $1",
                         param_str, domain_type->ToString()),
        file_table);
  }
  if (domain_type->IsTuple()) {
    const TupleType& tuple_type = domain_type->AsTuple();
    if (tuple_type.empty()) {
      // Empty tuple domain is allowed for any parameter type.
      return absl::OkStatus();
    }
    if (!param_type->IsTuple() && !param_type->IsArray()) {
      return TypeInferenceErrorStatus(
          domain->span(), param_type,
          absl::Substitute("Domain `$0` implies a tuple or array type, but "
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
        absl::Substitute("Unsupported domain `$0` of type `$1`.",
                         domain->ToString(), domain_type->ToString()),
        file_table);
  }

  absl::StatusOr<InterpValue> domain_value_status =
      Evaluate(domain, ti, warning_collector, import_data);
  if (!domain_value_status.ok()) {
    return TypeInferenceErrorStatus(
        domain->span(), param_type,
        absl::StrFormat("Domain must be a constexpr expression; evaluation "
                        "failed: %s",
                        domain_value_status.status().message()),
        file_table);
  }
  const InterpValue& domain_value = *domain_value_status;

  return ValidateDomainValue(domain_value, param_type, domain->span(),
                             param_str, file_table);
}

absl::Status ValidateFunctionDomains(const Function& fn,
                                     const XlsTuple* domains,
                                     const TypeInfo& ti,
                                     WarningCollector& warning_collector,
                                     const ImportData& import_data,
                                     const FileTable& file_table) {
  XLS_RET_CHECK(domains != nullptr);
  // Compare domains to params as appropriate, e.g., a u32 parameter should be
  // fuzzed/tabulated using a u32 range domain or array domain.
  int64_t domain_count = domains->members().size();
  XLS_RET_CHECK_EQ(domain_count, fn.params().size());
  for (int64_t i = 0; i < domain_count; ++i) {
    const Expr* domain = domains->members()[i];
    const Param* param = fn.params()[i];
    std::optional<Type*> param_type = ti.GetItem(param);
    XLS_RET_CHECK(param_type.has_value());
    XLS_RETURN_IF_ERROR(ValidateDomain(domain, *param_type, param->ToString(),
                                       ti, warning_collector, import_data,
                                       file_table));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<InterpValue>> EnumerateArbitraryDomain(
    std::string_view param_name, const Type& type, int64_t max_points,
    bool allow_composite) {
  XLS_RETURN_IF_ERROR(ValidateMaxPoints(max_points));
  if (allow_composite) {
    if (type.IsStruct()) {
      const StructType& struct_type = type.AsStruct();
      std::vector<std::vector<InterpValue>> product = {{}};
      for (size_t i = 0; i < struct_type.size(); ++i) {
        std::string field_ctx =
            absl::StrCat(param_name, ".", struct_type.GetMemberName(i));
        XLS_ASSIGN_OR_RETURN(
            std::vector<InterpValue> field_domain,
            EnumerateArbitraryDomain(field_ctx, struct_type.GetMemberType(i),
                                     max_points, /*allow_composite=*/true));
        XLS_RETURN_IF_ERROR(ExtendProduct(param_name, field_ctx, max_points,
                                          product, field_domain));
      }
      return WrapProductAsTuples(std::move(product));
    }
    if (type.IsTuple()) {
      const TupleType& tuple_type = type.AsTuple();
      std::vector<std::vector<InterpValue>> product = {{}};
      for (size_t i = 0; i < tuple_type.size(); ++i) {
        std::string elem_ctx = absl::StrFormat("%s.%d", param_name, i);
        XLS_ASSIGN_OR_RETURN(
            std::vector<InterpValue> elem_domain,
            EnumerateArbitraryDomain(elem_ctx, tuple_type.GetMemberType(i),
                                     max_points, /*allow_composite=*/true));
        XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                          product, elem_domain));
      }
      return WrapProductAsTuples(std::move(product));
    }
    if (IsCompositeArray(type)) {
      const ArrayType& array_type = type.AsArray();
      XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type.size().GetAsInt64());
      XLS_RETURN_IF_ERROR(
          ValidateArraySize(param_name, array_size, max_points));
      std::vector<std::vector<InterpValue>> product = {{}};
      for (int64_t i = 0; i < array_size; ++i) {
        std::string elem_ctx = absl::StrFormat("%s[%d]", param_name, i);
        XLS_ASSIGN_OR_RETURN(
            std::vector<InterpValue> elem_domain,
            EnumerateArbitraryDomain(elem_ctx, array_type.element_type(),
                                     max_points, /*allow_composite=*/true));
        XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                          product, elem_domain));
      }
      return WrapProductAsArrays(std::move(product));
    }
  }

  if (type.IsEnum()) {
    const std::vector<InterpValue>& members = type.AsEnum().members();
    if (members.empty()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Parameter `%s` has an empty domain", param_name));
    }
    if (static_cast<int64_t>(members.size()) > max_points) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Input domain for `%s` exceeds maximum size of %d "
                          "points",
                          param_name, max_points));
    }
    return members;
  }
  std::optional<BitsLikeProperties> bits_props = GetBitsLike(type);
  if (!bits_props.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Parameter `%s` has unsupported type `%s`; expected an enum, bool, or "
        "fixed-width integer type",
        param_name, type.ToString()));
  }
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_props->is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_props->size.GetAsInt64());
  int64_t max_bit_width = std::bit_width(static_cast<uint64_t>(max_points)) - 1;
  if (bit_count > max_bit_width) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Parameter `%s` has width %d bits, which exceeds the maximum single-"
        "parameter width (%d bits) for exhaustive tabulation",
        param_name, bit_count, max_bit_width));
  }
  if (IsBool(type)) {
    return std::vector<InterpValue>{InterpValue::MakeBool(false),
                                    InterpValue::MakeBool(true)};
  }
  if (bit_count == 0) {
    return std::vector<InterpValue>{
        InterpValue::MakeZeroValue(is_signed, /*bit_count=*/0)};
  }
  int64_t count = int64_t{1} << bit_count;
  std::vector<InterpValue> domain;
  domain.reserve(count);
  if (!is_signed) {
    for (int64_t v = 0; v < count; ++v) {
      domain.push_back(InterpValue::MakeUBits(bit_count, v));
    }
  } else {
    int64_t min_val = -(int64_t{1} << (bit_count - 1));
    int64_t max_val = (int64_t{1} << (bit_count - 1)) - 1;
    for (int64_t v = min_val; v <= max_val; ++v) {
      domain.push_back(InterpValue::MakeSBits(bit_count, v));
    }
  }
  return domain;
}

absl::StatusOr<std::vector<InterpValue>> EnumerateDomainForOneParam(
    ImportData* import_data, TypeInfo* type_info, std::string_view param_name,
    const Type& target_type, const Expr* expr, int64_t max_points) {
  XLS_RET_CHECK(import_data != nullptr);
  XLS_RET_CHECK(type_info != nullptr);
  XLS_RET_CHECK(expr != nullptr);
  XLS_RETURN_IF_ERROR(ValidateMaxPoints(max_points));

  if (expr->kind() == AstNodeKind::kStructInstance) {
    if (!target_type.IsStruct()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Struct domain expression `%s` cannot be used for non-struct "
          "parameter `%s` of type `%s`",
          expr->ToString(), param_name, target_type.ToString()));
    }
    const auto* struct_inst = absl::down_cast<const StructInstance*>(expr);
    const StructType& struct_type = target_type.AsStruct();
    for (const auto& [member_name, _] : struct_inst->GetUnorderedMembers()) {
      if (!struct_type.HasNamedMember(member_name)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Struct domain has unknown field `%s` for parameter `%s` of type "
            "`%s`",
            member_name, param_name, target_type.ToString()));
      }
    }
    std::vector<std::vector<InterpValue>> product = {{}};
    for (size_t i = 0; i < struct_type.size(); ++i) {
      std::string_view field_name = struct_type.GetMemberName(i);
      const Type& field_type = struct_type.GetMemberType(i);
      std::string field_ctx = absl::StrCat(param_name, ".", field_name);
      std::vector<InterpValue> field_domain;
      absl::StatusOr<Expr*> specified = struct_inst->GetExpr(field_name);
      if (specified.ok()) {
        XLS_ASSIGN_OR_RETURN(
            field_domain,
            EnumerateDomainForOneParam(import_data, type_info, field_ctx,
                                       field_type, *specified, max_points));
      } else if (absl::IsNotFound(specified.status())) {
        XLS_ASSIGN_OR_RETURN(
            field_domain,
            EnumerateArbitraryDomain(field_ctx, field_type, max_points,
                                     /*allow_composite=*/true));
      } else {
        return specified.status();
      }
      XLS_RETURN_IF_ERROR(ExtendProduct(param_name, field_ctx, max_points,
                                        product, field_domain));
    }
    return WrapProductAsTuples(std::move(product));
  }

  if (expr->kind() == AstNodeKind::kXlsTuple) {
    const auto* tuple_expr = absl::down_cast<const XlsTuple*>(expr);
    if (tuple_expr->members().empty()) {
      return EnumerateArbitraryDomain(param_name, target_type, max_points,
                                      /*allow_composite=*/true);
    }
    if (target_type.IsTuple()) {
      const TupleType& tuple_type = target_type.AsTuple();
      if (tuple_expr->members().size() != tuple_type.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Domain tuple size (%d) does not match parameter `%s` tuple size "
            "(%d)",
            tuple_expr->members().size(), param_name, tuple_type.size()));
      }
      std::vector<std::vector<InterpValue>> product = {{}};
      for (size_t i = 0; i < tuple_expr->members().size(); ++i) {
        std::string elem_ctx = absl::StrFormat("%s.%d", param_name, i);
        XLS_ASSIGN_OR_RETURN(
            std::vector<InterpValue> elem_domain,
            EnumerateDomainForOneParam(import_data, type_info, elem_ctx,
                                       tuple_type.GetMemberType(i),
                                       tuple_expr->members()[i], max_points));
        XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                          product, elem_domain));
      }
      return WrapProductAsTuples(std::move(product));
    }
    if (IsCompositeArray(target_type)) {
      const ArrayType& array_type = target_type.AsArray();
      XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type.size().GetAsInt64());
      if (static_cast<int64_t>(tuple_expr->members().size()) > array_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Domain array size (%d) exceeds parameter `%s` array size (%d)",
            tuple_expr->members().size(), param_name, array_size));
      }
      XLS_RETURN_IF_ERROR(
          ValidateArraySize(param_name, array_size, max_points));
      std::vector<std::vector<InterpValue>> product = {{}};
      for (int64_t i = 0; i < array_size; ++i) {
        std::string elem_ctx = absl::StrFormat("%s[%d]", param_name, i);
        std::vector<InterpValue> elem_domain;
        if (i < static_cast<int64_t>(tuple_expr->members().size())) {
          XLS_ASSIGN_OR_RETURN(
              elem_domain,
              EnumerateDomainForOneParam(import_data, type_info, elem_ctx,
                                         array_type.element_type(),
                                         tuple_expr->members()[i], max_points));
        } else {
          XLS_ASSIGN_OR_RETURN(
              elem_domain,
              EnumerateArbitraryDomain(elem_ctx, array_type.element_type(),
                                       max_points, /*allow_composite=*/true));
        }
        XLS_RETURN_IF_ERROR(ExtendProduct(param_name, elem_ctx, max_points,
                                          product, elem_domain));
      }
      return WrapProductAsArrays(std::move(product));
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tuple domain expression `%s` cannot be used for parameter `%s` of "
        "type `%s`",
        expr->ToString(), param_name, target_type.ToString()));
  }

  if (expr->kind() == AstNodeKind::kRange) {
    const auto* range_node = absl::down_cast<const Range*>(expr);
    XLS_ASSIGN_OR_RETURN(InterpValue start_val,
                         ConstexprEvaluator::EvaluateToValue(
                             import_data, type_info,
                             /*warning_collector=*/nullptr, ParametricEnv(),
                             range_node->start()));
    XLS_ASSIGN_OR_RETURN(
        InterpValue end_val,
        ConstexprEvaluator::EvaluateToValue(
            import_data, type_info,
            /*warning_collector=*/nullptr, ParametricEnv(), range_node->end()));
    return EnumerateRangeBounds(param_name, target_type, start_val, end_val,
                                range_node->inclusive_end(), max_points);
  }

  if (std::optional<Type*> expr_type = type_info->GetItem(expr);
      expr_type.has_value()) {
    if ((*expr_type)->IsTuple() && !(*expr_type)->AsTuple().empty() &&
        target_type.IsStruct()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Tuple domain expression `%s` cannot be used for struct parameter "
          "`%s` of type `%s`",
          expr->ToString(), param_name, target_type.ToString()));
    }
    if ((*expr_type)->IsStruct() && !target_type.IsStruct()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Struct domain expression `%s` cannot be used for non-struct "
          "parameter `%s` of type `%s`",
          expr->ToString(), param_name, target_type.ToString()));
    }
  }

  XLS_ASSIGN_OR_RETURN(
      InterpValue const_value,
      ConstexprEvaluator::EvaluateToValue(import_data, type_info,
                                          /*warning_collector=*/nullptr,
                                          ParametricEnv(), expr));
  return EnumerateValueDomain(param_name, target_type, const_value, max_points);
}

absl::StatusOr<std::vector<std::vector<InterpValue>>> EnumerateFunctionDomains(
    ImportData* import_data, TypeInfo* type_info, const Function& fn,
    const FunctionType& fn_type, std::optional<const XlsTuple*> domains_tuple,
    int64_t max_points) {
  XLS_RET_CHECK(import_data != nullptr);
  XLS_RET_CHECK(type_info != nullptr);
  XLS_RET_CHECK_EQ(fn.params().size(), fn_type.params().size());
  XLS_RETURN_IF_ERROR(ValidateMaxPoints(max_points));

  if (domains_tuple.has_value() &&
      (*domains_tuple)->members().size() != fn.params().size()) {
    int64_t domain_count = (*domains_tuple)->members().size();
    return absl::InvalidArgumentError(absl::StrFormat(
        "Domain tuple has %d argument%s, but function `%s` has %d parameter%s",
        domain_count, domain_count == 1 ? "" : "s", fn.identifier(),
        fn.params().size(), fn.params().size() == 1 ? "" : "s"));
  }

  std::vector<std::vector<InterpValue>> rows = {{}};
  for (size_t i = 0; i < fn.params().size(); ++i) {
    const Param* param = fn.params()[i];
    const Type& param_type = *fn_type.params()[i];
    std::vector<InterpValue> domain;
    if (domains_tuple.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          domain, EnumerateDomainForOneParam(
                      import_data, type_info, param->identifier(), param_type,
                      (*domains_tuple)->members()[i], max_points));
    } else {
      XLS_ASSIGN_OR_RETURN(
          domain,
          EnumerateArbitraryDomain(param->identifier(), param_type, max_points,
                                   /*allow_composite=*/false));
    }
    XLS_RETURN_IF_ERROR(ExtendProduct(fn.identifier(), param->identifier(),
                                      max_points, rows, domain));
  }
  return rows;
}

}  // namespace xls::dslx
