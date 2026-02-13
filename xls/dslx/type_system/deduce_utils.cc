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

#include "xls/dslx/type_system/deduce_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

// A validator for the type of an argument being formatted by a macro like
// `trace_fmt!` or `vtrace_fmt!`.
class FormatMacroArgumentValidator : public TypeVisitor {
 public:
  FormatMacroArgumentValidator(const FileTable& file_table, const Span& span)
      : file_table_(file_table), span_(span) {}

  absl::Status HandleArray(const ArrayType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleBits(const BitsType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleEnum(const EnumType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleToken(const TokenType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleStruct(const StructType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleProc(const ProcType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleTuple(const TupleType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleBitsConstructor(const BitsConstructorType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleFunction(const FunctionType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with function type",
        file_table_);
  }
  absl::Status HandleChannel(const ChannelType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with channel type",
        file_table_);
  }
  absl::Status HandleMeta(const MetaType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with meta type", file_table_);
  }
  absl::Status HandleModule(const ModuleType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with module type",
        file_table_);
  }

 private:
  const FileTable& file_table_;
  const Span& span_;
};

}  // namespace

absl::Status TryEnsureFitsInType(const Number& number,
                                 const BitsLikeProperties& bits_like,
                                 const Type& type) {
  const FileTable& file_table = *number.owner()->file_table();
  VLOG(5) << "TryEnsureFitsInType; number: " << number.ToString() << " @ "
          << number.span().ToString(file_table);

  bool is_signed = bits_like.is_signed.GetAsBool().value();

  // Characters have a `u8` type. They can support the dash (negation symbol).
  if (number.number_kind() != NumberKind::kCharacter &&
      number.text()[0] == '-' && !is_signed) {
    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Number %s invalid: "
                        "can't assign a negative value to an unsigned type.",
                        number.ToString()),
        file_table);
  }

  XLS_ASSIGN_OR_RETURN(const int64_t bit_count, bits_like.size.GetAsInt64());

  // Helper to give an informative error on the appropriate range when we
  // determine the numerical value given doesn't fit into the type.
  auto does_not_fit = [&]() -> absl::Status {
    std::string low;
    std::string high;

    if (is_signed) {
      low = BitsToString(Bits::MinSigned(bit_count),
                         FormatPreference::kSignedDecimal);
      high = BitsToString(Bits::MaxSigned(bit_count),
                          FormatPreference::kSignedDecimal);
    } else {
      low = BitsToString(Bits(bit_count), FormatPreference::kUnsignedDecimal);
      high = BitsToString(Bits::AllOnes(bit_count),
                          FormatPreference::kUnsignedDecimal);
    }

    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Value '%s' does not fit in "
                        "the bitwidth of a %s (%d). "
                        "Valid values are [%s, %s].",
                        number.text(), type.ToString(), bit_count, low, high),
        file_table);
  };

  XLS_ASSIGN_OR_RETURN(bool fits_in_type,
                       number.FitsInType(bit_count, is_signed));
  if (!fits_in_type) {
    return does_not_fit();
  }
  return absl::OkStatus();
}

absl::Status ValidateArrayTypeForIndex(const Index& node, const Type& type,
                                       const FileTable& file_table) {
  if (const auto* tuple_type = dynamic_cast<const TupleType*>(&type)) {
    return TypeInferenceErrorStatus(
        node.span(), tuple_type,
        "Tuples should not be indexed with array-style syntax. "
        "Use `tuple.<number>` syntax instead.",
        file_table);
  }

  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  if (bits_like.has_value()) {
    return TypeInferenceErrorStatus(
        node.span(), &type,
        "Bits-like value cannot be indexed, value to index is not an array.",
        file_table);
  }

  const auto* array_type = dynamic_cast<const ArrayType*>(&type);
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node.span(), &type, "Value to index is not an array.", file_table);
  }

  return absl::OkStatus();
}

absl::Status ValidateTupleTypeForIndex(const TupleIndex& node, const Type& type,
                                       const FileTable& file_table) {
  if (dynamic_cast<const TupleType*>(&type) != nullptr) {
    return absl::OkStatus();
  }
  return TypeInferenceErrorStatus(
      node.span(), &type,
      absl::StrCat("Attempted to use tuple indexing on a non-tuple: ",
                   node.ToString()),
      file_table);
}

absl::Status ValidateArrayIndex(const Index& node, const Type& array_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table) {
  std::optional<BitsLikeProperties> index_bits_like = GetBitsLike(index_type);
  if (!index_bits_like.has_value()) {
    return TypeInferenceErrorStatus(node.span(), &index_type,
                                    "Index is not bits typed.", file_table);
  }
  XLS_ASSIGN_OR_RETURN(bool is_signed, index_bits_like->is_signed.GetAsBool());
  if (is_signed) {
    return TypeInferenceErrorStatus(node.span(), &index_type,
                                    "Index is not unsigned-bits typed.",
                                    file_table);
  }

  const Expr* rhs = std::get<Expr*>(node.rhs());
  VLOG(10) << absl::StreamFormat("Index RHS: `%s` constexpr? %d",
                                 rhs->ToString(), ti.IsKnownConstExpr(rhs));

  // If we know the array size concretely and the index is a constexpr
  // expression, we can check it is in bounds. Note that in v2, the size will
  // never be parametric here, because v2 always resolves parametrics before
  // producing a `Type`.
  const auto& casted_array_type = dynamic_cast<const ArrayType&>(array_type);

  // Reject indexing into zero-sized arrays regardless of whether the index is
  // constexpr, since out-of-bounds semantics (return last element) are
  // undefined for empty arrays.

  XLS_ASSIGN_OR_RETURN(int64_t concrete_size,
                       casted_array_type.size().GetAsInt64());
  if (concrete_size == 0) {
    return TypeInferenceErrorStatus(node.span(), &array_type,
                                    "Zero-sized arrays cannot be indexed",
                                    file_table);
  }

  if (!ti.IsKnownConstExpr(rhs)) {
    return absl::OkStatus();
  }

  VLOG(10) << "Array type: " << casted_array_type.ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value, ti.GetConstExpr(rhs));
  VLOG(10) << "Index RHS is known constexpr value: " << constexpr_value;
  XLS_ASSIGN_OR_RETURN(uint64_t constexpr_index,
                       constexpr_value.GetBitValueUnsigned());
  XLS_ASSIGN_OR_RETURN(int64_t array_size,
                       casted_array_type.size().GetAsInt64());
  if (constexpr_index >= array_size) {
    return TypeInferenceErrorStatus(
        node.span(), &array_type,
        absl::StrFormat("Index has a compile-time constant value %d that is "
                        "out of bounds of the array type.",
                        constexpr_index),
        file_table);
  }
  return absl::OkStatus();
}

absl::Status ValidateTupleIndex(const TupleIndex& node, const Type& tuple_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table) {
  // TupleIndex RHSs are always constexpr numbers.
  const auto& casted_tuple_type = dynamic_cast<const TupleType&>(tuple_type);
  XLS_ASSIGN_OR_RETURN(InterpValue index_value, ti.GetConstExpr(node.index()));
  XLS_ASSIGN_OR_RETURN(int64_t index, index_value.GetBitValueViaSign());
  if (index >= casted_tuple_type.size()) {
    return TypeInferenceErrorStatus(
        node.span(), &tuple_type,
        absl::StrCat("Out-of-bounds tuple index specified: ",
                     node.index()->ToString()),
        file_table);
  }
  return absl::OkStatus();
}

bool IsNameRefTo(const Expr* e, const NameDef* name_def) {
  if (auto* name_ref = dynamic_cast<const NameRef*>(e)) {
    const AnyNameDef any_name_def = name_ref->name_def();
    return std::holds_alternative<const NameDef*>(any_name_def) &&
           std::get<const NameDef*>(any_name_def) == name_def;
  }
  return false;
}

absl::Status ValidateNumber(const Number& number, const Type& type) {
  VLOG(5) << "Validating " << number.ToString() << " vs " << type;

  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      bits_like.has_value()) {
    return TryEnsureFitsInType(number, bits_like.value(), type);
  }

  const FileTable& file_table = *number.owner()->file_table();
  return TypeInferenceErrorStatus(
      number.span(), &type,
      absl::StrFormat("Non-bits type (%s) used to define a numeric literal.",
                      type.GetDebugTypeName()),
      file_table);
}

absl::Status ValidateFormatMacroArgument(const Type& type, const Span& span,
                                         const FileTable& file_table) {
  FormatMacroArgumentValidator validator(file_table, span);
  return type.Accept(validator);
}

absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetMemberOrError<Proc>(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  std::optional<ImportSubject> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported_info,
                       type_info->GetImportedOrError(import.value()));
  return imported_info->module->GetMemberOrError<Proc>(colon_ref->attr());
}

absl::StatusOr<StartAndWidth> ResolveBitSliceIndices(
    int64_t bit_count, std::optional<int64_t> start_opt,
    std::optional<int64_t> limit_opt) {
  XLS_RET_CHECK_GE(bit_count, 0);
  int64_t start = 0;
  int64_t limit = bit_count;

  if (start_opt.has_value()) {
    start = *start_opt;
  }
  if (limit_opt.has_value()) {
    limit = *limit_opt;
  }

  if (start < 0) {
    start += bit_count;
  }
  if (limit < 0) {
    limit += bit_count;
  }

  limit = std::clamp(limit, int64_t{0}, bit_count);
  start = std::clamp(start, int64_t{0}, limit);
  return StartAndWidth{.start = start, .width = limit - start};
}

static int64_t Size(TupleTypeOrAnnotation t) {
  if (std::holds_alternative<const TupleType*>(t)) {
    return std::get<const TupleType*>(t)->size();
  }
  return std::get<const TupleTypeAnnotation*>(t)->size();
}

absl::Status TypeOrAnnotationErrorStatus(Span span, TupleTypeOrAnnotation type,
                                         std::string_view message,
                                         const FileTable& file_table) {
  if (std::holds_alternative<const TupleType*>(type)) {
    return TypeInferenceErrorStatus(span, std::get<const TupleType*>(type),
                                    message, file_table);
  }
  return TypeInferenceErrorStatusForAnnotation(
      span, std::get<const TupleTypeAnnotation*>(type), message, file_table);
}

absl::StatusOr<std::pair<int64_t, int64_t>> GetTupleSizes(
    const NameDefTree* name_def_tree, TupleTypeOrAnnotation tuple_type) {
  const FileTable& file_table = *name_def_tree->owner()->file_table();
  bool rest_of_tuple_found = false;
  for (const NameDefTree* node : name_def_tree->nodes()) {
    if (node->IsRestOfTupleLeaf()) {
      if (rest_of_tuple_found) {
        return TypeOrAnnotationErrorStatus(
            node->span(), tuple_type,
            absl::StrFormat("`..` can only be used once per tuple pattern."),
            file_table);
      }
      rest_of_tuple_found = true;
    }
  }
  int64_t number_of_tuple_elements = Size(tuple_type);
  int64_t number_of_names = name_def_tree->nodes().size();
  bool number_mismatch = number_of_names != number_of_tuple_elements;
  if (rest_of_tuple_found) {
    // There's a "rest of tuple" in the name def tree; we only need to have
    // enough tuple elements to bind to the required names.
    // Subtract 1 for the  ".."
    number_of_names--;
    number_mismatch = number_of_names > number_of_tuple_elements;
  }
  if (number_mismatch) {
    return TypeOrAnnotationErrorStatus(
        name_def_tree->span(), tuple_type,
        absl::StrFormat("Cannot match a %d-element tuple to %d values.",
                        number_of_tuple_elements, number_of_names),
        file_table);
  }
  return std::make_pair(number_of_tuple_elements, number_of_names);
}

static absl::StatusOr<TupleTypeOrAnnotation> GetTupleType(
    TypeOrAnnotation type, Span span, const FileTable& file_table) {
  std::string error_msg = "Expected a tuple type for these names, but got";
  if (std::holds_alternative<const Type*>(type)) {
    const Type* t = std::get<const Type*>(type);
    if (auto* tuple_type = dynamic_cast<const TupleType*>(t)) {
      return tuple_type;
    }
    return TypeInferenceErrorStatus(
        span, t, absl::StrFormat("%s %s", error_msg, t->ToString()),
        file_table);
  }
  const TypeAnnotation* type_annot = std::get<const TypeAnnotation*>(type);
  if (auto* tuple_type_annotation =
          dynamic_cast<const TupleTypeAnnotation*>(type_annot)) {
    return tuple_type_annotation;
  }
  return TypeInferenceErrorStatusForAnnotation(
      span, type_annot,
      absl::StrFormat("%s %s", error_msg, type_annot->ToString()), file_table);
}

static TypeOrAnnotation GetSubType(TupleTypeOrAnnotation type, int64_t index) {
  if (std::holds_alternative<const TupleType*>(type)) {
    auto* tuple_type = std::get<const TupleType*>(type);
    return &(tuple_type->GetMemberType(index));
  }
  return std::get<const TupleTypeAnnotation*>(type)->members()[index];
}

absl::Status MatchTupleNodeToType(
    std::function<absl::Status(AstNode*, TypeOrAnnotation,
                               std::optional<InterpValue>)>
        process_tuple_member,
    const NameDefTree* name_def_tree, const TypeOrAnnotation type,
    const FileTable& file_table, std::optional<InterpValue> constexpr_value) {
  if (name_def_tree->is_leaf()) {
    AstNode* name_def = ToAstNode(name_def_tree->leaf());
    XLS_RETURN_IF_ERROR(process_tuple_member(name_def, type, constexpr_value));
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(TupleTypeOrAnnotation tuple_type,
                       GetTupleType(type, name_def_tree->span(), file_table));

  XLS_ASSIGN_OR_RETURN((auto [number_of_tuple_elements, number_of_names]),
                       GetTupleSizes(name_def_tree, tuple_type));
  // Index into the current tuple type.
  int64_t tuple_index = 0;
  // Must iterate through the actual nodes size, not number_of_names, because
  // there may be a "rest of tuple" leaf which decreases the number of names.
  for (int64_t name_index = 0; name_index < name_def_tree->nodes().size();
       ++name_index) {
    NameDefTree* subtree = name_def_tree->nodes()[name_index];
    if (subtree->IsRestOfTupleLeaf()) {
      // Skip ahead.
      tuple_index += number_of_tuple_elements - number_of_names;
      continue;
    }
    TypeOrAnnotation subtype = GetSubType(tuple_type, tuple_index);
    XLS_RETURN_IF_ERROR(process_tuple_member(subtree, subtype, std::nullopt));

    std::optional<InterpValue> sub_value;
    if (constexpr_value.has_value()) {
      sub_value = constexpr_value.value().GetValuesOrDie()[tuple_index];
    }
    XLS_RETURN_IF_ERROR(MatchTupleNodeToType(process_tuple_member, subtree,
                                             subtype, file_table, sub_value));

    ++tuple_index;
  }
  return absl::OkStatus();
}

bool IsAcceptableCast(const Type& from, const Type& to) {
  if (from == to) {
    return true;
  }
  auto is_enum = [](const Type& ct) -> bool {
    return dynamic_cast<const EnumType*>(&ct) != nullptr;
  };
  auto is_bits_array = [&](const Type& ct) -> bool {
    const ArrayType* at = dynamic_cast<const ArrayType*>(&ct);
    if (at == nullptr) {
      return false;
    }
    if (IsBitsLike(at->element_type())) {
      return true;
    }
    return false;
  };
  if ((is_bits_array(from) && IsBitsLike(to)) ||
      (IsBitsLike(from) && is_bits_array(to))) {
    TypeDim from_total_bit_count = from.GetTotalBitCount().value();
    TypeDim to_total_bit_count = to.GetTotalBitCount().value();
    return from_total_bit_count == to_total_bit_count;
  }
  if ((IsBitsLike(from) || is_enum(from)) && IsBitsLike(to)) {
    return true;
  }
  if (IsBitsLike(from) && is_enum(to)) {
    return true;
  }
  return false;
}

absl::Status NoteBuiltinInvocationConstExpr(std::string_view fn_name,
                                            const Invocation* invocation,
                                            const FunctionType& fn_type,
                                            TypeInfo* ti,
                                            ImportData* import_data) {
  // array_size is always a constexpr result since it just needs the type
  // information
  if (fn_name == "array_size") {
    auto* array_type =
        absl::down_cast<const ArrayType*>(fn_type.params()[0].get());
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    ti->NoteConstExpr(invocation,
                      InterpValue::MakeU32(static_cast<int32_t>(array_size)));
  }

  // bit_count and element_count are similar to array_size, but use the
  // parametric argument rather than a value.
  if (fn_name == "bit_count" || fn_name == "element_count") {
    XLS_RET_CHECK_EQ(invocation->explicit_parametrics().size(), 1);
    std::optional<const Type*> explicit_parametric_type =
        ti->GetItem(ToAstNode(invocation->explicit_parametrics()[0]));
    XLS_RET_CHECK(explicit_parametric_type.has_value());
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        fn_name == "element_count"
            ? GetElementCountAsInterpValue(*explicit_parametric_type)
            : GetBitCountAsInterpValue(*explicit_parametric_type));
    ti->NoteConstExpr(invocation, value);
  }

  if (fn_name == "configured_value_or") {
    XLS_RET_CHECK_EQ(invocation->args().size(), 2);
    std::string key =
        absl::down_cast<const String*>(invocation->args()[0])->text();
    if (!invocation->owner()->configured_values().contains(key)) {
      Expr* default_value_expr = invocation->args()[1];
      XLS_ASSIGN_OR_RETURN(InterpValue default_value,
                           ti->GetConstExpr(default_value_expr));
      ti->NoteConstExpr(invocation, default_value);
      return absl::OkStatus();
    }
    std::optional<std::string> override_str =
        invocation->owner()->configured_values().at(key);
    std::optional<const Type*> explicit_parametric_type = ti->GetItem(
        std::get<TypeAnnotation*>(invocation->explicit_parametrics()[0]));
    XLS_ASSIGN_OR_RETURN(
        InterpValue result_value,
        GetConfiguredValueAsInterpValue(override_str.value(),
                                        explicit_parametric_type.value(), ti,
                                        import_data, invocation->span()));
    ti->NoteConstExpr(invocation, result_value);
  }
  return absl::OkStatus();
}

const TypeInfo& GetTypeInfoForNodeIfDifferentModule(
    AstNode* node, const TypeInfo& current_type_info,
    const ImportData& import_data) {
  if (node->owner() == current_type_info.module()) {
    return current_type_info;
  }
  absl::StatusOr<const TypeInfo*> type_info =
      import_data.GetRootTypeInfoForNode(node);
  CHECK_OK(type_info.status())
      << "Must be able to get root type info for node " << node->ToString();
  CHECK(type_info.value() != nullptr);
  return *type_info.value();
}

void WarnOnInappropriateConstantName(std::string_view identifier,
                                     const Span& span, const Module& module,
                                     WarningCollector* warning_collector) {
  if (!IsScreamingSnakeCase(identifier) &&
      !module.attributes().contains(
          ModuleAttribute::kAllowNonstandardConstantNaming)) {
    warning_collector->Add(
        span, WarningKind::kConstantNaming,
        absl::StrFormat("Standard style is SCREAMING_SNAKE_CASE for constant "
                        "identifiers; got: `%s`",
                        identifier));
  }
}

absl::StatusOr<InterpValue> GetBitCountAsInterpValue(const Type* type) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  XLS_ASSIGN_OR_RETURN(TypeDim bit_count_ctd, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       bit_count_ctd.value().GetBitValueViaSign());
  return InterpValue::MakeU32(bit_count);
}

absl::StatusOr<InterpValue> GetElementCountAsInterpValue(const Type* type) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  if (const auto* array_type = dynamic_cast<const ArrayType*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64_t size, array_type->size().GetAsInt64());
    CHECK(static_cast<uint32_t>(size) == size);
    return InterpValue::MakeU32(size);
  }
  if (const auto* tuple_type = dynamic_cast<const TupleType*>(type)) {
    return InterpValue::MakeU32(tuple_type->members().size());
  }
  if (const auto* struct_type = dynamic_cast<const StructType*>(type)) {
    return InterpValue::MakeU32(struct_type->members().size());
  }
  return GetBitCountAsInterpValue(type);
}

absl::StatusOr<InterpValue> GetConfiguredValueAsInterpValue(
    std::string override_value, const Type* type, const TypeInfo* type_info,
    ImportData* import_data, const Span& span) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  if (IsBool(*type)) {
    return InterpValue::MakeBool(override_value == "true");
  }

  if (const auto* enum_type = dynamic_cast<const EnumType*>(type)) {
    XLS_RET_CHECK(type_info != nullptr);
    XLS_RET_CHECK(import_data != nullptr);
    const EnumDef* enum_def = &enum_type->nominal_type();
    const TypeInfo& enum_type_info = GetTypeInfoForNodeIfDifferentModule(
        const_cast<EnumDef*>(enum_def), *type_info, *import_data);
    std::vector<std::string_view> parts = absl::StrSplit(override_value, "::");
    std::string_view enum_identifier = parts.back();
    for (const EnumMember& member : enum_def->values()) {
      if (member.name_def->identifier() == enum_identifier) {
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             enum_type_info.GetConstExpr(member.value));
        return InterpValue::MakeEnum(value.GetBitsOrDie(), enum_type, enum_def);
      }
    }
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Invalid value \'%s\' for enum %s", override_value,
                        enum_type->ToString()),
        import_data->file_table());
  }

  if (const auto* bits_type = dynamic_cast<const BitsType*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type->size().GetAsInt64());
    bool is_signed = bits_type->is_signed();
    std::string_view value_str = override_value;
    if (auto pos = value_str.find(':'); pos != std::string_view::npos) {
      value_str = value_str.substr(pos + 1);
    }
    XLS_ASSIGN_OR_RETURN(Bits bits, ParseNumber(value_str));

    if (bits.bit_count() > bit_count) {
      return TypeInferenceErrorStatus(
          span, nullptr,
          absl::StrFormat("Parsed value \'%s\' (which is %d bits) is too large "
                          "for type %s (which is %d bits)",
                          override_value, bits.bit_count(),
                          bits_type->ToString(), bit_count),
          import_data->file_table());
    }

    if (bits.bit_count() < bit_count) {
      if (is_signed) {
        bits = bits_ops::SignExtend(bits, bit_count);
      } else {
        bits = bits_ops::ZeroExtend(bits, bit_count);
      }
    }
    return InterpValue::MakeBits(is_signed, bits);
  }

  return TypeInferenceErrorStatus(
      span, nullptr,
      absl::StrFormat("Unsupported configured value type: %s for: \'%s\'",
                      type->ToString(), override_value),
      import_data->file_table());
}

std::string PatternsToString(const MatchArm* arm) {
  return absl::StrJoin(arm->patterns(), " | ",
                       [](std::string* out, NameDefTree* ndt) {
                         absl::StrAppend(out, ndt->ToString());
                       });
}

absl::Status CheckArrayDimTooLarge(Span span, uint64_t dim,
                                   const FileTable& file_table) {
  if ((dim >> 31) == 0) {
    return absl::OkStatus();
  }
  return ArrayDimTooLargeErrorStatus(span, dim, file_table);
}

}  // namespace xls::dslx
