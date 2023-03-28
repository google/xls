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
#include "xls/dslx/interp_value_helpers.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"

namespace xls::dslx {
namespace {

absl::StatusOr<InterpValue> InterpValueFromString(std::string_view s) {
  XLS_ASSIGN_OR_RETURN(Value value, Parser::ParseTypedValue(s));
  return dslx::ValueToInterpValue(value);
}

}  // namespace

absl::StatusOr<InterpValue> CastBitsToArray(const InterpValue& bits_value,
                                            const ArrayType& array_type) {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim element_bit_count,
                       array_type.element_type().GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bits_per_element,
                       element_bit_count.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, bits_value.GetBits());

  auto bit_slice_value_at_index = [&](int64_t i) -> InterpValue {
    int64_t lo = i * bits_per_element;
    Bits rev = bits_ops::Reverse(bits);
    Bits slice = rev.Slice(lo, bits_per_element);
    Bits result = bits_ops::Reverse(slice);
    return InterpValue::MakeBits(InterpValueTag::kUBits, result).value();
  };

  std::vector<InterpValue> values;
  XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type.size().GetAsInt64());
  for (int64_t i = 0; i < array_size; ++i) {
    values.push_back(bit_slice_value_at_index(i));
  }

  return InterpValue::MakeArray(values);
}

absl::StatusOr<InterpValue> CastBitsToEnum(const InterpValue& bits_value,
                                           const EnumType& enum_type) {
  const EnumDef& enum_def = enum_type.nominal_type();
  bool found = false;
  for (const InterpValue& member_value : enum_type.members()) {
    if (bits_value.GetBitsOrDie() == member_value.GetBitsOrDie()) {
      found = true;
      break;
    }
  }

  if (!found) {
    return absl::InternalError(
        absl::StrFormat("FailureError: Value is not valid for enum %s: %s",
                        enum_def.identifier(), bits_value.ToString()));
  }
  return InterpValue::MakeEnum(bits_value.GetBitsOrDie(), bits_value.IsSigned(),
                               &enum_def);
}

absl::StatusOr<InterpValue> CreateZeroValueFromType(
    const ConcreteType& concrete_type) {
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type->size().GetAsInt64());

    if (bits_type->is_signed()) {
      return InterpValue::MakeSBits(bit_count, /*value=*/0);
    }

    return InterpValue::MakeUBits(bit_count, /*value=*/0);
  }

  if (auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type)) {
    const int64_t tuple_size = tuple_type->size();

    std::vector<InterpValue> zero_elements;
    zero_elements.reserve(tuple_size);

    for (int64_t i = 0; i < tuple_size; ++i) {
      XLS_ASSIGN_OR_RETURN(
          InterpValue zero_element,
          CreateZeroValueFromType(tuple_type->GetMemberType(i)));
      zero_elements.push_back(zero_element);
    }

    return InterpValue::MakeTuple(zero_elements);
  }

  if (auto* struct_type = dynamic_cast<const StructType*>(&concrete_type)) {
    const int64_t struct_size = struct_type->size();

    std::vector<InterpValue> zero_elements;
    zero_elements.reserve(struct_size);

    for (int64_t i = 0; i < struct_size; ++i) {
      XLS_ASSIGN_OR_RETURN(
          InterpValue zero_element,
          CreateZeroValueFromType(struct_type->GetMemberType(i)));
      zero_elements.push_back(zero_element);
    }

    return InterpValue::MakeTuple(zero_elements);
  }

  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(const int64_t array_size,
                         array_type->size().GetAsInt64());

    if (array_size == 0) {
      return InterpValue::MakeArray({});
    }

    XLS_ASSIGN_OR_RETURN(InterpValue zero_element,
                         CreateZeroValueFromType(array_type->element_type()));
    std::vector<InterpValue> zero_elements(array_size, zero_element);
    return InterpValue::MakeArray(zero_elements);
  }

  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    if (!enum_type->members().empty()) {
      return enum_type->members().at(0);
    }
  }

  return absl::UnimplementedError("Cannot create zero value for type type: " +
                                  concrete_type.ToString());
}

absl::StatusOr<InterpValue> CreateZeroValue(const InterpValue& value) {
  switch (value.tag()) {
    case InterpValueTag::kSBits: {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
      return InterpValue::MakeSBits(bit_count, /*value=*/0);
    }
    case InterpValueTag::kUBits: {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
      return InterpValue::MakeUBits(bit_count, /*value=*/0);
    }
    case InterpValueTag::kTuple: {
      XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                           value.GetValues());
      std::vector<InterpValue> zero_elements;
      zero_elements.reserve(elements->size());
      for (const auto& element : *elements) {
        XLS_ASSIGN_OR_RETURN(InterpValue zero_element,
                             CreateZeroValue(element));
        zero_elements.push_back(zero_element);
      }
      return InterpValue::MakeTuple(zero_elements);
    }
    case InterpValueTag::kArray: {
      XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                           value.GetValues());
      if (elements->empty()) {
        return InterpValue::MakeArray({});
      }
      XLS_ASSIGN_OR_RETURN(InterpValue zero_element,
                           CreateZeroValue(elements->at(0)));
      std::vector<InterpValue> zero_elements(elements->size(), zero_element);
      return InterpValue::MakeArray(zero_elements);
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid InterpValueTag for zero-value generation: ",
                       TagToString(value.tag())));
  }
}

absl::Status FlattenTuple(const InterpValue& value,
                          std::vector<InterpValue>* result) {
  if (!value.IsTuple()) {
    result->push_back(value);
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       value.GetValues());
  for (const auto& value : *values) {
    XLS_RETURN_IF_ERROR(FlattenTuple(value, result));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::optional<int64_t>> FindFirstDifferingIndex(
    absl::Span<const InterpValue> lhs, absl::Span<const InterpValue> rhs) {
  if (lhs.size() != rhs.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("LHS and RHS must have the same size: %d vs. %d.",
                        lhs.size(), rhs.size()));
  }

  for (int64_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].Ne(rhs[i])) {
      return i;
    }
  }

  return std::nullopt;
}

absl::StatusOr<InterpValue> SignConvertValue(const ConcreteType& concrete_type,
                                             const InterpValue& value) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsTuple()) << value.ToString();
    const int64_t tuple_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64_t i = 0; i < tuple_size; ++i) {
      const InterpValue& e = value.GetValuesOrDie()[i];
      const ConcreteType& t = tuple_type->GetMemberType(i);
      XLS_ASSIGN_OR_RETURN(InterpValue converted, SignConvertValue(t, e));
      results.push_back(converted);
    }
    return InterpValue::MakeTuple(std::move(results));
  }
  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsArray()) << value.ToString();
    const ConcreteType& t = array_type->element_type();
    int64_t array_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64_t i = 0; i < array_size; ++i) {
      const InterpValue& e = value.GetValuesOrDie()[i];
      XLS_ASSIGN_OR_RETURN(InterpValue converted, SignConvertValue(t, e));
      results.push_back(converted);
    }
    return InterpValue::MakeArray(std::move(results));
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsBits()) << value.ToString();
    if (bits_type->is_signed()) {
      return InterpValue::MakeBits(InterpValueTag::kSBits,
                                   value.GetBitsOrDie());
    }
    return value;
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsBits()) << value.ToString();
    if (enum_type->is_signed()) {
      return InterpValue::MakeBits(InterpValueTag::kSBits,
                                   value.GetBitsOrDie());
    }
    return value;
  }
  return absl::UnimplementedError("Cannot sign convert type: " +
                                  concrete_type.ToString());
}

absl::StatusOr<std::vector<InterpValue>> SignConvertArgs(
    const FunctionType& fn_type, absl::Span<const InterpValue> args) {
  absl::Span<const std::unique_ptr<ConcreteType>> params = fn_type.params();
  XLS_RET_CHECK_EQ(params.size(), args.size());
  std::vector<InterpValue> converted;
  converted.reserve(args.size());
  for (int64_t i = 0; i < args.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         SignConvertValue(*params[i], args[i]));
    converted.push_back(value);
  }
  return converted;
}

absl::StatusOr<InterpValue> ValueToInterpValue(const Value& v,
                                               const ConcreteType* type) {
  switch (v.kind()) {
    case ValueKind::kBits: {
      InterpValueTag tag = InterpValueTag::kUBits;
      if (type != nullptr) {
        XLS_RET_CHECK(type != nullptr);
        auto* bits_type = dynamic_cast<const BitsType*>(type);
        tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                     : InterpValueTag::kUBits;
      }
      return InterpValue::MakeBits(tag, v.bits());
    }
    case ValueKind::kArray:
    case ValueKind::kTuple: {
      auto get_type = [&](int64_t i) -> const ConcreteType* {
        if (type == nullptr) {
          return nullptr;
        }
        if (v.kind() == ValueKind::kArray) {
          auto* array_type = dynamic_cast<const ArrayType*>(type);
          XLS_CHECK(array_type != nullptr);
          return &array_type->element_type();
        }
        auto* tuple_type = dynamic_cast<const TupleType*>(type);
        XLS_CHECK(tuple_type != nullptr);
        return &tuple_type->GetMemberType(i);
      };
      std::vector<InterpValue> members;
      for (int64_t i = 0; i < v.elements().size(); ++i) {
        const Value& e = v.elements()[i];
        XLS_ASSIGN_OR_RETURN(InterpValue iv,
                             ValueToInterpValue(e, get_type(i)));
        members.push_back(iv);
      }
      if (v.kind() == ValueKind::kTuple) {
        return InterpValue::MakeTuple(std::move(members));
      }
      return InterpValue::MakeArray(std::move(members));
    }
    default:
      return absl::InvalidArgumentError(
          "Cannot convert IR value to interpreter value: " + v.ToString());
  }
}

absl::StatusOr<std::vector<InterpValue>> ParseArgs(
    std::string_view args_text) {
  args_text = absl::StripAsciiWhitespace(args_text);
  std::vector<InterpValue> args;
  if (args_text.empty()) {
    return args;
  }
  for (std::string_view piece : absl::StrSplit(args_text, ';')) {
    piece = absl::StripAsciiWhitespace(piece);
    XLS_ASSIGN_OR_RETURN(InterpValue value, InterpValueFromString(piece));
    args.push_back(value);
  }
  return args;
}

absl::StatusOr<std::vector<std::vector<InterpValue>>> ParseArgsBatch(
    std::string_view args_text) {
  args_text = absl::StripAsciiWhitespace(args_text);
  std::vector<std::vector<InterpValue>> args_batch;
  if (args_text.empty()) {
    return args_batch;
  }
  for (std::string_view line : absl::StrSplit(args_text, '\n')) {
    XLS_ASSIGN_OR_RETURN(auto args, ParseArgs(line));
    args_batch.push_back(std::move(args));
  }
  return args_batch;
}

}  // namespace xls::dslx
