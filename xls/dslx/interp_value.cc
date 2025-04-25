// Copyright 2020 The XLS Authors
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

#include "xls/dslx/interp_value.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls::dslx {

std::string TagToString(InterpValueTag tag) {
  switch (tag) {
    case InterpValueTag::kUBits:
      return "ubits";
    case InterpValueTag::kSBits:
      return "sbits";
    case InterpValueTag::kTuple:
      return "tuple";
    case InterpValueTag::kArray:
      return "array";
    case InterpValueTag::kEnum:
      return "enum";
    case InterpValueTag::kFunction:
      return "function";
    case InterpValueTag::kToken:
      return "token";
    case InterpValueTag::kChannelReference:
      return "channel_reference";
  }
  return absl::StrFormat("<invalid InterpValueTag(%d)>",
                         static_cast<int64_t>(tag));
}

/* static */ InterpValue InterpValue::MakeTuple(
    std::vector<InterpValue> members) {
  return InterpValue{InterpValueTag::kTuple, std::move(members)};
}

/* static */ absl::StatusOr<InterpValue> InterpValue::MakeArray(
    std::vector<InterpValue> elements) {
  return InterpValue{InterpValueTag::kArray, std::move(elements)};
}

/* static */ InterpValue InterpValue::MakeUBits(int64_t bit_count,
                                                int64_t value) {
  return InterpValue{InterpValueTag::kUBits,
                     UBits(value, /*bit_count=*/bit_count)};
}

/* static */ InterpValue InterpValue::MakeZeroValue(bool is_signed,
                                                    int64_t bit_count) {
  return InterpValue{
      is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits,
      Bits(bit_count)};
}

/* static */ InterpValue InterpValue::MakeMaxValue(bool is_signed,
                                                   int64_t bit_count) {
  auto bits = Bits::AllOnes(bit_count);
  if (!is_signed) {
    return InterpValue{InterpValueTag::kUBits, std::move(bits)};
  }
  // Unset the highest bit to get the maximum value in two's complement form.
  if (bit_count > 0) {
    bits = bits.UpdateWithSet(bit_count - 1, false);
  }
  return InterpValue{InterpValueTag::kSBits, std::move(bits)};
}

/* static */ InterpValue InterpValue::MakeMinValue(bool is_signed,
                                                   int64_t bit_count) {
  auto bits = Bits(bit_count);
  if (!is_signed) {
    return InterpValue{InterpValueTag::kUBits, std::move(bits)};
  }
  // Set the highest bit to get the most-negative value in two's complement
  // form.
  if (bit_count > 0) {
    bits = bits.UpdateWithSet(bit_count - 1, true);
  }
  return InterpValue{InterpValueTag::kSBits, std::move(bits)};
}

/* static */ InterpValue InterpValue::MakeSBits(int64_t bit_count,
                                                int64_t value) {
  return InterpValue{InterpValueTag::kSBits,
                     SBits(value, /*bit_count=*/bit_count)};
}

// Converts an interp value (precondition: `v.IsBits()`) to a string, given a
// format preference.
static std::string InterpValueBitsToString(const InterpValue& v,
                                           FormatPreference format,
                                           bool include_type_prefix = true) {
  const Bits& bits = v.GetBitsOrDie();
  const int64_t bit_count = v.GetBitCount().value();

  auto bits_type_str = [bit_count, v]() {
    bool is_signed = v.tag() == InterpValueTag::kSBits;
    // Note: The DSL defines the first 128 bits as named builtin types, beyond
    // that we have to use the uN/sN array-looking variants.
    if (bit_count <= 128) {
      return absl::StrFormat("%c%d", is_signed ? 's' : 'u', bit_count);
    }
    return absl::StrFormat("%cN[%d]", is_signed ? 's' : 'u', bit_count);
  };

  switch (v.tag()) {
    case InterpValueTag::kUBits: {
      std::string value_str = BitsToString(bits, format);
      if (format == FormatPreference::kSignedDecimal && bits.msb()) {
        value_str =
            absl::StrCat("-", BitsToString(bits_ops::Negate(bits), format));
      }
      if (!include_type_prefix) {
        return value_str;
      }
      std::string type_str = bits_type_str();
      return absl::StrCat(type_str, ":", value_str);
    }
    case InterpValueTag::kSBits: {
      std::string value_str = BitsToString(bits, format);
      if ((format == FormatPreference::kSignedDecimal ||
           format == FormatPreference::kDefault) &&
          bits.msb()) {
        // If we're a signed number in decimal format, give the value for the
        // bit pattern that has the leading negative sign.
        value_str =
            absl::StrCat("-", BitsToString(bits_ops::Negate(bits), format));
      }
      if (!include_type_prefix) {
        return value_str;
      }
      std::string type_str = bits_type_str();
      return absl::StrCat(type_str, ":", value_str);
    }
    default:
      break;
  }
  LOG(FATAL) << "Invalid tag for InterpValueBitsToString: " << v.tag();
}

std::string InterpValue::ToString(bool humanize,
                                  FormatPreference format) const {
  auto make_guts = [&] {
    return absl::StrJoin(
        GetValuesOrDie(), ", ",
        [humanize, format](std::string* out, const InterpValue& v) {
          if (humanize && v.IsBits()) {
            absl::StrAppend(out,
                            InterpValueBitsToString(
                                v, format, /*include_type_prefix=*/!humanize));
            return;
          }
          absl::StrAppend(out, v.ToString(humanize, format));
        });
  };

  switch (tag_) {
    case InterpValueTag::kUBits:
    case InterpValueTag::kSBits:
      return InterpValueBitsToString(*this, format,
                                     /*include_type_prefix=*/!humanize);
    case InterpValueTag::kArray:
      return absl::StrFormat("[%s]", make_guts());
    case InterpValueTag::kTuple:
      return absl::StrFormat("(%s)", make_guts());
    case InterpValueTag::kEnum: {
      EnumData enum_data = std::get<EnumData>(payload_);
      return absl::StrFormat("%s:%v", enum_data.def->identifier(),
                             enum_data.value);
    }
    case InterpValueTag::kFunction:
      if (std::holds_alternative<Builtin>(GetFunctionOrDie())) {
        return absl::StrCat(
            "builtin:", BuiltinToString(std::get<Builtin>(GetFunctionOrDie())));
      }
      return absl::StrCat(
          "function:",
          std::get<UserFnData>(GetFunctionOrDie()).function->identifier());
    case InterpValueTag::kToken:
      return absl::StrFormat("token:%p", GetTokenData().get());
    case InterpValueTag::kChannelReference:
      return absl::StrFormat(
          "channel_reference(%s, channel_instance_id=%s)",
          ChannelDirectionToString(GetChannelReferenceOrDie().GetDirection()),
          GetChannelReferenceOrDie().GetChannelId().has_value()
              ? absl::StrCat(*GetChannelReferenceOrDie().GetChannelId())
              : "none");
  }
  LOG(FATAL) << "Unhandled tag: " << tag_;
}

static std::string IndentString(std::string_view s, int64_t n) {
  constexpr int64_t kIndentAmount = 4;
  return absl::StrFormat("%s%s", std::string(n * kIndentAmount, ' '), s);
}

absl::StatusOr<std::string> InterpValue::ToArrayString(
    const ValueFormatDescriptor& fmt_desc, bool include_type_prefix,
    int64_t indentation) const {
  XLS_RET_CHECK(fmt_desc.IsArray());
  std::vector<std::string> pieces;
  pieces.push_back("[");
  const std::vector<InterpValue>& values = GetValuesOrDie();
  for (size_t i = 0; i < values.size(); ++i) {
    const InterpValue& v = values.at(i);
    XLS_ASSIGN_OR_RETURN(
        std::string elem,
        v.ToFormattedString(fmt_desc.array_element_format(),
                            include_type_prefix, indentation + 1));
    std::string piece = IndentString(elem, indentation + 1);
    if (i + 1 != values.size()) {
      absl::StrAppend(&piece, ",");
    }
    pieces.push_back(piece);
  }
  pieces.push_back(IndentString("]", indentation));
  return absl::StrJoin(pieces, "\n");
}

absl::StatusOr<std::string> InterpValue::ToStructString(
    const ValueFormatDescriptor& fmt_desc, bool include_type_prefix,
    int64_t indentation) const {
  if (!IsTuple()) {
    return absl::FailedPreconditionError(
        "Can only format a tuple InterpValue as a struct");
  }
  const std::vector<InterpValue>& values = GetValuesOrDie();
  if (values.size() != fmt_desc.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Number of tuple elements (%d) did not correspond to "
                        "number of struct formatting elements (%d)",
                        values.size(), fmt_desc.size()));
  }
  std::vector<std::string> pieces;
  for (int64_t i = 0; i < values.size(); ++i) {
    const InterpValue& e = values.at(i);
    const ValueFormatDescriptor& fmt_element = fmt_desc.struct_elements()[i];
    std::string_view field_name = fmt_desc.struct_field_names()[i];
    XLS_ASSIGN_OR_RETURN(
        std::string element,
        e.ToFormattedString(fmt_element, include_type_prefix, indentation + 1));
    pieces.push_back(IndentString(
        absl::StrFormat("%s: %s", field_name, element), indentation + 1));
  }
  std::string prefix = absl::StrFormat("%s {", fmt_desc.struct_name());
  std::string interior = absl::StrJoin(pieces, ",\n");
  std::string suffix = IndentString("}", indentation);
  return absl::StrJoin({prefix, interior, suffix}, "\n");
}

absl::StatusOr<std::string> InterpValue::ToTupleString(
    const ValueFormatDescriptor& fmt_desc, bool include_type_prefix,
    int64_t indentation) const {
  if (!IsTuple()) {
    return absl::FailedPreconditionError(
        "Can only format a tuple InterpValue as a struct");
  }
  const std::vector<InterpValue>& values = GetValuesOrDie();
  XLS_RET_CHECK_EQ(values.size(), fmt_desc.size());

  std::vector<std::string> pieces;
  pieces.push_back("(");
  for (int64_t i = 0; i < values.size(); ++i) {
    const InterpValue& e = values.at(i);
    const ValueFormatDescriptor& fmt_element = fmt_desc.tuple_elements()[i];
    XLS_ASSIGN_OR_RETURN(
        std::string element,
        e.ToFormattedString(fmt_element, include_type_prefix, indentation + 1));
    if (i + 1 != values.size() || values.size() == 1) {
      pieces.push_back(
          IndentString(absl::StrCat(element, ","), indentation + 1));
    } else {
      pieces.push_back(IndentString(element, indentation + 1));
    }
  }
  pieces.push_back(IndentString(")", indentation));
  return absl::StrJoin(pieces, "\n");
}

absl::StatusOr<std::string> InterpValue::ToEnumString(
    const ValueFormatDescriptor& fmt_desc) const {
  const auto& value_to_name = fmt_desc.value_to_name();
  auto it = value_to_name.find(GetBitsOrDie());
  if (it == value_to_name.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Enum value %s was not found in enum descriptor for %s",
                        ToString(), fmt_desc.enum_name()));
  }
  // We show the underlying value after displaying the enum member.
  auto underlying = InterpValue::MakeBits(IsSigned(), GetBitsOrDie());
  return absl::StrFormat("%s::%s  // %s", fmt_desc.enum_name(), it->second,
                         underlying.ToString());
}

absl::StatusOr<std::string> InterpValue::ToFormattedString(
    const ValueFormatDescriptor& fmt_desc, bool include_type_prefix,
    int64_t indentation) const {
  class Visitor : public ValueFormatVisitor {
   public:
    explicit Visitor(const InterpValue& v, bool include_type_prefix,
                     int64_t indentation)
        : v_(v),
          include_type_prefix_(include_type_prefix),
          indentation_(indentation) {}

    absl::Status HandleStruct(const ValueFormatDescriptor& d) override {
      XLS_ASSIGN_OR_RETURN(
          result_, v_.ToStructString(d, include_type_prefix_, indentation_));
      return absl::OkStatus();
    }
    absl::Status HandleArray(const ValueFormatDescriptor& d) override {
      XLS_ASSIGN_OR_RETURN(
          result_, v_.ToArrayString(d, include_type_prefix_, indentation_));
      return absl::OkStatus();
    }
    absl::Status HandleEnum(const ValueFormatDescriptor& d) override {
      XLS_ASSIGN_OR_RETURN(result_, v_.ToEnumString(d));
      return absl::OkStatus();
    }
    absl::Status HandleTuple(const ValueFormatDescriptor& d) override {
      XLS_ASSIGN_OR_RETURN(
          result_, v_.ToTupleString(d, include_type_prefix_, indentation_));
      return absl::OkStatus();
    }
    absl::Status HandleLeafValue(const ValueFormatDescriptor& d) override {
      result_ =
          v_.ToString(/*humanize=*/!include_type_prefix_, d.leaf_format());
      return absl::OkStatus();
    }

    const std::optional<std::string>& result() const { return result_; }

   private:
    const InterpValue& v_;
    bool include_type_prefix_;
    const int64_t indentation_;
    std::optional<std::string> result_;
  };

  Visitor v(*this, include_type_prefix, indentation);
  XLS_RETURN_IF_ERROR(fmt_desc.Accept(v));
  XLS_RET_CHECK(v.result().has_value());
  return v.result().value();
}

bool InterpValue::Eq(const InterpValue& other) const {
  auto values_equal = [&] {
    const std::vector<InterpValue>& lhs = GetValuesOrDie();
    const std::vector<InterpValue>& rhs = other.GetValuesOrDie();
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (int64_t i = 0; i < lhs.size(); ++i) {
      const InterpValue& lhs_elem = lhs[i];
      const InterpValue& rhs_elem = rhs[i];
      if (lhs_elem.Ne(rhs_elem)) {
        return false;
      }
    }
    return true;
  };

  switch (tag_) {
    // Note: the interpreter doesn't always reify enums into enum-producing
    // expressions (e.g. casts) so we rely on the typechecker to determine when
    // enum equality comparisons are legitimate to perform and just check bit
    // patterns here.
    //
    // TODO(leary): 2020-10-18 This could be switched if we ensured that
    // enum-producing expressions always made an enum value, but as it stands a
    // bit value can be used in any place an enum type is annotated.
    case InterpValueTag::kSBits:
    case InterpValueTag::kUBits:
    case InterpValueTag::kEnum: {
      return other.HasBits() && GetBitsOrDie() == other.GetBitsOrDie();
    }
    case InterpValueTag::kToken:
      return other.IsToken() && GetTokenData() == other.GetTokenData();
    case InterpValueTag::kArray: {
      if (!other.IsArray()) {
        return false;
      }
      return values_equal();
    }
    case InterpValueTag::kTuple: {
      if (!other.IsTuple()) {
        return false;
      }
      return values_equal();
    }
    // Functions can't be compared for equality, as they may have parametrics
    // that would differentiate one function from another, even if they have the
    // same module and generic implementation.
    case InterpValueTag::kFunction:
      break;
    case InterpValueTag::kChannelReference:
      return GetChannelReferenceOrDie().GetDirection() ==
                 other.GetChannelReferenceOrDie().GetDirection() &&
             GetChannelReferenceOrDie().GetChannelId().has_value() &&
             GetChannelReferenceOrDie().GetChannelId() ==
                 other.GetChannelReferenceOrDie().GetChannelId();
  }
  LOG(FATAL) << "Unhandled tag: " << tag_;
}

bool InterpValue::operator==(const InterpValue& rhs) const { return Eq(rhs); }

/* static */ absl::StatusOr<InterpValue> InterpValue::Compare(
    const InterpValue& lhs, const InterpValue& rhs, CompareF ucmp,
    CompareF scmp) {
  if (lhs.tag_ != rhs.tag_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Same tag is required for a comparison operation: lhs "
                        "tag: %s, rhs tag: %s, lhs value: %s, rhs value: %s",
                        TagToString(lhs.tag_), TagToString(rhs.tag_),
                        lhs.ToString(), rhs.ToString()));
  }
  switch (lhs.tag_) {
    case InterpValueTag::kUBits:
      return MakeBool(ucmp(lhs.GetBitsOrDie(), rhs.GetBitsOrDie()));
    case InterpValueTag::kSBits:
      return MakeBool(scmp(lhs.GetBitsOrDie(), rhs.GetBitsOrDie()));
    default:
      return absl::InvalidArgumentError("Invalid values for comparison: " +
                                        TagToString(lhs.tag_));
  }
}

absl::StatusOr<InterpValue> InterpValue::Gt(const InterpValue& other) const {
  return Compare(*this, other, &bits_ops::UGreaterThan,
                 &bits_ops::SGreaterThan);
}

absl::StatusOr<InterpValue> InterpValue::Ge(const InterpValue& other) const {
  return Compare(*this, other, &bits_ops::UGreaterThanOrEqual,
                 &bits_ops::SGreaterThanOrEqual);
}

absl::StatusOr<InterpValue> InterpValue::Le(const InterpValue& other) const {
  return Compare(*this, other, &bits_ops::ULessThanOrEqual,
                 &bits_ops::SLessThanOrEqual);
}

absl::StatusOr<InterpValue> InterpValue::Lt(const InterpValue& other) const {
  return Compare(*this, other, &bits_ops::ULessThan, &bits_ops::SLessThan);
}

std::optional<InterpValue> InterpValue::Increment() const {
  CHECK(IsBits());
  Bits b = GetBitsOrDie();
  if (*this == MakeMaxValue(IsSigned(), GetBitCount().value())) {
    return std::nullopt;  // Overflow case.
  }
  return InterpValue(tag_, bits_ops::Increment(b));
}

std::optional<InterpValue> InterpValue::Decrement() const {
  CHECK(IsBits());
  if (*this == MakeMinValue(IsSigned(), GetBitCount().value())) {
    return std::nullopt;  // Underflow case.
  }
  Bits b = GetBitsOrDie();
  return InterpValue(tag_, bits_ops::Decrement(b));
}

absl::StatusOr<InterpValue> InterpValue::IncrementZeroExtendIfOverflow() const {
  XLS_ASSIGN_OR_RETURN(Bits bits, GetBits());
  if (*this == MakeMaxValue(IsSigned(), bits.bit_count())) {
    XLS_ASSIGN_OR_RETURN(InterpValue extended, ZeroExt(bits.bit_count() + 1));
    XLS_ASSIGN_OR_RETURN(bits, extended.GetBits());
  }
  return InterpValue(tag_, bits_ops::Increment(bits));
}

absl::StatusOr<InterpValue> InterpValue::Min(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(InterpValue lt, Lt(other));
  if (lt.IsTrue()) {
    return *this;
  }
  return other;
}

absl::StatusOr<InterpValue> InterpValue::Max(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(InterpValue lt, Lt(other));
  if (lt.IsTrue()) {
    return other;
  }
  return *this;
}

absl::StatusOr<InterpValue> InterpValue::BitwiseNegate() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return InterpValue(tag_, bits_ops::Not(b));
}

absl::StatusOr<InterpValue> InterpValue::BitwiseXor(
    const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  return InterpValue(tag_, bits_ops::Xor(lhs, rhs));
}

absl::StatusOr<InterpValue> InterpValue::BitwiseOr(
    const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  return InterpValue(tag_, bits_ops::Or(lhs, rhs));
}

absl::StatusOr<InterpValue> InterpValue::BitwiseAnd(
    const InterpValue& other) const {
  XLS_RET_CHECK_EQ(tag(), other.tag());
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  return InterpValue(tag_, bits_ops::And(lhs, rhs));
}

absl::StatusOr<InterpValue> InterpValue::Sub(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  if (lhs.bit_count() != rhs.bit_count()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Interpreter value sub requires lhs and rhs to have "
                        "same bit count; got %d vs %d",
                        lhs.bit_count(), rhs.bit_count()));
  }
  return InterpValue(tag_, bits_ops::Sub(lhs, rhs));
}

absl::StatusOr<InterpValue> InterpValue::Add(const InterpValue& other) const {
  XLS_RET_CHECK(IsBits() && other.IsBits());
  XLS_RET_CHECK_EQ(tag(), other.tag());
  XLS_RET_CHECK_EQ(GetBitCount().value(), other.GetBitCount().value());
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  return InterpValue(tag_, bits_ops::Add(lhs, rhs));
}

absl::StatusOr<InterpValue> InterpValue::SCmp(const InterpValue& other,
                                              std::string_view method) {
  // Note: no tag check, because this is an explicit request for a signed
  // comparison, we're conceptually "coercing" the operands if need be.
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  bool result;
  if (method == "slt") {
    result = bits_ops::SLessThan(lhs, rhs);
  } else if (method == "sle") {
    result = bits_ops::SLessThanOrEqual(lhs, rhs);
  } else if (method == "sgt") {
    result = bits_ops::SGreaterThan(lhs, rhs);
  } else if (method == "sge") {
    result = bits_ops::SGreaterThanOrEqual(lhs, rhs);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid scmp method: ", method));
  }
  return InterpValue::MakeBool(result);
}

absl::StatusOr<InterpValue> InterpValue::Mul(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  if (lhs.bit_count() != rhs.bit_count()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot mul different width values: lhs `%s` vs rhs `%s`",
        BitsToString(lhs, FormatPreference::kDefault, true),
        BitsToString(rhs, FormatPreference::kDefault, true)));
  }
  return InterpValue(tag_, bits_ops::UMul(lhs, rhs).Slice(0, lhs.bit_count()));
}

absl::StatusOr<InterpValue> InterpValue::Slice(
    const InterpValue& start, const InterpValue& length) const {
  if (IsBits()) {
    // Type checker should be enforcing that all of these are ubits.
    XLS_RET_CHECK(IsUBits());
    XLS_RET_CHECK(start.IsUBits());
    XLS_RET_CHECK(length.IsUBits());

    XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
    XLS_ASSIGN_OR_RETURN(Bits length_bits, length.GetBits());
    XLS_ASSIGN_OR_RETURN(uint64_t start_index, start_bits.ToUint64());
    XLS_ASSIGN_OR_RETURN(uint64_t length_index, length_bits.ToUint64());
    return MakeBits(InterpValueTag::kUBits,
                    GetBitsOrDie().Slice(start_index, length_index));
  }
  if (tag_ != InterpValueTag::kArray) {
    return absl::InvalidArgumentError("Can only slice bits and array values");
  }
  if (!start.IsUBits() || !length.IsArray()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Bad types to slice operation: subject %s start %s length %s",
        TagToString(tag()), TagToString(start.tag()),
        TagToString(length.tag())));
  }
  const std::vector<InterpValue>& subject = GetValuesOrDie();
  XLS_ASSIGN_OR_RETURN(const auto* length_values, length.GetValues());
  int64_t length_value = length_values->size();
  XLS_ASSIGN_OR_RETURN(int64_t start_width, start.GetBitCount());
  int64_t width = start_width + Bits::MinBitCountSigned(length_value) +
                  Bits::MinBitCountUnsigned(subject.size()) + 1;
  std::vector<InterpValue> result;
  for (int64_t i = 0; i < length_value; ++i) {
    XLS_ASSIGN_OR_RETURN(InterpValue start_big, start.ZeroExt(width));
    XLS_ASSIGN_OR_RETURN(InterpValue offset,
                         start_big.Add(MakeUBits(width, i)));
    XLS_ASSIGN_OR_RETURN(InterpValue out_of_bounds,
                         offset.Ge(MakeUBits(width, subject.size())));
    if (out_of_bounds.IsTrue()) {
      result.push_back(subject.back());
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t offset_int, offset.GetBitValueUnsigned());
      result.push_back(subject[offset_int]);
    }
  }
  return InterpValue(InterpValueTag::kArray, result);
}

absl::StatusOr<InterpValue> InterpValue::Index(int64_t index) const {
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* lhs, GetValues());
  if (lhs->size() <= index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Index out of bounds; index: %d >= %d elements; lhs: %s", index,
        lhs->size(), ToString()));
  }
  return (*lhs)[index];
}

absl::StatusOr<InterpValue> InterpValue::Index(const InterpValue& other) const {
  XLS_RET_CHECK(other.IsUBits());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* lhs, GetValues());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  XLS_ASSIGN_OR_RETURN(uint64_t index, rhs.ToUint64());
  if (lhs->size() <= index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Index out of bounds; index: %d >= %d elements; lhs: %s", index,
        lhs->size(), ToString()));
  }
  return (*lhs)[index];
}

absl::StatusOr<InterpValue> InterpValue::Update(
    const InterpValue& index, const InterpValue& value) const {
  absl::Span<const xls::dslx::InterpValue> indices;
  if (index.IsTuple()) {
    indices =
        absl::MakeConstSpan(std::get<std::vector<InterpValue>>(index.payload_));
  } else {
    indices = absl::MakeConstSpan(&index, 1);
  }
  InterpValue copy = *this;
  InterpValue* element = &copy;
  for (const auto& i : indices) {
    if (!element->IsArray()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Update of non-array element: %s", element->ToString()));
    }
    std::vector<InterpValue>& values =
        std::get<std::vector<InterpValue>>(element->payload_);
    XLS_ASSIGN_OR_RETURN(Bits index_bits, i.GetBits());
    XLS_ASSIGN_OR_RETURN(uint64_t index_value, index_bits.ToUint64());
    if (index_value >= values.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Update index %d is out of bounds; subject size: %d",
                          index_value, values.size()));
    }
    element = &values[index_value];
  }
  *element = value;
  return copy;
}

absl::StatusOr<InterpValue> InterpValue::ArithmeticNegate() const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());
  return InterpValue(tag_, bits_ops::Negate(arg));
}

absl::StatusOr<InterpValue> InterpValue::CeilOfLog2() const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());
  if (arg.IsZero()) {
    return InterpValue(tag_, UBits(0, 32));
  }
  // Subtract one to make sure we get the right result for exact powers of 2.
  int64_t min_bit_width =
      arg.bit_count() - bits_ops::Decrement(arg).CountLeadingZeros();
  return InterpValue(tag_, UBits(min_bit_width, 32));
}

absl::StatusOr<Bits> InterpValue::GetBits() const {
  if (std::holds_alternative<Bits>(payload_)) {
    return std::get<Bits>(payload_);
  }

  if (std::holds_alternative<EnumData>(payload_)) {
    return std::get<EnumData>(payload_).value;
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Value %s does not contain bits.", ToString()));
}

const Bits& InterpValue::GetBitsOrDie() const {
  if (std::holds_alternative<Bits>(payload_)) {
    return std::get<Bits>(payload_);
  }

  return std::get<EnumData>(payload_).value;
}

absl::StatusOr<InterpValue::ChannelReference> InterpValue::GetChannelReference()
    const {
  if (std::holds_alternative<ChannelReference>(payload_)) {
    return std::get<ChannelReference>(payload_);
  }
  return absl::InvalidArgumentError(
      "Value does not contain a channel reference.");
}

// Returns the minimum of the given bits value interpreted as an unsigned
// number and limit.
static int64_t ClampedUnsignedValue(const Bits& bits, int64_t limit) {
  if (limit < 0) {
    return limit;
  }
  std::optional<int64_t> bits_int = bits_ops::TryUnsignedBitsToInt64(bits);
  if (bits_int.has_value() && *bits_int <= limit) {
    return *bits_int;
  }
  return limit;
}

absl::StatusOr<InterpValue> InterpValue::Shl(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  int64_t amount64 = ClampedUnsignedValue(rhs, lhs.bit_count());
  return InterpValue(tag_, bits_ops::ShiftLeftLogical(lhs, amount64));
}

absl::StatusOr<InterpValue> InterpValue::Shrl(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  int64_t amount64 = ClampedUnsignedValue(rhs, lhs.bit_count());
  return InterpValue(tag_, bits_ops::ShiftRightLogical(lhs, amount64));
}

absl::StatusOr<InterpValue> InterpValue::Shra(const InterpValue& other) const {
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  int64_t amount64 = ClampedUnsignedValue(rhs, lhs.bit_count());
  return InterpValue(tag_, bits_ops::ShiftRightArith(lhs, amount64));
}

absl::StatusOr<InterpValue> InterpValue::ZeroExt(int64_t new_bit_count) const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  InterpValueTag new_tag =
      IsSigned() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  if (new_bit_count < b.bit_count()) {
    return MakeBits(new_tag, b.Slice(0, new_bit_count));
  }
  return InterpValue(new_tag, bits_ops::ZeroExtend(b, new_bit_count));
}

absl::StatusOr<InterpValue> InterpValue::Decode(int64_t new_bit_count) const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());

  absl::StatusOr<uint64_t> unsigned_index = arg.ToUint64();
  if (!unsigned_index.ok() ||
      *unsigned_index >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    // Index cannot be represented in a 64-bit signed integer - so it's
    // telling us to set a bit that's definitely out of range. Return 0.
    return InterpValue(InterpValueTag::kUBits, Bits(new_bit_count));
  }

  const int64_t index = static_cast<int64_t>(*unsigned_index);
  InlineBitmap result(new_bit_count);
  if (index < new_bit_count) {
    result.Set(index);
  }
  return InterpValue(InterpValueTag::kUBits,
                     Bits::FromBitmap(std::move(result)));
}

absl::StatusOr<InterpValue> InterpValue::Encode() const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());
  int64_t result = 0;
  for (int64_t i = 0; i < arg.bit_count(); ++i) {
    if (arg.Get(i)) {
      result |= i;
    }
  }
  return InterpValue(InterpValueTag::kUBits,
                     UBits(result, ::xls::CeilOfLog2(arg.bit_count())));
}

absl::StatusOr<InterpValue> InterpValue::OneHot(bool lsb_prio) const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());
  if (lsb_prio) {
    return InterpValue(InterpValueTag::kUBits, bits_ops::OneHotLsbToMsb(arg));
  }
  return InterpValue(InterpValueTag::kUBits, bits_ops::OneHotMsbToLsb(arg));
}

absl::StatusOr<InterpValue> InterpValue::SignExt(int64_t new_bit_count) const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  InterpValueTag new_tag =
      IsSigned() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  if (new_bit_count < b.bit_count()) {
    return MakeBits(new_tag, b.Slice(0, new_bit_count));
  }
  return InterpValue(new_tag, bits_ops::SignExtend(b, new_bit_count));
}

absl::StatusOr<InterpValue> InterpValue::FloorDiv(
    const InterpValue& other) const {
  if (tag_ != other.tag_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot floordiv values: %s vs %s", TagToString(tag_),
                        TagToString(other.tag_)));
  }
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  Bits result;
  if (IsSBits()) {
    result = bits_ops::SDiv(lhs, rhs);
  } else {
    result = bits_ops::UDiv(lhs, rhs);
  }
  return InterpValue(tag_, result);
}

absl::StatusOr<InterpValue> InterpValue::FloorMod(
    const InterpValue& other) const {
  if (tag_ != other.tag_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot floormod values: %s vs %s", TagToString(tag_),
                        TagToString(other.tag_)));
  }
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  Bits result;
  if (IsSBits()) {
    result = bits_ops::SMod(lhs, rhs);
  } else {
    result = bits_ops::UMod(lhs, rhs);
  }
  return InterpValue(tag_, result);
}

absl::StatusOr<InterpValue> InterpValue::Concat(
    const InterpValue& other) const {
  if (tag_ != other.tag_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot concatenate values: %s vs %s",
                        TagToString(tag_), TagToString(other.tag_)));
  }
  switch (tag_) {
    case InterpValueTag::kArray: {
      std::vector<InterpValue> result = GetValuesOrDie();
      for (const InterpValue& o : other.GetValuesOrDie()) {
        result.push_back(o);
      }
      return InterpValue(InterpValueTag::kArray, std::move(result));
    }
    case InterpValueTag::kUBits:
      return InterpValue(
          tag_, bits_ops::Concat({GetBitsOrDie(), other.GetBitsOrDie()}));
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Cannot concatenate %s values", TagToString(tag_)));
  }
}

absl::StatusOr<InterpValue> InterpValue::Flatten() const {
  if (tag_ == InterpValueTag::kUBits) {
    return *this;
  }
  if (tag_ == InterpValueTag::kArray) {
    Bits accum;
    for (const InterpValue& e : GetValuesOrDie()) {
      XLS_ASSIGN_OR_RETURN(InterpValue flat, e.Flatten());
      accum = bits_ops::Concat({accum, flat.GetBitsOrDie()});
    }
    return InterpValue(InterpValueTag::kUBits, accum);
  }
  return absl::InvalidArgumentError("Cannot flatten values with tag: " +
                                    TagToString(tag_));
}

absl::StatusOr<int64_t> InterpValue::GetBitCount() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return b.bit_count();
}

absl::StatusOr<int64_t> InterpValue::GetBitValueViaSign() const {
  if (IsEnum()) {
    EnumData enum_data = std::get<EnumData>(payload_);
    if (enum_data.is_signed) {
      return GetBitValueSigned();
    }

    return GetBitValueUnsigned();
  }
  if (IsSBits()) {
    return GetBitValueSigned();
  }
  if (IsUBits()) {
    return GetBitValueUnsigned();
  }
  return absl::InvalidArgumentError("Value cannot be converted to bits: " +
                                    ToHumanString());
}

absl::StatusOr<uint64_t> InterpValue::GetBitValueUnsigned() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return b.ToUint64();
}

bool InterpValue::FitsInUint64() const {
  return HasBits() && GetBitsOrDie().FitsInUint64();
}

bool InterpValue::FitsInNBitsUnsigned(int64_t n) const {
  return HasBits() && GetBitsOrDie().FitsInNBitsUnsigned(n);
}

absl::StatusOr<int64_t> InterpValue::GetBitValueSigned() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return b.ToInt64();
}

bool InterpValue::FitsInNBitsSigned(int64_t n) const {
  return HasBits() && GetBitsOrDie().FitsInNBitsSigned(n);
}

/* static */ absl::StatusOr<std::vector<xls::Value>>
InterpValue::ConvertValuesToIr(absl::Span<InterpValue const> values) {
  std::vector<xls::Value> converted;
  for (const InterpValue& v : values) {
    XLS_ASSIGN_OR_RETURN(Value c, v.ConvertToIr());
    converted.push_back(c);
  }
  return converted;
}

absl::StatusOr<xls::Value> InterpValue::ConvertToIr() const {
  switch (tag_) {
    case InterpValueTag::kUBits: {
      return xls::Value(GetBitsOrDie());
    }
    case InterpValueTag::kSBits: {
      return xls::Value(GetBitsOrDie());
    }
    case InterpValueTag::kArray: {
      XLS_ASSIGN_OR_RETURN(std::vector<xls::Value> converted,
                           ConvertValuesToIr(GetValuesOrDie()));
      return xls::Value::Array(converted);
    }
    case InterpValueTag::kTuple: {
      XLS_ASSIGN_OR_RETURN(std::vector<xls::Value> converted,
                           ConvertValuesToIr(GetValuesOrDie()));
      return xls::Value::Tuple(converted);
    }
    case InterpValueTag::kEnum: {
      return xls::Value(GetBitsOrDie());
    }
    case InterpValueTag::kToken: {
      return xls::Value::Token();
    }
    case InterpValueTag::kFunction: {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot convert functions to IR: %s", ToString(/*humanize=*/true)));
    }
    case InterpValueTag::kChannelReference: {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cannot convert channel-reference-typed values to IR."));
    }
  }
  LOG(FATAL) << "Unhandled tag: " << tag_;
}

bool InterpValue::operator<(const InterpValue& rhs) const {
  if (IsBits()) {
    if (!rhs.IsBits()) {
      return true;
    }

    return Lt(rhs).value().IsTrue();
  }

  if (IsTuple()) {
    if (rhs.IsArray()) {
      return true;
    }

    if (rhs.IsBits()) {
      return false;
    }
  } else {
    if (!rhs.IsArray()) {
      return false;
    }
  }

  // Common code for arrays & tuples.
  int64_t lhs_length = GetLength().value();
  int64_t rhs_length = rhs.GetLength().value();
  if (lhs_length < rhs_length) {
    return true;
  }

  if (rhs_length < lhs_length) {
    return false;
  }

  for (int i = 0; i < lhs_length; i++) {
    InterpValue index = MakeU32(i);
    InterpValue lhs_element = Index(index).value();
    InterpValue rhs_element = rhs.Index(index).value();
    if (lhs_element < rhs_element) {
      return true;
    }

    if (rhs_element < lhs_element) {
      return false;
    }
  }
  return false;
}

bool InterpValue::operator>=(const InterpValue& rhs) const {
  return !(*this < rhs);
}

}  // namespace xls::dslx
