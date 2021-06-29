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

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {

absl::StatusOr<Builtin> BuiltinFromString(absl::string_view name) {
#define TRY_NAME(__name, __enum) \
  if (name == __name) {          \
    return Builtin::__enum;      \
  }
  XLS_DSLX_BUILTIN_EACH(TRY_NAME)
#undef TRY_NAME
  return absl::InvalidArgumentError(
      absl::StrFormat("Name is not a DSLX builtin: \"%s\"", name));
}

std::string BuiltinToString(Builtin builtin) {
  switch (builtin) {
#define CASIFY(__str, __enum) \
  case Builtin::__enum:       \
    return __str;
    XLS_DSLX_BUILTIN_EACH(CASIFY)
#undef CASIFY
  }
  return absl::StrFormat("<invalid Builtin(%d)>",
                         static_cast<int64_t>(builtin));
}

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

/* static */ InterpValue InterpValue::MakeSBits(int64_t bit_count,
                                                int64_t value) {
  return InterpValue{InterpValueTag::kSBits,
                     UBits(value, /*bit_count=*/bit_count)};
}

// Converts an interp value (precondition: `v.IsBits()`) to a string, given a
// format preference.
static std::string BitsToString(const InterpValue& v, FormatPreference format,
                                bool include_type = true) {
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
      std::string value_str = bits.ToString(format);
      if (!include_type) {
        return value_str;
      }
      std::string type_str = bits_type_str();
      return absl::StrCat(type_str, ":", value_str);
    }
    case InterpValueTag::kSBits: {
      std::string value_str = bits.ToString(format);
      if ((format == FormatPreference::kDecimal ||
           format == FormatPreference::kDefault) &&
          bits.msb()) {
        // If we're a signed number in decimal format, give the value for the
        // bit pattern that has the leading negative sign.
        value_str = absl::StrCat("-", bits_ops::Negate(bits).ToString(format));
      }
      if (!include_type) {
        return value_str;
      }
      std::string type_str = bits_type_str();
      return absl::StrCat(type_str, ":", value_str);
    }
    default:
      break;
  }
  XLS_LOG(FATAL) << "Invalid tag for BitsToString: " << v.tag();
}

std::string InterpValue::ToString(bool humanize,
                                  FormatPreference format) const {
  auto make_guts = [&] {
    return absl::StrJoin(
        GetValuesOrDie(), ", ",
        [humanize, format](std::string* out, const InterpValue& v) {
          if (humanize && v.IsBits()) {
            absl::StrAppend(out,
                            BitsToString(v, format, /*include_type=*/false));
            return;
          }
          absl::StrAppend(out, v.ToString(humanize, format));
        });
  };

  switch (tag_) {
    case InterpValueTag::kUBits:
    case InterpValueTag::kSBits:
      return BitsToString(*this, format);
    case InterpValueTag::kArray:
      return absl::StrFormat("[%s]", make_guts());
    case InterpValueTag::kTuple:
      return absl::StrFormat("(%s)", make_guts());
    case InterpValueTag::kEnum:
      return absl::StrFormat("%s:%s", type_->identifier(),
                             GetBitsOrDie().ToString());
    case InterpValueTag::kFunction:
      if (absl::holds_alternative<Builtin>(GetFunctionOrDie())) {
        return absl::StrCat(
            "builtin:",
            BuiltinToString(absl::get<Builtin>(GetFunctionOrDie())));
      }
      return absl::StrCat(
          "function:",
          absl::get<UserFnData>(GetFunctionOrDie()).function->identifier());
    case InterpValueTag::kToken:
      return absl::StrFormat("token:%p", GetTokenData().get());
  }
  XLS_LOG(FATAL) << "Unhandled tag: " << tag_;
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
    case InterpValueTag::kEnum:
      return other.HasBits() && GetBitsOrDie() == other.GetBitsOrDie();
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
    // Note: functions are ephemeral values in the interpreter (not first class
    // values users can hold / manipulate) and cannot currently be compared for
    // equality.
    case InterpValueTag::kFunction:
      break;
  }
  XLS_LOG(FATAL) << "Unhandled tag: " << tag_;
}

bool InterpValue::operator==(const InterpValue& rhs) const { return Eq(rhs); }

/* static */ absl::StatusOr<InterpValue> InterpValue::Compare(
    const InterpValue& lhs, const InterpValue& rhs, CompareF ucmp,
    CompareF scmp) {
  if (lhs.tag_ != rhs.tag_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Same tag is required for a comparison operation: lhs %s rhs %s",
        TagToString(lhs.tag_), TagToString(rhs.tag_)));
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
                                              absl::string_view method) {
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
        "Cannot mul different width values: lhs %d bits, rhs %d bits",
        lhs.bit_count(), rhs.bit_count()));
  }
  return InterpValue(tag_, bits_ops::UMul(lhs, rhs).Slice(0, lhs.bit_count()));
}

absl::StatusOr<InterpValue> InterpValue::AddWithCarry(
    const InterpValue& other) const {
  XLS_RET_CHECK(IsUBits());
  XLS_RET_CHECK(other.IsUBits());
  XLS_ASSIGN_OR_RETURN(Bits lhs, GetBits());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  int64_t extended = std::max(lhs.bit_count(), rhs.bit_count()) + 1;
  Bits new_lhs = bits_ops::ZeroExtend(lhs, extended);
  Bits new_rhs = bits_ops::ZeroExtend(rhs, extended);
  Bits result = bits_ops::Add(new_lhs, new_rhs);
  InterpValue low_bits(InterpValueTag::kUBits,
                       result.Slice(0, /*width=*/extended - 1));
  InterpValue carry(InterpValueTag::kUBits,
                    result.Slice(extended - 1, /*width=*/1));
  return InterpValue::MakeTuple({carry, low_bits});
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
  int64_t width = start_width +
                  Bits::MinBitCountSigned(length_value) +
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
      XLS_ASSIGN_OR_RETURN(uint64_t offset_int, offset.GetBitValueUint64());
      result.push_back(subject[offset_int]);
    }
  }
  return InterpValue(InterpValueTag::kArray, result);
}

absl::StatusOr<InterpValue> InterpValue::Index(const InterpValue& other) const {
  XLS_RET_CHECK(other.IsUBits());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* lhs, GetValues());
  XLS_ASSIGN_OR_RETURN(Bits rhs, other.GetBits());
  XLS_ASSIGN_OR_RETURN(uint64_t index, rhs.ToUint64());
  if (lhs->size() <= index) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Index out of bounds: %d >= %d elements", index, lhs->size()));
  }
  return (*lhs)[index];
}

absl::StatusOr<InterpValue> InterpValue::Update(
    const InterpValue& index, const InterpValue& value) const {
  XLS_RET_CHECK(index.IsUBits());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* lhs, GetValues());
  XLS_ASSIGN_OR_RETURN(Bits index_bits, index.GetBits());
  XLS_ASSIGN_OR_RETURN(uint64_t index_value, index_bits.ToUint64());
  if (index_value >= lhs->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Update index %d is out of bounds; subject size: %d",
                        index_value, lhs->size()));
  }
  std::vector<InterpValue> copy = *lhs;
  copy[index_value] = value;
  return InterpValue(tag_, std::move(copy));
}

absl::StatusOr<InterpValue> InterpValue::ArithmeticNegate() const {
  XLS_ASSIGN_OR_RETURN(Bits arg, GetBits());
  return InterpValue(tag_, bits_ops::Negate(arg));
}

absl::StatusOr<Bits> InterpValue::GetBits() const {
  if (absl::holds_alternative<Bits>(payload_)) {
    return absl::get<Bits>(payload_);
  }
  return absl::InvalidArgumentError("Value does not contain bits.");
}

// Returns the minimum of the given bits value interpreted as an unsigned
// number and limit.
static int64_t ClampedUnsignedValue(const Bits& bits, int64_t limit) {
  if (limit < 0 || bits_ops::UGreaterThanOrEqual(bits, limit)) {
    return limit;
  }
  return static_cast<int64_t>(bits.ToUint64().value());
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

absl::StatusOr<int64_t> InterpValue::GetBitValueCheckSign() const {
  if (IsSBits() || (IsEnum() && type_->signedness().value())) {
    return GetBitValueInt64();
  }
  if (IsUBits() || (IsEnum() && !type_->signedness().value())) {
    XLS_ASSIGN_OR_RETURN(uint64_t x, GetBitValueUint64());
    return x;
  }
  return absl::InvalidArgumentError("Value cannot be converted to bits: " +
                                    ToHumanString());
}

absl::StatusOr<uint64_t> InterpValue::GetBitValueUint64() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return b.ToUint64();
}

bool InterpValue::FitsInUint64() const {
  return HasBits() && GetBitsOrDie().FitsInUint64();
}

absl::StatusOr<int64_t> InterpValue::GetBitValueInt64() const {
  XLS_ASSIGN_OR_RETURN(Bits b, GetBits());
  return b.ToInt64();
}

bool InterpValue::FitsInInt64() const {
  return HasBits() && GetBitsOrDie().FitsInInt64();
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
  }
  XLS_LOG(FATAL) << "Unhandled tag: " << tag_;
}

bool InterpValue::operator<(const InterpValue& rhs) const {
  if (IsBits()) {
    if (!rhs.IsBits()) {
      return true;
    }

    return Lt(rhs).value().IsTrue();
  } else if (IsTuple()) {
    if (rhs.IsArray()) {
      return true;
    } else if (rhs.IsBits()) {
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
  } else if (rhs_length < lhs_length) {
    return false;
  }

  for (int i = 0; i < lhs_length; i++) {
    InterpValue index = MakeU32(i);
    InterpValue lhs_element = Index(index).value();
    InterpValue rhs_element = rhs.Index(index).value();
    if (lhs_element < rhs_element) {
      return true;
    } else if (rhs_element < lhs_element) {
      return false;
    }
  }
  return false;
}

absl::optional<Module*> GetFunctionValueOwner(
    const InterpValue& function_value) {
  if (function_value.IsBuiltinFunction()) {
    return absl::nullopt;
  }
  const auto& fn_data =
      absl::get<InterpValue::UserFnData>(function_value.GetFunctionOrDie());
  return fn_data.function->owner();
}

}  // namespace xls::dslx
