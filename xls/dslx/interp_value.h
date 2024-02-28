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

#ifndef XLS_DSLX_INTERP_VALUE_H_
#define XLS_DSLX_INTERP_VALUE_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls::dslx {

#define XLS_DSLX_BUILTIN_EACH(X)          \
  X("add_with_carry", kAddWithCarry)      \
  X("and_reduce", kAndReduce)             \
  X("array_rev", kArrayRev)               \
  X("array_size", kArraySize)             \
  X("assert_eq", kAssertEq)               \
  X("assert_lt", kAssertLt)               \
  X("bit_slice", kBitSlice)               \
  X("bit_slice_update", kBitSliceUpdate)  \
  X("checked_cast", kCheckedCast)         \
  X("clz", kClz)                          \
  X("cover!", kCover)                     \
  X("ctz", kCtz)                          \
  X("gate!", kGate)                       \
  X("enumerate", kEnumerate)              \
  X("fail!", kFail)                       \
  X("map", kMap)                          \
  X("decode", kDecode)                    \
  X("encode", kEncode)                    \
  X("one_hot", kOneHot)                   \
  X("one_hot_sel", kOneHotSel)            \
  X("or_reduce", kOrReduce)               \
  X("priority_sel", kPriorityhSel)        \
  X("range", kRange)                      \
  X("rev", kRev)                          \
  X("widening_cast", kWideningCast)       \
  X("select", kSelect)                    \
  X("signex", kSignex)                    \
  X("smulp", kSMulp)                      \
  X("slice", kSlice)                      \
  X("trace!", kTrace)                     \
  X("umulp", kUMulp)                      \
  X("update", kUpdate)                    \
  X("xor_reduce", kXorReduce)             \
  X("join", kJoin)                        \
  /* send/recv routines */                \
  X("send", kSend)                        \
  X("send_if", kSendIf)                   \
  X("recv", kRecv)                        \
  X("recv_if", kRecvIf)                   \
  X("recv_nonblocking", kRecvNonBlocking) \
  X("recv_if_nonblocking", kRecvIfNonBlocking)

// Enum that represents all the DSLX builtin functions.
//
// Functions can be held in values, either as user defined ones or builtin ones
// (represented via this enumerated value).
enum class Builtin {
#define ENUMIFY(__str, __enum, ...) __enum,
  XLS_DSLX_BUILTIN_EACH(ENUMIFY)
#undef ENUMIFY
};

absl::StatusOr<Builtin> BuiltinFromString(std::string_view name);

std::string BuiltinToString(Builtin builtin);

static const Builtin kAllBuiltins[] = {
#define ELEMIFY(__str, __enum, ...) Builtin::__enum,
    XLS_DSLX_BUILTIN_EACH(ELEMIFY)
#undef ELEMIFY
};

// Tags a value to denote its payload.
//
// Note this goes beyond InterpValue::Payload annotating things like whether the
// bits should be interpreted as signed or unsigned, which can change the
// behavior of interpreted operators like '<'.
enum class InterpValueTag {
  kUBits,
  kSBits,
  kTuple,
  kArray,
  kEnum,
  kFunction,
  kToken,
  kChannel,
};

std::string TagToString(InterpValueTag tag);

inline std::ostream& operator<<(std::ostream& os, InterpValueTag tag) {
  os << TagToString(tag);
  return os;
}

struct TokenData {
  // Currently empty.
};

// A DSLX interpreter value (variant), with InterpValueTag as a discriminator.
class InterpValue {
 public:
  struct UserFnData {
    Module* module;
    Function* function;
  };
  using FnData = std::variant<Builtin, UserFnData>;
  using Channel = std::deque<InterpValue>;

  // Factories

  // TODO(leary): 2020-10-18 Port to be consistent with xls/ir/bits.h
  static InterpValue MakeUBits(int64_t bit_count, int64_t value);
  static InterpValue MakeSBits(int64_t bit_count, int64_t value);

  static InterpValue MakeZeroValue(bool is_signed, int64_t bit_count);
  static InterpValue MakeMaxValue(bool is_signed, int64_t bit_count);

  static InterpValue MakeUnit() { return MakeTuple({}); }
  static InterpValue MakeU8(uint8_t value) {
    return MakeUBits(/*bit_count=*/8, value);
  }
  static InterpValue MakeS32(uint32_t value) {
    return MakeSBits(/*bit_count=*/32, value);
  }
  static InterpValue MakeU32(uint32_t value) {
    return MakeUBits(/*bit_count=*/32, value);
  }
  static InterpValue MakeS64(uint64_t value) {
    return MakeSBits(/*bit_count=*/64, value);
  }
  static InterpValue MakeU64(uint64_t value) {
    return MakeUBits(/*bit_count=*/64, value);
  }
  static InterpValue MakeEnum(Bits bits, bool is_signed, const EnumDef* def) {
    return InterpValue(InterpValueTag::kEnum,
                       EnumData{def, is_signed, std::move(bits)});
  }
  static InterpValue MakeSigned(Bits bits) {
    return InterpValue(InterpValueTag::kSBits, std::move(bits));
  }
  static InterpValue MakeUnsigned(Bits bits) {
    return InterpValue(InterpValueTag::kUBits, std::move(bits));
  }
  static InterpValue MakeBits(bool is_signed, Bits bits) {
    return InterpValue(
        is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits,
        std::move(bits));
  }
  static InterpValue MakeTuple(std::vector<InterpValue> members);
  static InterpValue MakeChannel() {
    return InterpValue(InterpValueTag::kChannel, std::make_shared<Channel>());
  }
  static absl::StatusOr<InterpValue> MakeArray(
      std::vector<InterpValue> elements);
  static InterpValue MakeBool(bool value) {
    return MakeUBits(1, value ? 1 : 0);
  }

  static absl::StatusOr<InterpValue> MakeBits(InterpValueTag tag, Bits bits) {
    if (tag != InterpValueTag::kUBits && tag != InterpValueTag::kSBits) {
      return absl::InvalidArgumentError(
          "Bits tag is required to make a bits Value; got: " +
          TagToString(tag));
    }
    return InterpValue(tag, std::move(bits));
  }

  static InterpValue MakeFunction(Builtin b) {
    return InterpValue(InterpValueTag::kFunction, b);
  }
  static InterpValue MakeFunction(FnData fn_data) {
    return InterpValue(InterpValueTag::kFunction, std::move(fn_data));
  }

  static InterpValue MakeToken() {
    return InterpValue(InterpValueTag::kToken, std::make_shared<TokenData>());
  }

  // Queries

  bool IsTuple() const { return tag_ == InterpValueTag::kTuple; }
  bool IsArray() const { return tag_ == InterpValueTag::kArray; }
  bool IsUnit() const { return IsTuple() && GetLength().value() == 0; }
  bool IsUBits() const { return tag_ == InterpValueTag::kUBits; }
  bool IsSBits() const { return tag_ == InterpValueTag::kSBits; }
  bool IsBits() const { return IsUBits() || IsSBits(); }
  bool IsBool() const { return IsBits() && GetBitCount().value() == 1; }
  bool IsEnum() const { return tag_ == InterpValueTag::kEnum; }
  bool IsFunction() const { return tag_ == InterpValueTag::kFunction; }
  bool IsBuiltinFunction() const {
    return IsFunction() && std::holds_alternative<Builtin>(GetFunctionOrDie());
  }
  bool IsChannel() const { return tag_ == InterpValueTag::kChannel; }

  bool IsTraceBuiltin() const {
    return IsBuiltinFunction() &&
           std::get<Builtin>(GetFunctionOrDie()) == Builtin::kTrace;
  }

  bool IsFalse() const { return IsBool() && GetBitsOrDie().IsZero(); }
  bool IsTrue() const { return IsBool() && GetBitsOrDie().IsAllOnes(); }

  bool IsNegative() const {
    return IsSigned() && (GetBitsOrDie().GetFromMsb(0));
  }

  bool IsSigned() const {
    CHECK(IsBits() || IsEnum());
    if (IsEnum()) {
      EnumData enum_data = std::get<EnumData>(payload_);
      return enum_data.is_signed;
    }

    return IsSBits();
  }

  // Note that this equality for bit value holding tags ignores the tag and just
  // checks bit pattern equivalence. We expect that the type checker will
  // validate the comparisons are done properly between types before the
  // interpreter performs these operations.
  bool Eq(const InterpValue& other) const;
  bool Ne(const InterpValue& other) const { return !Eq(other); }

  // Various operations -- tag is not applicable for the operation, a status
  // error should be returned.

  // Gets the underlying bit value and inspects the type(-tag)'s signedness to
  // determine whether it should sign-extend or zero-extend to produce the
  // resulting int64_t.
  absl::StatusOr<int64_t> GetBitValueViaSign() const;

  absl::StatusOr<int64_t> GetBitCount() const;

  // Gets the underlying bit value with zero-extension to produce a uint64_t.
  absl::StatusOr<uint64_t> GetBitValueUnsigned() const;

  // Gets the underlying bit value with sign-extension to produce a int64_t.
  absl::StatusOr<int64_t> GetBitValueSigned() const;

  // Returns true iff the value HasBits and the bits values fits in a
  // (u)int64_t.
  bool FitsInUint64() const;
  bool FitsInInt64() const;

  // Returns true if the value HasBits and the (unsigned/signed) value fits in
  // 'n' bits.
  bool FitsInNBitsUnsigned(int64_t n) const;
  bool FitsInNBitsSigned(int64_t n) const;

  absl::StatusOr<int64_t> GetLength() const {
    if (IsTuple() || IsArray()) {
      return GetValuesOrDie().size();
    }
    return absl::InvalidArgumentError("Invalid tag for length query: " +
                                      TagToString(tag_));
  }

  absl::StatusOr<InterpValue> ZeroExt(int64_t new_bit_count) const;
  absl::StatusOr<InterpValue> SignExt(int64_t new_bit_count) const;
  absl::StatusOr<InterpValue> Concat(const InterpValue& other) const;

  // Performs an add of two uN[N]s and returns a 2-tuple of:
  //
  //  `(carry: bool, sum: uN[N])`
  absl::StatusOr<InterpValue> AddWithCarry(const InterpValue& other) const;

  absl::StatusOr<InterpValue> Add(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Sub(const InterpValue& other) const;

  absl::StatusOr<InterpValue> Mul(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Shl(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Shrl(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Shra(const InterpValue& other) const;
  absl::StatusOr<InterpValue> BitwiseXor(const InterpValue& other) const;
  absl::StatusOr<InterpValue> BitwiseNegate() const;
  absl::StatusOr<InterpValue> BitwiseOr(const InterpValue& other) const;
  absl::StatusOr<InterpValue> BitwiseAnd(const InterpValue& other) const;
  absl::StatusOr<InterpValue> ArithmeticNegate() const;
  absl::StatusOr<InterpValue> CeilOfLog2() const;
  absl::StatusOr<InterpValue> FloorDiv(const InterpValue& other) const;
  absl::StatusOr<InterpValue> FloorMod(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Index(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Index(int64_t index) const;
  absl::StatusOr<InterpValue> Update(const InterpValue& index,
                                     const InterpValue& value) const;
  absl::StatusOr<InterpValue> Slice(const InterpValue& start,
                                    const InterpValue& length) const;
  absl::StatusOr<InterpValue> Flatten() const;
  absl::StatusOr<InterpValue> Decode(int64_t new_bit_count) const;
  absl::StatusOr<InterpValue> Encode() const;
  absl::StatusOr<InterpValue> OneHot(bool lsb_prio) const;

  absl::StatusOr<InterpValue> Lt(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Le(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Gt(const InterpValue& other) const;
  absl::StatusOr<InterpValue> Ge(const InterpValue& other) const;

  // Performs the signed comparison defined by "method".
  //
  // "method" should be a value in the set {slt, sle, sgt, sge} or an
  // InvalidArgumentError is returned.
  absl::StatusOr<InterpValue> SCmp(const InterpValue& other,
                                   std::string_view method);

  // Converts this value into a string for display.
  //
  // Args:
  //  humanize: Whether to maximize readability of the value for human
  //    consumption -- e.g. indicates whether bits-based values should have
  //    leading types. When binary formatting is requested, the types are always
  //    given, however (because it's hard to intuit leading zeros without a
  //    displayed width).
  //  format: What radix to use for converting the value to string.
  std::string ToString(
      bool humanize = false,
      FormatPreference format = FormatPreference::kDefault) const;
  std::string ToHumanString() const { return ToString(/*humanize=*/true); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const InterpValue& v) {
    absl::Format(&sink, "%s", v.ToString());
  }

  absl::StatusOr<std::string> ToFormattedString(
      const ValueFormatDescriptor& fmt_desc, int64_t indentation = 0) const;

  InterpValueTag tag() const { return tag_; }

  absl::StatusOr<const std::vector<InterpValue>*> GetValues() const {
    if (!std::holds_alternative<std::vector<InterpValue>>(payload_)) {
      return absl::InvalidArgumentError("Value does not hold element values");
    }
    return &std::get<std::vector<InterpValue>>(payload_);
  }
  const std::vector<InterpValue>& GetValuesOrDie() const {
    return std::get<std::vector<InterpValue>>(payload_);
  }
  absl::StatusOr<const FnData*> GetFunction() const {
    if (!std::holds_alternative<FnData>(payload_)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Value does not hold function data: ", ToString(), "."));
    }
    return &std::get<FnData>(payload_);
  }
  const FnData& GetFunctionOrDie() const { return *GetFunction().value(); }
  absl::StatusOr<Bits> GetBits() const;
  const Bits& GetBitsOrDie() const;
  absl::StatusOr<std::shared_ptr<Channel>> GetChannel() const;
  std::shared_ptr<Channel> GetChannelOrDie() const {
    return std::get<std::shared_ptr<Channel>>(payload_);
  }

  // For enum values, returns the enum that the bit pattern is interpreted by is
  // referred to by the interpreter value, along with type metadata.
  struct EnumData {
    const EnumDef* def;
    bool is_signed;
    Bits value;
  };

  std::optional<EnumData> GetEnumData() const {
    if (IsEnum()) {
      return std::get<EnumData>(payload_);
    }

    return std::nullopt;
  }

  // Note: different from IsBits() which is checking whether the tag is sbits or
  // ubits; this is checking whether there are bits in the payload, which would
  // apply to enum values as well.
  bool HasBits() const {
    return std::holds_alternative<Bits>(payload_) ||
           std::holds_alternative<EnumData>(payload_);
  }

  bool HasValues() const {
    return std::holds_alternative<std::vector<InterpValue>>(payload_);
  }

  bool IsToken() const { return tag_ == InterpValueTag::kToken; }
  const std::shared_ptr<TokenData>& GetTokenData() const {
    return std::get<std::shared_ptr<TokenData>>(payload_);
  }

  // Convert any non-function InterpValue to an IR Value.
  // Note: Enum conversions are lossy because enum types don't exist in the IR.
  absl::StatusOr<xls::Value> ConvertToIr() const;

  // Convert many non-function InterpValues to IR Values.
  // Note: Enum conversions are lossy because enum types don't exist in the IR.
  static absl::StatusOr<std::vector<xls::Value>> ConvertValuesToIr(
      absl::Span<InterpValue const> values);

  // Convenience wrappers around Eq()/Ne() so InterpValues can act as C++ value
  // types; e.g. in testing assertions and such.
  bool operator==(const InterpValue& rhs) const;
  bool operator!=(const InterpValue& rhs) const { return !(*this == rhs); }

  // Lt() only performs comparisons on bits-valued InterpValues, whereas this
  // compares across Bits-, array-, and tuple-valued objects. For this set, the
  // ordering Bits < arrays < tuples has been arbitrarily defined.
  bool operator<(const InterpValue& rhs) const;
  bool operator>=(const InterpValue& rhs) const;

 private:
  friend struct InterpValuePickler;

  // Formats this tuple value using the given struct format description.
  //
  // Returns an error status if the struct descriptor does not correspond to the
  // tuple structure appropriately.
  //
  // Precondition: IsTuple()
  absl::StatusOr<std::string> ToStructString(
      const StructFormatDescriptor& fmt_desc, int64_t indentation) const;

  // As above but for tuple values (that are not participating in a struct
  // type).
  absl::StatusOr<std::string> ToTupleString(
      const TupleFormatDescriptor& fmt_desc, int64_t indentation) const;

  // As above but for array values.
  absl::StatusOr<std::string> ToArrayString(
      const ArrayFormatDescriptor& fmt_desc, int64_t indentation) const;

  // As above but for enum values.
  absl::StatusOr<std::string> ToEnumString(
      const EnumFormatDescriptor& fmt_desc) const;

  // Note: currently InterpValues are not scoped to a lifetime, so we use a
  // shared_ptr for referring to token data for identity purposes.
  //
  // TODO(leary): 2020-02-10 When all Python bindings are eliminated we can more
  // easily make an interpreter scoped lifetime that InterpValues can live in.
  using Payload =
      std::variant<Bits, EnumData, std::vector<InterpValue>, FnData,
                   std::shared_ptr<TokenData>, std::shared_ptr<Channel>>;

  InterpValue(InterpValueTag tag, Payload payload)
      : tag_(tag), payload_(std::move(payload)) {}

  using CompareF = bool (*)(const Bits& lhs, const Bits& rhs);

  // Helper for various comparisons.
  static absl::StatusOr<InterpValue> Compare(const InterpValue& lhs,
                                             const InterpValue& rhs,
                                             CompareF ucmp, CompareF scmp);

  InterpValueTag tag_;
  Payload payload_;
};

// Retrieves the module associated with the function_value if it is user
// defined.
//
// Check-fails if function_value is not a function-typed value.
std::optional<Module*> GetFunctionValueOwner(
    const InterpValue& function_value);

template <typename H>
H AbslHashValue(H state, const InterpValue::UserFnData& v) {
  return H::combine(std::move(state), v.module, v.function);
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_INTERP_VALUE_H_
