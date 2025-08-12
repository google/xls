// Copyright 2023 The XLS Authors
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

#include "xls/dslx/type_system/type_zero_value.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

inline constexpr std::string_view kZeroValueName = "zero-value";
inline constexpr std::string_view kAllOnesValueName = "all-ones-value";

// Returns whether enum type t has a member that is definitively zero-valued.
absl::StatusOr<bool> HasKnownZeroValue(const EnumType& t,
                                       const ImportData& import_data) {
  const EnumDef& def = t.nominal_type();
  XLS_ASSIGN_OR_RETURN(const TypeInfo* type_info,
                       import_data.GetRootTypeInfoForNode(&def));

  for (const EnumMember& member : def.values()) {
    XLS_ASSIGN_OR_RETURN(InterpValue v, type_info->GetConstExpr(member.value));
    if (v.GetBitsOrDie().IsZero()) {
      return true;
    }
  }

  return false;
}

// Returns whether enum type t has a member that is definitively
// all-ones-valued.
absl::StatusOr<bool> HasKnownAllOnesValue(const EnumType& t,
                                          const ImportData& import_data) {
  const EnumDef& def = t.nominal_type();
  XLS_ASSIGN_OR_RETURN(const TypeInfo* type_info,
                       import_data.GetRootTypeInfoForNode(&def));

  for (const EnumMember& member : def.values()) {
    XLS_ASSIGN_OR_RETURN(InterpValue v, type_info->GetConstExpr(member.value));
    if (v.GetBitsOrDie().IsAllOnes()) {
      return true;
    }
  }

  return false;
}

absl::StatusOr<InterpValue> ZeroOfBitsLike(const BitsLikeProperties& bits_like,
                                           const ImportData& import_data,
                                           const Span& span) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like.is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(int64_t size, bits_like.size.GetAsInt64());
  return InterpValue::MakeBits(is_signed, Bits(size));
}

absl::StatusOr<InterpValue> ZeroOfLeafType(const Type& type,
                                           const ImportData& import_data,
                                           const Span& span) {
  if (const auto* enum_type = dynamic_cast<const EnumType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(bool has_known_zero,
                         HasKnownZeroValue(*enum_type, import_data));
    if (!has_known_zero) {
      return TypeInferenceErrorStatus(
          span, enum_type,
          absl::StrFormat("Enum type '%s' does not have a known zero value.",
                          enum_type->nominal_type().identifier()),
          import_data.file_table());
    }

    XLS_ASSIGN_OR_RETURN(int64_t size, enum_type->size().GetAsInt64());
    return InterpValue::MakeEnum(Bits(size), enum_type->is_signed(),
                                 &enum_type->nominal_type());
  }
  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      bits_like.has_value()) {
    return ZeroOfBitsLike(bits_like.value(), import_data, span);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Type '%s' is not a leaf type.", type.ToString()));
}

absl::StatusOr<InterpValue> AllOnesOfBitsLike(
    const BitsLikeProperties& bits_like, const ImportData& import_data,
    const Span& span) {
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like.is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(int64_t size, bits_like.size.GetAsInt64());
  return InterpValue::MakeBits(is_signed, Bits::AllOnes(size));
}

absl::StatusOr<InterpValue> AllOnesOfLeafType(const Type& type,
                                              const ImportData& import_data,
                                              const Span& span) {
  if (const auto* enum_type = dynamic_cast<const EnumType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(bool has_known_all_ones,
                         HasKnownAllOnesValue(*enum_type, import_data));
    if (!has_known_all_ones) {
      return TypeInferenceErrorStatus(
          span, enum_type,
          absl::StrFormat(
              "Enum type '%s' does not have a known all-ones value.",
              enum_type->nominal_type().identifier()),
          import_data.file_table());
    }
    XLS_ASSIGN_OR_RETURN(int64_t size, enum_type->size().GetAsInt64());
    return InterpValue::MakeEnum(Bits::AllOnes(size), enum_type->is_signed(),
                                 &enum_type->nominal_type());
  }
  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      bits_like.has_value()) {
    return AllOnesOfBitsLike(bits_like.value(), import_data, span);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Type '%s' is not a leaf type.", type.ToString()));
}

class MakeValueVisitor : public TypeVisitor {
  using LeafTypeFn = std::function<absl::StatusOr<InterpValue>(
      const Type&, const ImportData&, const Span&)>;
  using LeafBitsLikeFn = std::function<absl::StatusOr<InterpValue>(
      const BitsLikeProperties&, const ImportData&, const Span&)>;

 public:
  MakeValueVisitor(LeafTypeFn&& leaf_type_fn,
                   LeafBitsLikeFn&& leaf_bits_like_fn,
                   const ImportData& import_data, const Span& span,
                   std::string_view value_name)
      : leaf_type_fn_(std::move(leaf_type_fn)),
        leaf_bits_like_fn_(std::move(leaf_bits_like_fn)),
        import_data_(import_data),
        span_(span),
        value_name_(value_name) {}

  absl::Status HandleEnum(const EnumType& t) final {
    XLS_ASSIGN_OR_RETURN(result_, leaf_type_fn_(t, import_data_, span_));
    return absl::OkStatus();
  }

  absl::Status HandleBits(const BitsType& t) final {
    XLS_ASSIGN_OR_RETURN(result_, leaf_type_fn_(t, import_data_, span_));
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const FunctionType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of function type.", value_name_),
        file_table());
  }
  absl::Status HandleChannel(const ChannelType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of channel type.", value_name_),
        file_table());
  }
  absl::Status HandleToken(const TokenType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of token type.", value_name_),
        file_table());
  }
  absl::Status HandleBitsConstructor(const BitsConstructorType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of bits-constructor type.",
                        value_name_),
        file_table());
  }
  absl::Status HandleStruct(const StructType& t) final {
    std::vector<InterpValue> elems;
    for (const auto& member : t.members()) {
      XLS_RETURN_IF_ERROR(member->Accept(*this));
      XLS_ASSIGN_OR_RETURN(InterpValue elem_value, ResultOrError());
      elems.push_back(std::move(elem_value));
    }
    result_ = InterpValue::MakeTuple(std::move(elems));
    return absl::OkStatus();
  }
  absl::Status HandleProc(const ProcType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of proc type.", value_name_),
        file_table());
  }

  absl::Status HandleTuple(const TupleType& t) final {
    std::vector<InterpValue> elems;
    for (const auto& m : t.members()) {
      XLS_RETURN_IF_ERROR(m->Accept(*this));
      XLS_ASSIGN_OR_RETURN(InterpValue elem_value, ResultOrError());
      elems.push_back(std::move(elem_value));
    }
    result_ = InterpValue::MakeTuple(std::move(elems));
    return absl::OkStatus();
  }

  absl::Status HandleArray(const ArrayType& t) final {
    std::optional<BitsLikeProperties> bits_like = GetBitsLike(t);
    if (bits_like.has_value()) {
      return HandleBitsLike(bits_like.value());
    }

    XLS_RETURN_IF_ERROR(t.element_type().Accept(*this));
    XLS_ASSIGN_OR_RETURN(InterpValue elem_value, ResultOrError());
    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    XLS_ASSIGN_OR_RETURN(
        result_,
        InterpValue::MakeArray(std::vector<InterpValue>(size, elem_value)));
    return absl::OkStatus();
  }

  absl::Status HandleMeta(const MetaType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of a meta-type.", value_name_),
        file_table());
  }

  absl::Status HandleModule(const ModuleType& t) final {
    return TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Cannot make a %s of a module type.", value_name_),
        file_table());
  }

  absl::StatusOr<InterpValue> ResultOrError() const {
    XLS_RET_CHECK(result_.has_value());
    return *result_;
  }

  const FileTable& file_table() const { return import_data_.file_table(); }

 private:
  absl::Status HandleBitsLike(const BitsLikeProperties& bits_like) {
    // Make a BitsType with the same properties.
    XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like.is_signed.GetAsBool());
    std::unique_ptr<BitsType> bits_type =
        std::make_unique<BitsType>(is_signed, bits_like.size);
    XLS_ASSIGN_OR_RETURN(result_,
                         leaf_bits_like_fn_(bits_like, import_data_, span_));
    return absl::OkStatus();
  }

  LeafTypeFn leaf_type_fn_;
  LeafBitsLikeFn leaf_bits_like_fn_;
  const ImportData& import_data_;
  const Span& span_;
  std::string_view value_name_;
  std::optional<InterpValue> result_;
};

}  // namespace

absl::StatusOr<InterpValue> MakeZeroValue(const Type& type,
                                          const ImportData& import_data,
                                          const Span& span) {
  VLOG(5) << "MakeZeroValue; type: " << type;
  MakeValueVisitor v(ZeroOfLeafType, ZeroOfBitsLike, import_data, span,
                     kZeroValueName);
  XLS_RETURN_IF_ERROR(type.Accept(v));
  return v.ResultOrError();
}

absl::StatusOr<InterpValue> MakeAllOnesValue(const Type& type,
                                             const ImportData& import_data,
                                             const Span& span) {
  VLOG(5) << "MakeAllOnesValue; type: " << type;
  MakeValueVisitor v(AllOnesOfLeafType, AllOnesOfBitsLike, import_data, span,
                     kAllOnesValueName);
  XLS_RETURN_IF_ERROR(type.Accept(v));
  return v.ResultOrError();
}

}  // namespace xls::dslx
