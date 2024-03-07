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

#include "xls/dslx/type_system/concrete_type_zero_value.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {
namespace {

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

class MakeZeroVisitor : public TypeVisitor {
 public:
  MakeZeroVisitor(const ImportData& import_data, const Span& span)
      : import_data_(import_data), span_(span) {}

  absl::Status HandleEnum(const EnumType& t) override {
    XLS_ASSIGN_OR_RETURN(bool has_known_zero,
                         HasKnownZeroValue(t, import_data_));
    if (!has_known_zero) {
      return TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Enum type '%s' does not have a known zero value.",
                          t.nominal_type().identifier()));
    }

    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    result_ =
        InterpValue::MakeEnum(Bits(size), t.is_signed(), &t.nominal_type());
    return absl::OkStatus();
  }

  absl::Status HandleBits(const BitsType& t) override {
    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    result_ = InterpValue::MakeBits(t.is_signed(), Bits(size));
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const FunctionType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, "Cannot make a zero-value of function type.");
  }
  absl::Status HandleChannel(const ChannelType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, "Cannot make a zero-value of channel type.");
  }
  absl::Status HandleToken(const TokenType& t) override {
    return TypeInferenceErrorStatus(span_, &t,
                                    "Cannot make a zero-value of token type.");
  }
  absl::Status HandleStruct(const StructType& t) override {
    std::vector<InterpValue> elems;
    for (const auto& member : t.members()) {
      XLS_ASSIGN_OR_RETURN(InterpValue z,
                           MakeZeroValue(*member, import_data_, span_));
      elems.push_back(std::move(z));
    }
    result_ = InterpValue::MakeTuple(std::move(elems));
    return absl::OkStatus();
  }
  absl::Status HandleTuple(const TupleType& t) override {
    std::vector<InterpValue> elems;
    for (const auto& m : t.members()) {
      XLS_ASSIGN_OR_RETURN(InterpValue zero,
                           MakeZeroValue(*m, import_data_, span_));
      elems.push_back(std::move(zero));
    }
    result_ = InterpValue::MakeTuple(std::move(elems));
    return absl::OkStatus();
  }

  absl::Status HandleArray(const ArrayType& t) override {
    XLS_ASSIGN_OR_RETURN(InterpValue elem_value,
                         MakeZeroValue(t.element_type(), import_data_, span_));
    XLS_ASSIGN_OR_RETURN(int64_t size, t.size().GetAsInt64());
    XLS_ASSIGN_OR_RETURN(
        result_,
        InterpValue::MakeArray(std::vector<InterpValue>(size, elem_value)));
    return absl::OkStatus();
  }
  absl::Status HandleMeta(const MetaType& t) override {
    return TypeInferenceErrorStatus(span_, &t,
                                    "Cannot make a zero-value of a meta-type.");
  }

  const std::optional<InterpValue>& result() const { return result_; }

 private:
  const ImportData& import_data_;
  const Span& span_;
  std::optional<InterpValue> result_;
};

}  // namespace

absl::StatusOr<InterpValue> MakeZeroValue(const Type& type,
                                          const ImportData& import_data,
                                          const Span& span) {
  XLS_VLOG(5) << "MakeZeroValue; type: " << type << " @ " << span;
  MakeZeroVisitor v(import_data, span);
  XLS_RETURN_IF_ERROR(type.Accept(v));
  XLS_RET_CHECK(v.result().has_value());
  return v.result().value();
}

}  // namespace xls::dslx
