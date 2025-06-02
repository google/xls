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

#include "xls/dslx/type_system/deduce_enum_def.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"

namespace xls::dslx {

// When enums have no type annotation explicitly placed on them we infer the
// width of the enum from the values contained inside of its definition.
static absl::StatusOr<std::unique_ptr<Type>> DeduceEnumSansUnderlyingType(
    const EnumDef* node, DeduceCtx* ctx) {
  VLOG(5) << "Deducing enum without underlying type: " << node->ToString();
  std::vector<std::pair<const EnumMember*, std::unique_ptr<Type>>> deduced;
  for (const EnumMember& member : node->values()) {
    bool is_boolean = false;
    if (IsBareNumber(member.value, &is_boolean) != nullptr && !is_boolean) {
      continue;  // We'll validate these below.
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> t, ctx->Deduce(member.value));
    deduced.emplace_back(&member, std::move(t));
  }
  if (deduced.empty()) {
    return TypeInferenceErrorStatus(
        node->span(), nullptr, "Could not deduce underlying type for enum.",
        ctx->file_table());
  }
  const Type& target = *deduced.front().second;
  for (int64_t i = 1; i < deduced.size(); ++i) {
    const Type& got = *deduced.at(i).second;
    if (target != got) {
      return ctx->TypeMismatchError(
          deduced.at(i).first->GetSpan(), nullptr, target, nullptr, got,
          "Inconsistent member types in enum definition.");
    }
  }

  VLOG(5) << "Underlying type of EnumDef " << node->identifier() << ": "
          << target;

  // Note the deduced type for all the "bare number" members.
  for (const EnumMember& member : node->values()) {
    if (const Number* number = IsBareNumber(member.value)) {
      XLS_RETURN_IF_ERROR(ValidateNumber(*number, target));
      ctx->type_info()->SetItem(number, target);
    }
  }

  return std::move(deduced.front().second);
}

absl::StatusOr<std::unique_ptr<Type>> DeduceEnumDef(const EnumDef* node,
                                                    DeduceCtx* ctx) {
  std::unique_ptr<Type> underlying_type;
  if (node->type_annotation() == nullptr) {
    XLS_ASSIGN_OR_RETURN(underlying_type,
                         DeduceEnumSansUnderlyingType(node, ctx));
  } else {
    XLS_ASSIGN_OR_RETURN(underlying_type,
                         ctx->DeduceAndResolve(node->type_annotation()));
    XLS_ASSIGN_OR_RETURN(
        underlying_type,
        UnwrapMetaType(std::move(underlying_type),
                       node->type_annotation()->span(), "enum underlying type",
                       ctx->file_table()));
  }

  std::optional<BitsLikeProperties> bits_like = GetBitsLike(*underlying_type);
  if (!bits_like.has_value()) {
    return TypeInferenceErrorStatus(node->span(), underlying_type.get(),
                                    "Underlying type for an enum "
                                    "must be a bits type.",
                                    ctx->file_table());
  }

  auto matches_underlying = [&](const Type& t) -> bool {
    std::optional<BitsLikeProperties> t_bits_like = GetBitsLike(t);
    return t_bits_like.has_value() && *t_bits_like == *bits_like;
  };

  // Grab the bit count of the Enum's underlying type.
  const TypeDim& bit_count = bits_like->size;
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());

  std::vector<InterpValue> members;
  members.reserve(node->values().size());
  for (const EnumMember& member : node->values()) {
    if (const Number* number = dynamic_cast<const Number*>(member.value);
        number != nullptr && number->type_annotation() == nullptr) {
      XLS_RETURN_IF_ERROR(ValidateNumber(*number, *underlying_type));
      ctx->type_info()->SetItem(number, *underlying_type);
      XLS_RETURN_IF_ERROR(ConstexprEvaluator::Evaluate(
          ctx->import_data(), ctx->type_info(), ctx->warnings(),
          ctx->GetCurrentParametricEnv(), number, underlying_type.get()));
    } else {
      // Some other constexpr expression that should have the same type as the
      // underlying bits type for this enum.
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> member_value_type,
                           ctx->Deduce(member.value));
      if (!matches_underlying(*member_value_type)) {
        return ctx->TypeMismatchError(
            member.value->span(), nullptr, *member_value_type, nullptr,
            *underlying_type,
            "Enum-member type did not match the enum's underlying type.");
      }
    }

    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        ConstexprEvaluator::EvaluateToValue(
            ctx->import_data(), ctx->type_info(), ctx->warnings(),
            ctx->GetCurrentParametricEnv(), member.value));

    // Right now we may have the underlying type as the noted constexpr value
    // (e.g. if we evaluated a number that was given as an enum value
    // expression), but we want to wrap that up in the enum type appropriately.
    XLS_RET_CHECK((value.IsEnum() && value.GetEnumData().value().def == node) ||
                  value.IsBits());
    if (value.IsBits()) {
      value = InterpValue::MakeEnum(value.GetBitsOrDie(), is_signed, node);
      ctx->type_info()->NoteConstExpr(member.name_def, value);
      ctx->type_info()->NoteConstExpr(member.value, value);
    }

    members.push_back(value);
  }

  auto enum_type =
      std::make_unique<EnumType>(*node, bit_count, is_signed, members);

  for (const EnumMember& member : node->values()) {
    ctx->type_info()->SetItem(member.name_def, *enum_type);
  }

  auto meta_type = std::make_unique<MetaType>(std::move(enum_type));
  ctx->type_info()->SetItem(node->name_def(), *meta_type);
  ctx->type_info()->SetItem(node, *meta_type);
  return meta_type;
}

}  // namespace xls::dslx
