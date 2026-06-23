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

#include "xls/dslx/sum_type_encoding.h"

#include <cstdint>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

Phase1SumTypeEncoding::Phase1SumTypeEncoding(const SumType& type)
    : type_(type) {
  int64_t payload_start = 0;
  variants_.reserve(type_.variants().size());
  for (int64_t variant_index = 0; variant_index < type_.variants().size();
       ++variant_index) {
    const SumTypeVariant& variant = type_.variants().at(variant_index);
    variants_.push_back(VariantInfo(variant_index, variant, payload_start));
    for (int64_t member_index = 0; member_index < variant.size();
         ++member_index) {
      payload_slot_types_.push_back(&variant.GetMemberType(member_index));
    }
    payload_start += variant.size();
  }
}

absl::StatusOr<int64_t> Phase1SumTypeEncoding::tag_bit_count() const {
  return type_.tag_bit_count().GetAsInt64();
}

absl::StatusOr<Phase1SumTypeEncoding::VariantInfo>
Phase1SumTypeEncoding::GetVariant(std::string_view variant_name) const {
  XLS_ASSIGN_OR_RETURN(const VariantInfo* variant, FindVariant(variant_name));
  return *variant;
}

absl::Status Phase1SumTypeEncoding::ForEachVariant(
    absl::FunctionRef<absl::Status(const VariantInfo& variant)> visitor) const {
  for (const VariantInfo& variant : variants_) {
    XLS_RETURN_IF_ERROR(visitor(variant));
  }
  return absl::OkStatus();
}

absl::Status Phase1SumTypeEncoding::ForEachPayloadType(
    absl::FunctionRef<absl::Status(const Type& type)> visitor) const {
  for (const Type* payload_type : payload_slot_types_) {
    XLS_RETURN_IF_ERROR(visitor(*payload_type));
  }
  return absl::OkStatus();
}

absl::Status Phase1SumTypeEncoding::VisitPayloadAssemblyOrder(
    const VariantInfo& variant,
    absl::FunctionRef<absl::Status(int64_t active_index)> active_visitor,
    absl::FunctionRef<absl::Status(const Type& inactive_type)> inactive_visitor)
    const {
  XLS_RETURN_IF_ERROR(ValidateVariantInfo(variant));

  int64_t active_index = 0;
  for (int64_t slot_index = 0; slot_index < payload_slot_types_.size();
       ++slot_index) {
    const bool is_active = slot_index >= variant.payload_start &&
                           slot_index < variant.payload_end();
    XLS_RETURN_IF_ERROR(
        is_active ? active_visitor(active_index++)
                  : inactive_visitor(*payload_slot_types_.at(slot_index)));
  }
  return absl::OkStatus();
}

absl::Status Phase1SumTypeEncoding::ForEachActivePayloadSlot(
    const VariantInfo& variant,
    absl::FunctionRef<absl::Status(int64_t slot_index, int64_t active_index,
                                   const Type& type)>
        visitor) const {
  XLS_RETURN_IF_ERROR(ValidateVariantInfo(variant));

  int64_t active_index = 0;
  for (int64_t slot_index = variant.payload_start;
       slot_index < variant.payload_end(); ++slot_index) {
    XLS_RETURN_IF_ERROR(visitor(slot_index, active_index++,
                                *payload_slot_types_.at(slot_index)));
  }
  return absl::OkStatus();
}

absl::Status Phase1SumTypeEncoding::ValidateVariantInfo(
    const VariantInfo& variant) const {
  const int64_t variant_count = static_cast<int64_t>(variants_.size());
  if (variant.variant_index < 0 || variant.variant_index >= variant_count) {
    return absl::OutOfRangeError(absl::StrCat(
        "Variant index ", variant.variant_index, " is out of range for sum `",
        type_.nominal_type().identifier(), "` with ", variant_count,
        " variants."));
  }

  const VariantInfo& stored_variant = variants_.at(variant.variant_index);
  if (stored_variant.variant != variant.variant) {
    return absl::InvalidArgumentError(absl::StrCat(
        "VariantInfo for `", variant.variant->variant().identifier(),
        "` does not belong to sum `", type_.nominal_type().identifier(),
        "` at variant index ", variant.variant_index, "."));
  }
  return absl::OkStatus();
}

absl::StatusOr<const Phase1SumTypeEncoding::VariantInfo*>
Phase1SumTypeEncoding::FindVariant(std::string_view variant_name) const {
  for (const VariantInfo& variant : variants_) {
    if (variant.variant->variant().identifier() == variant_name) {
      return &variant;
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No variant `", variant_name, "` in sum `",
                   type_.nominal_type().identifier(), "`."));
}

}  // namespace xls::dslx
