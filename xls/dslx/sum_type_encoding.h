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

#ifndef XLS_DSLX_SUM_TYPE_ENCODING_H_
#define XLS_DSLX_SUM_TYPE_ENCODING_H_

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// View of the current semantic-sum storage encoding.
//
// Interpreter value construction and validation share the `(tag,
// payload_slots)` layout exposed here. The storage tag is a dense
// declaration-order index, not a source-level discriminant. Payload slots for
// every variant remain present, with inactive slots holding canonical
// placeholders. The storage layout is an internal implementation detail rather
// than a stable source-language or serialization contract.
class Phase1SumTypeEncoding {
 public:
  struct VariantInfo {
    const int64_t variant_index;
    const SumTypeVariant* const variant;
    // First payload slot for this variant in the flattened payload tuple.
    const int64_t payload_start;

    int64_t payload_size() const { return variant->size(); }
    int64_t payload_end() const { return payload_start + payload_size(); }

   private:
    friend class Phase1SumTypeEncoding;

    VariantInfo(int64_t variant_index, const SumTypeVariant& variant,
                int64_t payload_start)
        : variant_index(variant_index),
          variant(&variant),
          payload_start(payload_start) {}
  };

  explicit Phase1SumTypeEncoding(const SumType& type);

  int64_t payload_slot_count() const { return payload_slot_types_.size(); }
  absl::StatusOr<int64_t> tag_bit_count() const;

  absl::StatusOr<VariantInfo> GetVariant(std::string_view variant_name) const;
  absl::Status ForEachVariant(
      absl::FunctionRef<absl::Status(const VariantInfo& variant)> visitor)
      const;
  // Visits stored payload slot types in canonical storage order.
  absl::Status ForEachPayloadType(
      absl::FunctionRef<absl::Status(const Type& type)> visitor) const;
  // Visits only the active payload members for one variant, providing the
  // canonical storage slot index and the payload index within the variant.
  absl::Status ForEachActivePayloadSlot(
      const VariantInfo& variant,
      absl::FunctionRef<absl::Status(int64_t slot_index, int64_t active_index,
                                     const Type& type)>
          visitor) const;
  // Replays canonical payload storage order for one variant without exposing
  // raw slot metadata to callers.
  absl::Status VisitPayloadAssemblyOrder(
      const VariantInfo& variant,
      absl::FunctionRef<absl::Status(int64_t active_index)> active_visitor,
      absl::FunctionRef<absl::Status(const Type& inactive_type)>
          inactive_visitor) const;

 private:
  absl::Status ValidateVariantInfo(const VariantInfo& variant) const;
  absl::StatusOr<const VariantInfo*> FindVariant(
      std::string_view variant_name) const;

  const SumType& type_;
  std::vector<const Type*> payload_slot_types_;
  std::vector<VariantInfo> variants_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_SUM_TYPE_ENCODING_H_
