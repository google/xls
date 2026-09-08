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

#ifndef XLS_DSLX_VALUE_FORMAT_DESCRIPTOR_H_
#define XLS_DSLX_VALUE_FORMAT_DESCRIPTOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

class ValueFormatDescriptor;

class ValueFormatSumVariantDescriptor;
class ValueFormatSumVariantView;

enum class ValueFormatSumVariantKind : int8_t {
  kUnit,
  kTuple,
  kStruct,
};

// Visits concrete types in the ValueFormatDescriptor hierarchy.
class ValueFormatVisitor {
 public:
  virtual ~ValueFormatVisitor() = default;

  virtual absl::Status HandleArray(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleEnum(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleLeafValue(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleSum(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleStruct(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleTuple(const ValueFormatDescriptor& d) = 0;
};

enum class ValueFormatDescriptorKind : int8_t {
  kLeafValue,
  kEnum,
  kArray,
  kTuple,
  kStruct,
  kSum,
};

// Class for the description of how to format values (according to the structure
// of the type as determined after type-inferencing time).
//
// These are generally static summaries of information determined by the type
// inference process so they can be used after IR conversion or in bytecode
// interpretation, where the types are fully concrete and we only need limited
// metadata in order to print them out properly. This data structure can be one
// of several kinds (enum, tuple, array, struct, sum, or leaf) corresponding to
// the respective DSLX type.
//
// Sum descriptors retain variant order, payload offsets, and enough constructor
// metadata to recover unit, tuple, and struct spellings. Their layout is copied
// from the canonical sum encoder.
class ValueFormatDescriptor {
 public:
  ValueFormatDescriptor() : kind_(ValueFormatDescriptorKind::kLeafValue) {}

  static ValueFormatDescriptor MakeLeafValue(FormatPreference format);
  static ValueFormatDescriptor MakeEnum(
      std::string_view enum_name,
      absl::flat_hash_map<Bits, std::string> value_to_name);
  static ValueFormatDescriptor MakeArray(
      const ValueFormatDescriptor& element_format, size_t size);
  static ValueFormatDescriptor MakeTuple(
      absl::Span<const ValueFormatDescriptor> elements);
  static ValueFormatDescriptor MakeStruct(
      std::string_view struct_name, absl::Span<const std::string> field_names,
      absl::Span<const ValueFormatDescriptor> field_formats);
  ValueFormatDescriptorKind kind() const { return kind_; }

  bool IsLeafValue() const {
    return kind() == ValueFormatDescriptorKind::kLeafValue;
  };
  bool IsArray() const { return kind() == ValueFormatDescriptorKind::kArray; };
  bool IsTuple() const { return kind() == ValueFormatDescriptorKind::kTuple; };
  bool IsStruct() const {
    return kind() == ValueFormatDescriptorKind::kStruct;
  };
  bool IsEnum() const { return kind() == ValueFormatDescriptorKind::kEnum; };
  bool IsSum() const { return kind() == ValueFormatDescriptorKind::kSum; }

  // Leaf methods.
  FormatPreference leaf_format() const {
    CHECK(IsLeafValue());
    return format_;
  }

  // Enum methods.
  std::string_view enum_name() const {
    CHECK(IsEnum());
    return std::get<EnumFormat>(nominal_format_).name;
  }
  const absl::flat_hash_map<Bits, std::string>& value_to_name() const {
    CHECK(IsEnum());
    return std::get<EnumFormat>(nominal_format_).value_to_name;
  }

  // Array methods.
  const ValueFormatDescriptor& array_element_format() const {
    CHECK(IsArray());
    return children_.front();
  }
  // Struct methods.
  std::string_view struct_name() const {
    CHECK(IsStruct());
    return std::get<StructFormat>(nominal_format_).name;
  }
  absl::Span<const std::string> struct_field_names() const {
    CHECK(IsStruct());
    return std::get<StructFormat>(nominal_format_).field_names;
  }
  absl::Span<const ValueFormatDescriptor> struct_elements() const {
    CHECK(IsStruct());
    return children_;
  }

  // Tuple methods.
  absl::Span<const ValueFormatDescriptor> tuple_elements() const {
    CHECK(IsTuple());
    return children_;
  }

  // Sum methods.
  std::string_view sum_name() const {
    CHECK(IsSum());
    return std::get<SumFormat>(nominal_format_).name;
  }
  size_t sum_variant_count() const {
    CHECK(IsSum());
    return std::get<SumFormat>(nominal_format_).variants.size();
  }
  ValueFormatSumVariantView sum_variant(size_t i) const;
  // Total number of slots in the internal flattened sum payload tuple.
  size_t sum_payload_slot_count() const {
    CHECK(IsSum());
    return std::get<SumFormat>(nominal_format_).payload_slot_count;
  }

  // Number of elements in tuple, array, or struct descriptors. Sum descriptors
  // instead expose their constructor count through sum_variant_count().
  size_t size() const {
    CHECK(IsTuple() || IsArray() || IsStruct());
    return size_;
  }

  absl::Status Accept(ValueFormatVisitor& v) const;

 private:
  friend class SumValueFormatBuilder;

  explicit ValueFormatDescriptor(ValueFormatDescriptorKind kind)
      : kind_(kind) {}

  // Sum formatting receives these offsets from its canonical encoder.
  static ValueFormatDescriptor MakeSum(
      std::string_view sum_name,
      absl::Span<const ValueFormatSumVariantDescriptor> variants,
      absl::Span<const size_t> payload_starts, size_t payload_slot_count);

  struct SumVariantFormat {
    SumVariantFormat(std::string name, ValueFormatSumVariantKind kind,
                     size_t payload_start, std::vector<std::string> field_names,
                     std::vector<ValueFormatDescriptor> payload_formats)
        : name(std::move(name)),
          kind(kind),
          payload_start(payload_start),
          field_names(std::move(field_names)),
          payload_formats(std::move(payload_formats)) {}

    std::string name;
    ValueFormatSumVariantKind kind;
    size_t payload_start;
    std::vector<std::string> field_names;
    std::vector<ValueFormatDescriptor> payload_formats;
  };

  struct EnumFormat {
    std::string name;
    absl::flat_hash_map<Bits, std::string> value_to_name;
  };

  struct StructFormat {
    std::string name;
    std::vector<std::string> field_names;
  };

  struct SumFormat {
    std::string name;
    std::vector<SumVariantFormat> variants;
    size_t payload_slot_count = 0;
  };

  ValueFormatDescriptorKind kind_;
  std::vector<ValueFormatDescriptor> children_;

  // Leaf data members;
  FormatPreference format_ = FormatPreference::kDefault;

  // Size of array or tuple.
  size_t size_ = 0;

  // A descriptor describes at most one nominal kind. Sharing its storage keeps
  // semantic-sum support from enlarging every ordinary bytecode instruction.
  std::variant<std::monostate, EnumFormat, StructFormat, SumFormat>
      nominal_format_;
};

// Describes one constructor inside a sum formatting descriptor.
//
// Callers build descriptors through the kind-specific factory functions so the
// unit / tuple / struct distinction is encoded at construction time instead of
// by mutable cross-field invariants.
class ValueFormatSumVariantDescriptor {
 public:
  static ValueFormatSumVariantDescriptor MakeUnit(std::string name);
  static ValueFormatSumVariantDescriptor MakeTuple(
      std::string name, std::vector<ValueFormatDescriptor> payload_formats);
  static ValueFormatSumVariantDescriptor MakeStruct(
      std::string name, std::vector<std::string> field_names,
      std::vector<ValueFormatDescriptor> payload_formats);

  std::string_view name() const { return name_; }
  ValueFormatSumVariantKind kind() const { return kind_; }
  absl::Span<const std::string> field_names() const { return field_names_; }
  absl::Span<const ValueFormatDescriptor> payload_formats() const {
    return payload_formats_;
  }

 private:
  ValueFormatSumVariantDescriptor(
      std::string name, ValueFormatSumVariantKind kind,
      std::vector<std::string> field_names,
      std::vector<ValueFormatDescriptor> payload_formats)
      : name_(std::move(name)),
        kind_(kind),
        field_names_(std::move(field_names)),
        payload_formats_(std::move(payload_formats)) {}

  std::string name_;
  ValueFormatSumVariantKind kind_;
  std::vector<std::string> field_names_;
  std::vector<ValueFormatDescriptor> payload_formats_;
};

// Read-only view of one constructor inside a sum formatting descriptor.
//
// The payload formats and canonical storage offset let callers format one
// variant without reconstructing the broader flattened sum storage layout.
class ValueFormatSumVariantView {
 public:
  std::string_view name() const { return name_; }
  ValueFormatSumVariantKind kind() const { return kind_; }
  size_t payload_start() const { return payload_start_; }
  size_t payload_slot_count() const { return payload_formats_.size(); }
  absl::Span<const std::string> field_names() const { return field_names_; }
  absl::Span<const ValueFormatDescriptor> payload_formats() const {
    return payload_formats_;
  }

 private:
  friend class ValueFormatDescriptor;

  ValueFormatSumVariantView(
      std::string_view name, ValueFormatSumVariantKind kind,
      size_t payload_start, absl::Span<const std::string> field_names,
      absl::Span<const ValueFormatDescriptor> payload_formats)
      : name_(name),
        kind_(kind),
        payload_start_(payload_start),
        field_names_(field_names),
        payload_formats_(payload_formats) {}

  std::string_view name_;
  ValueFormatSumVariantKind kind_;
  size_t payload_start_;
  absl::Span<const std::string> field_names_;
  absl::Span<const ValueFormatDescriptor> payload_formats_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_VALUE_FORMAT_DESCRIPTOR_H_
