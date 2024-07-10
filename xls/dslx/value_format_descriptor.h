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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

class ValueFormatDescriptor;

// Visits concrete types in the ValueFormatDescriptor hierarchy.
class ValueFormatVisitor {
 public:
  virtual ~ValueFormatVisitor() = default;

  virtual absl::Status HandleArray(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleEnum(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleLeafValue(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleStruct(const ValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleTuple(const ValueFormatDescriptor& d) = 0;
};

enum class ValueFormatDescriptorKind : int8_t {
  kLeafValue,
  kEnum,
  kArray,
  kTuple,
  kStruct,
};

// Class for the description of how to format values (according to the structure
// of the type as determined after type-inferencing time).
//
// These are generally static summaries of information determined by the type
// inference process so they can be used after IR conversion or in bytecode
// interpretation, where the types are fully concrete and we only need limited
// metadata in order to print them out properly. This data structure can be one
// of several kinds (enum, tuple, array, struct, or leaf) corresponding to the
// respective DSLX type.
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

  // Leaf methods.
  FormatPreference leaf_format() const {
    CHECK(IsLeafValue());
    return format_;
  }

  // Enum methods.
  std::string_view enum_name() const {
    CHECK(IsEnum());
    return enum_name_;
  }
  const absl::flat_hash_map<Bits, std::string>& value_to_name() const {
    CHECK(IsEnum());
    return value_to_name_;
  }

  // Array methods.
  const ValueFormatDescriptor& array_element_format() const {
    CHECK(IsArray());
    return children_.front();
  }
  // Struct methods.
  std::string_view struct_name() const {
    CHECK(IsStruct());
    return struct_name_;
  }
  absl::Span<const std::string> struct_field_names() const {
    CHECK(IsStruct());
    return struct_field_names_;
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

  // Methods for aggregate kinds.
  size_t size() const {
    CHECK(IsTuple() || IsArray() || IsStruct());
    return size_;
  }

  absl::Status Accept(ValueFormatVisitor& v) const;

 private:
  explicit ValueFormatDescriptor(ValueFormatDescriptorKind kind)
      : kind_(kind) {}

  ValueFormatDescriptorKind kind_;
  std::vector<ValueFormatDescriptor> children_;

  // Leaf data members;
  FormatPreference format_ = FormatPreference::kDefault;

  // Enum data members.
  std::string enum_name_;
  absl::flat_hash_map<Bits, std::string> value_to_name_;

  // Size of array or tuple.
  size_t size_ = 0;

  std::string struct_name_;
  std::vector<std::string> struct_field_names_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_VALUE_FORMAT_DESCRIPTOR_H_
