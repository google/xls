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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

class ArrayFormatDescriptor;
class EnumFormatDescriptor;
class LeafValueFormatDescriptor;
class StructFormatDescriptor;
class TupleFormatDescriptor;

// Visits concrete types in the ValueFormatDescriptor hierarchy.
class ValueFormatVisitor {
 public:
  virtual ~ValueFormatVisitor() = default;

  virtual absl::Status HandleArray(const ArrayFormatDescriptor& d) = 0;
  virtual absl::Status HandleEnum(const EnumFormatDescriptor& d) = 0;
  virtual absl::Status HandleLeafValue(const LeafValueFormatDescriptor& d) = 0;
  virtual absl::Status HandleStruct(const StructFormatDescriptor& d) = 0;
  virtual absl::Status HandleTuple(const TupleFormatDescriptor& d) = 0;
};

// Abstract base class (existential) for the description of how to format values
// (according to the structure of the type as determined after type-inferencing
// time).
//
// These are generally static summaries of information determined by the type
// inference process so they can be used after IR conversion or in bytecode
// interpretation, where the types are fully concrete and we only need limited
// metadata in order to print them out properly. This type hierarchy effectively
// corresponds to that of `Type`.
class ValueFormatDescriptor {
 public:
  virtual ~ValueFormatDescriptor() = default;

  // Accepts the visitor and calls `v.Handle*(*this)` -- note that there is no
  // recursive traversal, just double dispatch, traversal would need to be
  // implemented by the visitor implementation.
  virtual absl::Status Accept(ValueFormatVisitor& v) const = 0;
};

class LeafValueFormatDescriptor : public ValueFormatDescriptor {
 public:
  explicit LeafValueFormatDescriptor(FormatPreference format)
      : format_(format) {}

  FormatPreference format() const { return format_; }

  absl::Status Accept(ValueFormatVisitor& v) const override {
    return v.HandleLeafValue(*this);
  }

 private:
  FormatPreference format_;
};

class EnumFormatDescriptor : public ValueFormatDescriptor {
 public:
  EnumFormatDescriptor(std::string enum_name,
                       absl::flat_hash_map<Bits, std::string> value_to_name)
      : enum_name_(std::move(enum_name)),
        value_to_name_(std::move(value_to_name)) {}

  absl::Status Accept(ValueFormatVisitor& v) const override {
    return v.HandleEnum(*this);
  }

  const std::string& enum_name() const { return enum_name_; }
  const absl::flat_hash_map<Bits, std::string>& value_to_name() const {
    return value_to_name_;
  }

 private:
  std::string enum_name_;
  absl::flat_hash_map<Bits, std::string> value_to_name_;
};

class ArrayFormatDescriptor : public ValueFormatDescriptor {
 public:
  ArrayFormatDescriptor(std::unique_ptr<ValueFormatDescriptor> element_format,
                        size_t size)
      : element_format_(std::move(element_format)), size_(size) {}

  absl::Status Accept(ValueFormatVisitor& v) const override {
    return v.HandleArray(*this);
  }

  const ValueFormatDescriptor& element_format() const {
    return *element_format_;
  }
  size_t size() const { return size_; }

 private:
  std::unique_ptr<ValueFormatDescriptor> element_format_;
  size_t size_;
};

class TupleFormatDescriptor : public ValueFormatDescriptor {
 public:
  explicit TupleFormatDescriptor(
      std::vector<std::unique_ptr<ValueFormatDescriptor>> elements)
      : elements_(std::move(elements)) {}

  absl::Status Accept(ValueFormatVisitor& v) const override {
    return v.HandleTuple(*this);
  }

  int64_t size() const { return elements_.size(); }
  absl::Span<const std::unique_ptr<ValueFormatDescriptor>> elements() const {
    return elements_;
  }

 private:
  std::vector<std::unique_ptr<ValueFormatDescriptor>> elements_;
};

// Describes how a struct should be formatted.
//
// (Note: recursive type, as this is also used for sub-structs under the top
// level struct.)
class StructFormatDescriptor : public ValueFormatDescriptor {
 public:
  // A given element has a field name and either describes a leaf of formatting
  // (a value in a field) or a sub-struct via a boxed StructFormatDescriptor.
  struct Element {
    std::string field_name;
    std::unique_ptr<ValueFormatDescriptor> fmt;
  };

  StructFormatDescriptor(std::string struct_name, std::vector<Element> elements)
      : struct_name_(std::move(struct_name)), elements_(std::move(elements)) {}

  absl::Status Accept(ValueFormatVisitor& v) const override {
    return v.HandleStruct(*this);
  }

  const std::string& struct_name() const { return struct_name_; }
  const std::vector<Element>& elements() const { return elements_; }

 private:
  std::string struct_name_;
  std::vector<Element> elements_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_VALUE_FORMAT_DESCRIPTOR_H_
