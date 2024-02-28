// Copyright 2022 The XLS Authors
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

#ifndef XLS_JIT_TYPE_LAYOUT_H_
#define XLS_JIT_TYPE_LAYOUT_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/type_layout.pb.h"

namespace xls {

// Data structure describing the layout of a single leaf element of an xls::Type
// in the native layout used by the JIT. All offsets and sizes are in
// bytes. Sub-byte alignment is not supported.
// TODO(meheff): 2022/11/08 Add support for sub-byte alignment to allow
// expression of our "packed" representation.
struct ElementLayout {
  // Byte offset of this leaf element in the type it is contained in.
  int64_t offset;

  // The number of bytes containing actual data.
  int64_t data_size;

  // The number of bytes that the data is padded out to with zeros. All bytes
  // beyond `data_size` up to `padded_size` must be zero.
  int64_t padded_size;

  std::string ToString() const {
    return absl::StrFormat(
        "ElementLayout{.offset=%d, .data_size=%d, .padded_size=%d}", offset,
        data_size, padded_size);
  }

  bool operator==(const ElementLayout& other) const {
    return offset == other.offset && data_size == other.data_size &&
           padded_size == other.padded_size;
  }
  bool operator!=(const ElementLayout& other) const {
    return !(*this == other);
  }
};

// Abstraction describing the native data layout used by the JIT of a specific
// XLS type. The layout is encapsulated in the `ElementLayout` values, one for
// each leaf element of the type, which includes the offset and size of each
// leaf element. These are represented as a flattened vector. Examples:
//
// XLS type            Element layouts
// -------------------------------------------
// ()                  {}
// bits[1]             {ElementLayout{.offset=0, .data_size=1, .padded_size=1}}
// bits[32]            {ElementLayout{.offset=0, .data_size=4, .padded_size=4}}
// bits[42]            {ElementLayout{.offset=0, .data_size=5, .padded_size=8}}
// (bits[2], bits[9])  {ElementLayout{.offset=0, .data_size=1, .padded_size=1},
//                      ElementLayout{.offset=2, .data_size=2, .padded_size=2}}
// bits[37][3]         {ElementLayout{.offset=0, .data_size=5, .padded_size=8},
//                      ElementLayout{.offset=8, .data_size=5, .padded_size=8},
//                      ElementLayout{.offset=16, .data_size=5, .padded_size=8}}
// (bits[2], (bits[15], bits[4]))
//                     {ElementLayout{.offset=0, .data_size=1, .padded_size=8},
//                      ElementLayout{.offset=8, .data_size=2, .padded_size=4},
//                      ElementLayout{.offset=12, .data_size=1, .padded_size=4}}
//
// TODO(https://github.com/google/xls/issues/760): Reduce the redundancy in the
// array element layouts.
class TypeLayout {
 public:
  explicit TypeLayout(Type* type, int64_t size,
                      absl::Span<const ElementLayout> elements)
      : type_(type), size_(size), elements_(elements.begin(), elements.end()) {
    CHECK_EQ(elements.size(), type->leaf_count());
  }

  // Converts TypeLayout objects to/from TypeLayoutProtos.
  static absl::StatusOr<TypeLayout> FromProto(const TypeLayoutProto& proto,
                                              Package* package);
  TypeLayoutProto ToProto() const;

  // Writes `value` out to `buffer` in the native layout of the type. `buffer`
  // must have room for at least `size()` bytes.
  void ValueToNativeLayout(const Value& value, uint8_t* buffer) const;

  // Returns a Value object representing the data of XLS type `type()` stored in
  // `buffer`.
  Value NativeLayoutToValue(const uint8_t* buffer) const;

  absl::Span<const ElementLayout> elements() const { return elements_; }

  // Returns the number of bytes an instances of the type occupies.
  int64_t size() const { return size_; }

  Type* type() const { return type_; }

  std::string ToString() const;

 private:
  Value NativeLayoutToValueInternal(Type* element_type, const uint8_t* buffer,
                                    int64_t* leaf_index) const;

  Type* type_;
  int64_t size_;
  std::vector<ElementLayout> elements_;
};

std::ostream& operator<<(std::ostream& os, ElementLayout layout);
std::ostream& operator<<(std::ostream& os, const TypeLayout& layout);

}  // namespace xls

#endif  // XLS_JIT_TYPE_LAYOUT_H_
