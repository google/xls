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

#include "xls/ir/fuzz_type_domain.h"

#include <cstdint>
#include <optional>

#include "xls/common/math_util.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace {
// Returns the number of bits in the type described by type proto.
//
// If the number of bits is more than max_size, returns nullopt instead. This
// behavior is useful for avoiding overflow for large types.
std::optional<int64_t> TypeProtoSize(const TypeProto& type_proto,
                                     int64_t max_size) {
  // Go element-by-element and subtract each element's size from max_size. If we
  // get a negative number, we return nullopt indicating num_bits(type_proto) >
  // max_size.
  switch (type_proto.type_enum()) {
    case TypeProto::BITS:
      if (type_proto.bit_count() > max_size) {
        return std::nullopt;
      }
      return type_proto.bit_count();
    case TypeProto::TUPLE: {
      int64_t total_size = 0;
      for (const auto& element : type_proto.tuple_elements()) {
        std::optional<int64_t> element_size = TypeProtoSize(element, max_size);
        if (!element_size.has_value()) {
          return std::nullopt;
        }
        max_size -= *element_size;
        if (max_size < 0) {
          return std::nullopt;
        }
        total_size += *element_size;
      }
      return total_size;
    }
    case TypeProto::ARRAY: {
      std::optional<int64_t> element_size =
          TypeProtoSize(type_proto.array_element(), max_size);
      if (!element_size.has_value()) {
        return std::nullopt;
      }
      int64_t element_size_max = CeilOfRatio(max_size, type_proto.array_size());
      if (*element_size > element_size_max) {
        return std::nullopt;
      }
      int64_t array_size = *element_size * type_proto.array_size();
      if (array_size > max_size) {
        return std::nullopt;
      }
      return array_size;
    }
    default:
      return std::nullopt;
  }
}
}  // namespace

bool TypeProtoSizeInRange(const TypeProto& type_proto, int64_t min_size,
                          int64_t max_size) {
  std::optional<int64_t> size = TypeProtoSize(type_proto, max_size);
  return size.has_value() && *size >= min_size && *size < max_size;
}

}  // namespace xls
