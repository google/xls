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

#ifndef XLS_IR_FUZZ_TYPE_DOMAIN_H_
#define XLS_IR_FUZZ_TYPE_DOMAIN_H_

#include <cstdint>

#include "fuzztest/fuzztest.h"
#include "absl/log/check.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
using fuzztest::Arbitrary;
using fuzztest::Domain;
using fuzztest::DomainBuilder;
using fuzztest::Filter;
using fuzztest::InRange;
using fuzztest::Just;
using fuzztest::OneOf;
using fuzztest::Positive;
using fuzztest::VectorOf;

// Domain that produces bits types.
inline Domain<TypeProto> BitsTypeDomain(int64_t max_bit_count) {
  return Arbitrary<TypeProto>()
      .WithFieldUnset("tuple_elements")
      .WithFieldUnset("array_size")
      .WithFieldUnset("array_element")
      .WithEnumFieldAlwaysSet("type_enum", Just<int>(TypeProto::BITS))
      .WithInt64FieldAlwaysSet("bit_count", InRange<int64_t>(1, max_bit_count));
}

// Domain that produces tuple types for a given element domain.
inline Domain<TypeProto> TupleTypeDomain(
    const Domain<TypeProto>& element_domain, int64_t max_elements) {
  return Arbitrary<TypeProto>()
      .WithFieldUnset("bit_count")
      .WithFieldUnset("array_size")
      .WithFieldUnset("array_element")
      .WithEnumFieldAlwaysSet("type_enum", Just<int>(TypeProto::TUPLE))
      .WithRepeatedProtobufField(
          "tuple_elements",
          VectorOf(element_domain).WithMinSize(0).WithMaxSize(max_elements));
}

// Domain that produces array types for a given element domain.
inline Domain<TypeProto> ArrayTypeDomain(
    const Domain<TypeProto>& element_domain, int64_t max_elements) {
  return Arbitrary<TypeProto>()
      .WithFieldUnset("bit_count")
      .WithFieldUnset("tuple_elements")
      .WithEnumFieldAlwaysSet("type_enum", Just<int>(TypeProto::ARRAY))
      .WithInt64FieldAlwaysSet("array_size", InRange<int64_t>(1, max_elements))
      .WithProtobufFieldAlwaysSet("array_element", element_domain);
}

namespace internal {
// Ideally, we'd use a DomainBuilder to allow arbitrary nesting, but it seems
// that the protobuf field domains do not defer initialization of their domains,
// so trying to use them will give an error. Instead, we set a maximum nesting
// level as a parameter.
inline constexpr int64_t kDefaultNestingLevel = 5;
}

// Domain that produces arbitrary bits, tuple, and array types.
// Aggregates can be nested to nesting_level.
inline Domain<TypeProto> TypeDomain(
    int64_t max_bit_count, int64_t max_elements,
    int64_t nesting_level = internal::kDefaultNestingLevel) {
  CHECK_GT(nesting_level, 0);
  if (nesting_level == 1) {
    return BitsTypeDomain(max_bit_count);
  }
  nesting_level--;
  return OneOf(BitsTypeDomain(max_bit_count),
               TupleTypeDomain(TypeDomain(/*max_bit_count=*/max_bit_count,
                                          /*max_elements=*/max_elements,
                                          /*nesting_level=*/nesting_level),
                               max_elements),
               ArrayTypeDomain(TypeDomain(/*max_bit_count=*/max_bit_count,
                                          /*max_elements=*/max_elements,
                                          /*nesting_level=*/nesting_level),
                               max_elements));
}

// Returns true if size(type_proto) in [min_size, max_size)
bool TypeProtoSizeInRange(const TypeProto& type_proto, int64_t min_size,
                          int64_t max_size);

// A filter for TypeProto Domains that limits the total number of bits to in
// [min_size, max_size).
inline Domain<TypeProto> TypeDomainWithSizeInRange(
    int64_t min_size, int64_t max_size, const Domain<TypeProto>& type_domain) {
  return Filter(
      [max_size](const TypeProto& type_proto) {
        return TypeProtoSizeInRange(type_proto, /*min_size=*/1,
                                    /*max_size=*/max_size);
      },
      type_domain);
}

}  // namespace xls

#endif  // XLS_IR_FUZZ_TYPE_DOMAIN_H_
