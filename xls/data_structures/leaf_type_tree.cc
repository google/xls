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

#include "xls/data_structures/leaf_type_tree.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/type.h"

namespace xls {
namespace leaf_type_tree_internal {

bool IncrementArrayIndex(absl::Span<const int64_t> bounds,
                         std::vector<int64_t>* array_index) {
  CHECK_EQ(bounds.size(), array_index->size());
  for (int64_t i = array_index->size() - 1; i >= 0; --i) {
    ++(*array_index)[i];
    if ((*array_index)[i] < bounds[i]) {
      return false;
    }
    (*array_index)[i] = 0;
  }
  return true;
}

absl::StatusOr<SubArraySize> GetSubArraySize(Type* type, int64_t index_depth) {
  std::vector<int64_t> bounds;
  Type* subtype = type;
  for (int64_t i = 0; i < index_depth; ++i) {
    if (!subtype->IsArray()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Type has fewer than %d array dimensions: %s",
                          index_depth, type->ToString()));
    }
    int64_t bound = subtype->AsArrayOrDie()->size();
    bounds.push_back(bound);
    subtype = subtype->AsArrayOrDie()->element_type();
  }

  return SubArraySize{.type = subtype,
                      .bounds = std::move(bounds),
                      .element_count = subtype->leaf_count()};
}

namespace {

std::pair<Type*, int64_t> GetSubtypeAndOffsetHelper(
    Type* t, absl::Span<int64_t const> index, int64_t offset) {
  if (index.empty()) {
    return {t, offset};
  }
  if (t->IsArray()) {
    CHECK(!index.empty());
    CHECK_LT(index[0], t->AsArrayOrDie()->size());
    Type* element_type = t->AsArrayOrDie()->element_type();
    return GetSubtypeAndOffsetHelper(
        element_type, index.subspan(1),
        offset + index[0] * element_type->leaf_count());
  }
  CHECK(t->IsTuple());
  TupleType* tuple_type = t->AsTupleOrDie();
  CHECK_LT(index[0], tuple_type->size());
  int64_t element_offset = 0;
  for (int64_t i = 0; i < index[0]; ++i) {
    element_offset += tuple_type->element_type(i)->leaf_count();
  }
  return GetSubtypeAndOffsetHelper(tuple_type->element_type(index[0]),
                                   index.subspan(1), offset + element_offset);
}

void GetLeafTypesHelper(Type* t, absl::InlinedVector<Type*, 1>* leaf_types) {
  if (IsLeafType(t)) {
    leaf_types->push_back(t);
    return;
  }
  if (t->IsArray()) {
    for (int64_t i = 0; i < t->AsArrayOrDie()->size(); ++i) {
      GetLeafTypesHelper(t->AsArrayOrDie()->element_type(), leaf_types);
    }
    return;
  }
  CHECK(t->IsTuple());
  for (int64_t i = 0; i < t->AsTupleOrDie()->size(); ++i) {
    GetLeafTypesHelper(t->AsTupleOrDie()->element_type(i), leaf_types);
  }
}

std::string ToStringHelper(Type* subtype,
                           absl::Span<const std::string> elements,
                           bool multiline, int64_t indent,
                           int64_t& linear_index) {
  std::string indentation(indent, ' ');
  if (subtype->IsArray()) {
    std::vector<std::string> pieces;
    for (int64_t i = 0; i < subtype->AsArrayOrDie()->size(); ++i) {
      pieces.push_back(ToStringHelper(subtype->AsArrayOrDie()->element_type(),
                                      elements, multiline, indent + 2,
                                      linear_index));
    }
    if (multiline) {
      return absl::StrFormat("%s[\n%s\n%s]", indentation,
                             absl::StrJoin(pieces, ",\n"), indentation);
    }
    return absl::StrFormat("[%s]", absl::StrJoin(pieces, ", "));
  }
  if (subtype->IsTuple()) {
    std::vector<std::string> pieces;
    for (int64_t i = 0; i < subtype->AsTupleOrDie()->size(); ++i) {
      pieces.push_back(ToStringHelper(subtype->AsTupleOrDie()->element_type(i),
                                      elements, multiline, indent + 2,
                                      linear_index));
    }
    if (multiline) {
      if (pieces.empty()) {
        return absl::StrFormat("%s()", indentation);
      }
      return absl::StrFormat("%s(\n%s\n%s)", indentation,
                             absl::StrJoin(pieces, ",\n"), indentation);
    }
    return absl::StrFormat("(%s)", absl::StrJoin(pieces, ", "));
  }
  if (multiline) {
    return absl::StrFormat("%s%s", indentation, elements[linear_index++]);
  }
  return elements[linear_index++];
}

}  // namespace

std::pair<Type*, int64_t> GetSubtypeAndOffset(Type* t,
                                              absl::Span<int64_t const> index) {
  return GetSubtypeAndOffsetHelper(t, index, /*offset=*/0);
}

absl::InlinedVector<Type*, 1> GetLeafTypes(Type* t) {
  absl::InlinedVector<Type*, 1> leaf_types;
  GetLeafTypesHelper(t, &leaf_types);
  return leaf_types;
}

int64_t GetLeafTypeOffset(Type* t, absl::Span<int64_t const> index) {
  auto [subtype, offset] = GetSubtypeAndOffset(t, index);
  CHECK(IsLeafType(subtype));
  return offset;
}

std::string ToString(Type* t, absl::Span<const std::string> elements,
                     bool multiline) {
  int64_t linear_index = 0;
  return ToStringHelper(t, elements, multiline, /*indent=*/0, linear_index);
}

}  // namespace leaf_type_tree_internal
}  // namespace xls
