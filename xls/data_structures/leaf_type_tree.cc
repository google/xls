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
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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

std::string ToStringHelper(Type* subtype,
                           absl::Span<const std::string> elements,
                           bool multiline, int64_t indent,
                           int64_t& linear_index) {
  std::string indentation(indent, ' ');
  if (subtype->IsArray()) {
    std::vector<std::string> pieces;
    pieces.reserve(subtype->AsArrayOrDie()->size());
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
    pieces.reserve(subtype->AsTupleOrDie()->size());
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

std::string ToString(Type* t, absl::Span<const std::string> elements,
                     bool multiline) {
  int64_t linear_index = 0;
  return ToStringHelper(t, elements, multiline, /*indent=*/0, linear_index);
}

LeafTypeTreeIterator::LeafTypeTreeIterator(
    Type* type, absl::Span<const int64_t> index_prefix)
    : root_type_(type),
      prefix_size_(index_prefix.size()),
      linear_index_(0),
      type_index_(index_prefix.begin(), index_prefix.end()) {}

bool LeafTypeTreeIterator::Advance() {
  CHECK(!AtEnd());
  ++linear_index_;

  // Reset the type index to just the prefix; the rest of it will be computed
  // on demand if needed in `type_index()` below.
  type_index_.resize(prefix_size_);

  return !AtEnd();
}

absl::Span<const int64_t> LeafTypeTreeIterator::type_index() const {
  CHECK(!AtEnd());
  if (prefix_size_ == 0) {
    return root_type_->tree_index_vectors()[linear_index_];
  }
  if (type_index_.size() > prefix_size_) {
    // Already computed!
    return type_index_;
  }

  // Lazily initialize the type index; it's currently equal to the prefix.
  absl::Span<const int64_t> tree_index =
      root_type_->tree_index_vectors()[linear_index_];
  if (tree_index.empty()) {
    // It turns out the prefix *is* the type index.
    return type_index_;
  }
  type_index_.reserve(prefix_size_ + tree_index.size());
  absl::c_copy(tree_index, std::back_inserter(type_index_));
  return type_index_;
}

std::string LeafTypeTreeIterator::ToString() const {
  if (AtEnd()) {
    return absl::StrFormat("root_type=%s, END", root_type_->ToString());
  }
  return absl::StrFormat(
      "root_type=%s, leaf_type=%s, type_index={%s}, linear_index=%d",
      root_type_->ToString(), root_type_->leaf_type(linear_index_)->ToString(),
      absl::StrJoin(type_index(), ","), linear_index_);
}

}  // namespace leaf_type_tree_internal
}  // namespace xls
