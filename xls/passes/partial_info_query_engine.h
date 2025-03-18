// Copyright 2025 The XLS Authors
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

#ifndef XLS_PASSES_PARTIAL_INFO_QUERY_ENGINE_H_
#define XLS_PASSES_PARTIAL_INFO_QUERY_ENGINE_H_

#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/ternary.h"
#include "xls/passes/lazy_query_engine.h"
#include "xls/passes/query_engine.h"

namespace xls {

class PartialInfoQueryEngine : public LazyQueryEngine<PartialInformation> {
 public:
  std::optional<SharedTernaryTree> GetTernary(Node* node) const override;
  IntervalSetTree GetIntervals(Node* node) const override;

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;

  // This query engine provides little information about bit implications.
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return false;
  }
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return std::nullopt;
  }

 protected:
  LeafTypeTree<PartialInformation> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<PartialInformation>* const> operand_infos)
      const override;

  absl::Status MergeWithGiven(PartialInformation& info,
                              const PartialInformation& given) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_PARTIAL_INFO_QUERY_ENGINE_H_
