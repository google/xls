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

#ifndef XLS_PASSES_BDD_EVALUATOR_H_
#define XLS_PASSES_BDD_EVALUATOR_H_

#include <cstdint>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/abstract_evaluator.h"

namespace xls {

// Construct a BDD-based abstract evaluator. The expressions in the BDD
// saturates at a particular number of paths from the expression node to the
// terminal nodes 0 and 1 in the BDD. When the path limit is met, a new BDD
// variable is created in its place effective forgetting any information about
// the value. This avoids exponential blowup problems when constructing the BDD
// at the cost of precision. The primitive bit element of the abstract evaluator
// is a sum type consisting of a BDD node and a sentinel value TooManyPaths. The
// TooManyPaths value is produced if the number of paths in the computed
// expression exceed some limit. Any logical operation performed with a
// TooManyPaths value produces a TooManyPaths value.
struct TooManyPaths : public std::monostate {};
using SaturatingBddNodeIndex = std::variant<BddNodeIndex, TooManyPaths>;
using SaturatingBddNodeVector = std::vector<SaturatingBddNodeIndex>;

inline bool HasTooManyPaths(const SaturatingBddNodeIndex& input) {
  return std::holds_alternative<TooManyPaths>(input);
}

inline bool HasTooManyPaths(const SaturatingBddNodeVector& input) {
  return absl::c_any_of(input, [](const SaturatingBddNodeIndex& x) {
    return HasTooManyPaths(x);
  });
}

inline BddNodeIndex ToBddNode(const SaturatingBddNodeIndex& input) {
  CHECK(!HasTooManyPaths(input));
  return std::get<BddNodeIndex>(input);
}

// Converts the given saturating BDD vector to a normal vector of BDD nodes. The
// input vector must not contain any TooManyPaths values.
inline std::vector<BddNodeIndex> ToBddNodeVector(
    const SaturatingBddNodeVector& input) {
  std::vector<BddNodeIndex> result(input.size());
  for (int64_t i = 0; i < input.size(); ++i) {
    CHECK(!HasTooManyPaths(input[i]));
    result[i] = std::get<BddNodeIndex>(input[i]);
  }
  return result;
}

// The abstract evaluator based on a BDD with path-saturating logic.
class SaturatingBddEvaluator
    : public AbstractEvaluator<SaturatingBddNodeIndex, SaturatingBddEvaluator> {
 public:
  SaturatingBddEvaluator(int64_t path_limit, BinaryDecisionDiagram* bdd)
      : path_limit_(path_limit), bdd_(bdd) {}

  SaturatingBddNodeIndex One() const { return bdd_->one(); }

  SaturatingBddNodeIndex Zero() const { return bdd_->zero(); }

  SaturatingBddNodeIndex Not(const SaturatingBddNodeIndex& input) const {
    if (HasTooManyPaths(input)) {
      return TooManyPaths();
    }
    BddNodeIndex result = bdd_->Not(std::get<BddNodeIndex>(input));
    if (path_limit_ > 0 && bdd_->path_count(result) > path_limit_) {
      return TooManyPaths();
    }
    return result;
  }

  SaturatingBddNodeIndex And(const SaturatingBddNodeIndex& a,
                             const SaturatingBddNodeIndex& b) const {
    if (HasTooManyPaths(a) || HasTooManyPaths(b)) {
      return TooManyPaths();
    }
    BddNodeIndex result =
        bdd_->And(std::get<BddNodeIndex>(a), std::get<BddNodeIndex>(b));
    if (path_limit_ > 0 && bdd_->path_count(result) > path_limit_) {
      return TooManyPaths();
    }
    return result;
  }

  SaturatingBddNodeIndex Or(const SaturatingBddNodeIndex& a,
                            const SaturatingBddNodeIndex& b) const {
    if (HasTooManyPaths(a) || HasTooManyPaths(b)) {
      return TooManyPaths();
    }
    BddNodeIndex result =
        bdd_->Or(std::get<BddNodeIndex>(a), std::get<BddNodeIndex>(b));
    if (path_limit_ > 0 && bdd_->path_count(result) > path_limit_) {
      return TooManyPaths();
    }
    return result;
  }

  SaturatingBddNodeIndex If(const SaturatingBddNodeIndex& sel,
                            const SaturatingBddNodeIndex& consequent,
                            const SaturatingBddNodeIndex& alternate) const {
    if (HasTooManyPaths(sel) || HasTooManyPaths(consequent) ||
        HasTooManyPaths(alternate)) {
      return TooManyPaths();
    }
    BddNodeIndex result = bdd_->IfThenElse(std::get<BddNodeIndex>(sel),
                                           std::get<BddNodeIndex>(consequent),
                                           std::get<BddNodeIndex>(alternate));
    if (path_limit_ > 0 && bdd_->path_count(result) > path_limit_) {
      return TooManyPaths();
    }
    return result;
  }

 private:
  int64_t path_limit_;
  BinaryDecisionDiagram* bdd_;
};

}  // namespace xls

#endif  // XLS_PASSES_BDD_EVALUATOR_H_
