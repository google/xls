// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_CSE_PASS_H_
#define XLS_PASSES_CSE_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// This function is called by the `CsePass` to merge together common
// subexpressions. It exists so that you can call it inside other passes and
// extract which nodes were merged. Each replacement done by the pass is added
// to the `replacements` hash map if it is not `nullptr`. Note that for many
// common uses of the `replacements` map, you'll want to compute the transitive
// closure of the relation rather than using it as-is.
absl::StatusOr<bool> RunCse(FunctionBase* f,
                            absl::flat_hash_map<Node*, Node*>* replacements);

// Computes the fixed point of a strict partial order, i.e.: the relation that
// solves the equation `F = R ∘ F` where `R` is the given strict partial order.
template <typename T>
absl::flat_hash_map<T, T> FixedPointOfSPO(
    const absl::flat_hash_map<T, T>& relation) {
  absl::flat_hash_map<T, T> result;
  for (const auto& [source, target] : relation) {
    T node = target;
    while (relation.contains(node)) {
      if (node == relation.at(node)) {
        break;
      }
      node = relation.at(node);
    }
    result[source] = node;
  }
  return result;
}

// Pass which performs common subexpression elimination. Equivalent ops with the
// same operands are commoned. The pass can find arbitrarily large common
// expressions.
class CsePass : public FunctionBasePass {
 public:
  CsePass() : FunctionBasePass("cse", "Common subexpression elimination") {}
  ~CsePass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CSE_PASS_H_
