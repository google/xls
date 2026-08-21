// Copyright 2026 The XLS Authors
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

#ifndef XLS_PASSES_RESOURCE_SHARING_EQUIVALENCE_H_
#define XLS_PASSES_RESOURCE_SHARING_EQUIVALENCE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

// `EquivalenceMapping` is an abstract class representing a concrete plan to
// map an `original` node into a `variant` node by transforming its operands
// (and optionally its output) so that the `variant` node produces an equivalent
// result.
//
// Implementations of this class decide their own internal data structures to
// represent the required transformations and provide a `TryCreate` static
// method to test if a mapping is supported:
//
//   static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
//       const Node* original, const Node* variant);
//
class EquivalenceMapping {
 public:
  EquivalenceMapping(const Node* original, const Node* variant)
      : original_(original), variant_(variant) {}
  virtual ~EquivalenceMapping() = default;

  // Returns the original node this mapping was created for.
  const Node* original() const { return original_; }

  // Returns the variant node this mapping maps to.
  const Node* variant() const { return variant_; }

  // Applies this mapping's operand transformations to `original_operands`,
  // returning a new vector of operands suitable for `variant`.
  // `original_operands` can be `original()->operands()` or any other vector of
  // operands (e.g. from cloned nodes).
  virtual absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> original_operands) const = 0;

  // Applies the output transformation to `variant_output`, returning a node
  // that is functionally equivalent to `original()`.
  virtual absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                              Node* variant_output) const = 0;

  // Returns the estimated area overhead incurred by this mapping (e.g.
  // negation logic, bit-extension, output slicing). `operands` and `output`
  // allow the mapping to inspect the specific nodes being transformed
  // (defaulting to `original()->operands()` and `original()`).
  virtual absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      const Node* output) const {
    return 0.0;
  }

  // Returns true if this mapping modifies any of the operands.
  virtual bool RequiresOperandTransformation() const { return true; }

  // Returns true if this mapping modifies the output of the variant node.
  virtual bool RequiresOutputTransformation() const { return false; }

 protected:
  const Node* original_;
  const Node* variant_;
};

// Returns true if both nodes are bits-typed.
inline bool IsBitsTyped(const Node* a, const Node* b) {
  return a->GetType()->IsBits() && b->GetType()->IsBits();
}

// Returns true if both nodes are bits-typed and have the same bitwidth.
inline bool HasSameBitWidth(const Node* a, const Node* b) {
  return IsBitsTyped(a, b) && a->BitCountOrDie() == b->BitCountOrDie();
}

// Returns true if all corresponding operands of `a` and `b` are bits-typed
// and have the same bitwidth.
inline bool HaveSameOperandBitWidths(const Node* a, const Node* b) {
  if (a->operand_count() != b->operand_count()) {
    return false;
  }
  for (int64_t i = 0; i < a->operand_count(); ++i) {
    if (!HasSameBitWidth(a->operand(i), b->operand(i))) {
      return false;
    }
  }
  return true;
}

// Handles nodes with identical ops and bit widths.
class IdentityEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant);

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f,
      absl::Span<Node* const> original_operands) const override {
    return std::vector<Node*>(original_operands.begin(),
                              original_operands.end());
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* variant_output) const override {
    return variant_output;
  }

  bool RequiresOperandTransformation() const override { return false; }
  bool RequiresOutputTransformation() const override { return false; }
};

// Equivalence mapping between logical shift left and right.
//
// y = x << s is equivalent to y = reverse(reverse(x) >> s)
// y = x >> s is equivalent to y = reverse(reverse(x) << s)
class ShiftEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant);

  using EquivalenceMapping::EquivalenceMapping;

  bool RequiresOperandTransformation() const override { return true; }
  bool RequiresOutputTransformation() const override { return true; }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f,
      absl::Span<Node* const> original_operands) const override;

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* variant_output) const override;

  absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator,
      absl::Span<Node* const> original_operands,
      const Node* original_node) const override;
};

// `NodeEquivalenceMapper` is a thread-safe registry of `EquivalenceMapping`
// implementations. It iterates over all registered `EquivalenceMapping`
// `TryCreate` factories to find a valid mapping between two nodes.
class NodeEquivalenceMapper {
 public:
  using Factory = std::optional<std::unique_ptr<EquivalenceMapping>> (*)(
      const Node* original, const Node* variant);

  NodeEquivalenceMapper() = default;
  ~NodeEquivalenceMapper() = default;

  // `NodeEquivalenceMapper` is move-only to preserve mutex safety.
  NodeEquivalenceMapper(NodeEquivalenceMapper&&) = default;
  NodeEquivalenceMapper& operator=(NodeEquivalenceMapper&&) = default;
  NodeEquivalenceMapper(const NodeEquivalenceMapper&) = delete;
  NodeEquivalenceMapper& operator=(const NodeEquivalenceMapper&) = delete;

  // Registers a new `EquivalenceMapping` `TryCreate` factory by function
  // pointer.
  void Register(Factory factory);

  // Registers a new `EquivalenceMapping` implementation.
  template <typename EqMapping>
  void Register() {
    static_assert(std::is_base_of_v<EquivalenceMapping, EqMapping>,
                  "EqMapping must be a subclass of EquivalenceMapping");
    Register(&EqMapping::TryCreate);
  }

  // Returns an `EquivalenceMapping` if `original` can be mapped to `variant`
  // using any registered `EquivalenceMapping` implementation, or `std::nullopt`
  // if no mapping applies.
  std::optional<std::unique_ptr<EquivalenceMapping>> ComputeMapping(
      const Node* original, const Node* variant) const;

 private:
  mutable absl::Mutex mutex_;
  std::vector<Factory> factories_ ABSL_GUARDED_BY(mutex_);
};

// Returns the global singleton `NodeEquivalenceMapper`.
NodeEquivalenceMapper& GetNodeEquivalenceMapper();

// Helper to register an `EquivalenceMapping` implementation `T` with the global
// `NodeEquivalenceMapper`.
template <typename T>
void RegisterEquivalenceMapping() {
  GetNodeEquivalenceMapper().Register<T>();
}

}  // namespace xls

#endif  // XLS_PASSES_RESOURCE_SHARING_EQUIVALENCE_H_
