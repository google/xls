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
// map an `original` node onto a `variant` node by transforming some
// combination of the original node's operands, the node used to unify
// `original` with `variant`, and/or the output of the unified node.
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
  // returning a new vector of operands suitable for `variant`, or the unified
  // version of `variant` if `variant` itself requires modification too.
  // `original_operands` can be `original()->operands()` or any other vector of
  // operands (e.g. from cloned nodes).
  virtual absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> original_operands) const = 0;

  // Applies the output transformation to `unified_output`, returning a node
  // that is functionally equivalent to `original()`.
  virtual absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                              Node* unified_output) const = 0;

  // Returns an EquivalenceMapping that adapts `variant()` to `unified_node`
  // when `ModifiesVariantNode()` is true, e.g. a bit width extending mapper if
  // `unified_node` requires more bits than either `original` or `variant`.
  virtual std::unique_ptr<EquivalenceMapping> GetVariantMapping(
      const Node* unified_node) const;

  // Creates the unified node given the multiplexed operands.
  virtual absl::StatusOr<Node*> CreateUnifiedNode(
      FunctionBase* f, absl::Span<Node* const> operands) const {
    return variant_->CloneInNewFunction(operands, f);
  }

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

  // Returns true if this mapping modifies the output of the unified node.
  virtual bool RequiresOutputTransformation() const { return false; }

  // Returns true if this mapping modifies variant_ when creating the unified
  // node, e.g. widening needed to accommodate transformations done to make
  // `original` and `variant` compatible.
  virtual bool ModifiesVariantNode() const { return false; }

 protected:
  const Node* original_;
  const Node* variant_;
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

// `IdentityEquivalenceMapping` handles nodes with identical ops and bit widths.
// This mapping is exposed for ease of default mapping in constructors.
class IdentityEquivalenceMapping : public EquivalenceMapping {
 public:
  static std::optional<std::unique_ptr<EquivalenceMapping>> TryCreate(
      const Node* original, const Node* variant) {
    if (!original->IsDefinitelyEqualTo(variant)) {
      return std::nullopt;
    }
    return std::make_unique<IdentityEquivalenceMapping>(original, variant);
  }

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

}  // namespace xls

#endif  // XLS_PASSES_RESOURCE_SHARING_EQUIVALENCE_H_
