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
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

// `EquivalenceMapping` is an abstract class representing a concrete plan to
// map a source node onto a destination node by transforming some combination
// of the `src` node's operands and/or the output as used by the original users
// of `src`. The `dst` node is then usable as-is in place of `src`.
class EquivalenceMapping {
 public:
  EquivalenceMapping(Node* src, Node* dst,
                     std::shared_ptr<Package> tmp_package = nullptr)
      : src_(src), dst_(dst), tmp_package_(std::move(tmp_package)) {}
  EquivalenceMapping(const EquivalenceMapping&) = delete;
  EquivalenceMapping& operator=(const EquivalenceMapping&) = delete;
  EquivalenceMapping(EquivalenceMapping&&) = default;
  EquivalenceMapping& operator=(EquivalenceMapping&&) = default;
  virtual ~EquivalenceMapping() = default;

  Node* src() const { return src_; }
  Node* dst() const { return dst_; }

  // Clones this mapping. If `original_node_to_clone` is non-empty, `src_` and
  // `dst_`, if not from `tmp_package_`, must be in `original_node_to_clone`.
  virtual absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const = 0;

  // Applies this mapping's operand transformations to `src_operands`,
  // returning a new vector of operands suitable for `dst`.
  // `src_operands` can be `src()->operands()` or any other vector of
  // operands (e.g. from cloned nodes).
  virtual absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const = 0;

  // Applies output transformations to `dst_output`, returning a node that is
  // functionally equivalent to `src()`. Note that `dst_output` is commonly a
  // clone of `dst` or a node that will replace both `src` and `dst`.
  virtual absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                              Node* dst_output) const = 0;

  // Returns the estimated area overhead incurred by this mapping (e.g.
  // negation logic, bit-extension, output slicing). `operands` and `output`
  // allow the mapping to inspect the specific nodes being transformed
  // (defaulting to `src()->operands()` and `src()`).
  virtual absl::StatusOr<double> EstimateAreaOverhead(
      const AreaEstimator& area_estimator, absl::Span<Node* const> operands,
      Node* output) const {
    return 0.0;
  }

  // Returns true if this mapping modifies any of the operands.
  virtual absl::StatusOr<bool> RequiresOperandTransformation() const {
    return true;
  }

  // Returns true if this mapping modifies the output of the dst node.
  virtual absl::StatusOr<bool> RequiresOutputTransformation() const {
    return false;
  }

 protected:
  absl::StatusOr<Node*> MapNode(
      Node* node, std::optional<const absl::flat_hash_map<Node*, Node*>*>
                      original_node_to_clone) const;

  // Clones EquivalenceMapping if it has only the members from the base class.
  template <typename EqMapping>
  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> CloneEqMapping(
      const EqMapping* mapping,
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const {
    XLS_ASSIGN_OR_RETURN(
        Node * new_src,
        mapping->MapNode(mapping->src_, original_node_to_clone));
    XLS_ASSIGN_OR_RETURN(
        Node * new_dst,
        mapping->MapNode(mapping->dst_, original_node_to_clone));
    return std::make_unique<EqMapping>(new_src, new_dst, mapping->tmp_package_);
  }

  Node* src_;
  Node* dst_;
  // Used to hold `dst_` if `dst` is a temporary created by the caller that
  // constructed this mapping. Uses shared_ptr because multiple mappings may be
  // created to the same temporary `dst`.
  std::shared_ptr<Package> tmp_package_;
};

// `NodeEquivalenceMapper` is a thread-safe registry of `EquivalenceMapping`
// implementations. It iterates over all registered `EquivalenceMapping`
// `TryCreate` factories to find a valid mapping between source nodes and a
// destination node. The returned mappings are for each source, and optionally
// for the destination, if it must be modified to support the sources. If `dst`
// must be modified, all mappings are to that modified version of `dst`, NOT to
// `dst`; in other words, the original `dst` becomes a source of sorts, e.g:
//   if you want to map shrl -> shra then the shra needs to be 0-padded, so
//   the returned mappings are shrl -> wide_shra and shra -> wide_shra.
class NodeEquivalenceMapper {
 public:
  using Factory = absl::StatusOr<std::optional<
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>>> (*)(
      absl::Span<Node* const> sources, Node* dst);

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

  // Returns an `EquivalenceMapping` map if `sources` can be mapped to `dst`
  // using any registered `EquivalenceMapping` implementation, or `std::nullopt`
  // if no mapping applies. The map contains mappings for every source AND for
  // the dst such that the dst mapping suffices for all the sources.
  absl::StatusOr<std::optional<
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>>>
  ComputeMappings(absl::Span<Node* const> sources, Node* dst) const;

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

// Populates hash map of the equivalence mappings from each source to the dest.
template <typename EqMapping>
absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>
ComputeMappingsSourcesToDest(absl::Span<Node* const> sources, Node* dst) {
  absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>> mappings;
  mappings.reserve(sources.size());
  for (Node* src : sources) {
    mappings[src] = std::make_unique<EqMapping>(src, dst);
  }
  return mappings;
}

// `IdentityEquivalenceMapping` handles nodes with identical ops and bit widths.
// This mapping is exposed for ease of default mapping in constructors.
class IdentityEquivalenceMapping : public EquivalenceMapping {
 public:
  static absl::StatusOr<std::optional<
      absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>>>
  TryCreate(absl::Span<Node* const> sources, Node* dst) {
    for (Node* src : sources) {
      if (!src->IsDefinitelyEqualTo(dst)) {
        return std::nullopt;
      }
    }
    return ComputeMappingsSourcesToDest<IdentityEquivalenceMapping>(sources,
                                                                    dst);
  }

  using EquivalenceMapping::EquivalenceMapping;

  absl::StatusOr<std::unique_ptr<EquivalenceMapping>> Clone(
      std::optional<const absl::flat_hash_map<Node*, Node*>*>
          original_node_to_clone) const override {
    return CloneEqMapping(this, original_node_to_clone);
  }

  absl::StatusOr<std::vector<Node*>> ApplyToOperands(
      FunctionBase* f, absl::Span<Node* const> src_operands) const override {
    return std::vector<Node*>(src_operands.begin(), src_operands.end());
  }

  absl::StatusOr<Node*> ApplyToOutput(FunctionBase* f,
                                      Node* dst_output) const override {
    return dst_output;
  }

  absl::StatusOr<bool> RequiresOperandTransformation() const override {
    return false;
  }
  absl::StatusOr<bool> RequiresOutputTransformation() const override {
    return false;
  }
};

}  // namespace xls

#endif  // XLS_PASSES_RESOURCE_SHARING_EQUIVALENCE_H_
