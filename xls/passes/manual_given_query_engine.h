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

#ifndef XLS_PASSES_MANUAL_GIVEN_QUERY_ENGINE_H_
#define XLS_PASSES_MANUAL_GIVEN_QUERY_ENGINE_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/forwarding_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

// A query engine helper that holds givens and allows for manual invalidation of
// results. This is meant to wrap another query engine and be used to maintain a
// set of expensive pre-calculated results and use it to seed the internal query
// engine.
class ManualGivenQueryEngine : public ForwardingQueryEngine {
 public:
  explicit ManualGivenQueryEngine(UnionQueryEngine&& qe)
      : ForwardingQueryEngine(),
        base_union_(std::move(qe)),
        specialized_union_(std::nullopt) {}

  explicit ManualGivenQueryEngine(
      std::vector<std::unique_ptr<QueryEngine>> owned_engines,
      std::vector<QueryEngine*> unowned_engines)
      : ForwardingQueryEngine(),
        base_union_(std::move(owned_engines), std::move(unowned_engines)),
        specialized_union_(std::nullopt) {}

  // Populate the underlying query engine. Resets any previous information.
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    specialized_union_.reset();
    return base_union_.Populate(f);
  }

  absl::Status ClearGivens() {
    specialized_union_.reset();
    return absl::OkStatus();
  }

  bool has_givens() const { return specialized_union_.has_value(); }

  // Sets the givens for the nodes. Resets all prior information.
  absl::Status SetGivens(absl::Span<Node* const> nodes,
                         const QueryEngine& information_source);

  template <typename... Engines>
  static ManualGivenQueryEngine Of(Engines... e) {
    return ManualGivenQueryEngine(
        UnionQueryEngine::Of(std::forward<Engines>(e)...));
  }

 protected:
  QueryEngine& real() override {
    return specialized_union_ ? **specialized_union_
                              : static_cast<QueryEngine&>(base_union_);
  }

  const QueryEngine& real() const override {
    return specialized_union_ ? **specialized_union_
                              : static_cast<const QueryEngine&>(base_union_);
  }

 private:
  UnionQueryEngine base_union_;
  std::optional<std::unique_ptr<QueryEngine>> specialized_union_;
};

}  // namespace xls

#endif  // XLS_PASSES_MANUAL_GIVEN_QUERY_ENGINE_H_
