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

#ifndef XLS_PASSES_CACHED_STATE_ELEMENT_QUERY_ENGINE_H_
#define XLS_PASSES_CACHED_STATE_ELEMENT_QUERY_ENGINE_H_

#include "absl/status/status.h"
#include "xls/ir/function_base.h"
#include "xls/passes/manual_given_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/proc_state_provenance_narrowing_pass.h"
#include "xls/passes/proc_state_range_query_engine.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

// A query-engine which caches the results of inductive state element
// computations.
//
// The state element bounds are only recreated when manually invalidated.
//
// This query engine does a full inductive proc-state analysis the first time it
// is populated and then caches that.
class CachedStateElementQueryEngine : public ManualGivenQueryEngine {
 public:
  // Creates a base query engine which we will cache results for.
  //
  // QE is not populated
  static UnionQueryEngine MakeBaseQueryEngine() {
    return UnionQueryEngine::Of(StatelessQueryEngine(),
                                ProcStateRangeQueryEngine());
  }

  // Helper to create a cached-state qe, pulled from the opt context and using
  // the opt-context to provide the base query engine.
  template <typename EngineT, typename... Args>
  static CachedStateElementQueryEngine* FromContext(OptimizationContext& ctx,
                                                    Proc* p, Args... args) {
    return ctx.SharedQueryEngine<CachedStateElementQueryEngine>(
        p, ctx.SharedQueryEngine<EngineT>(p, std::forward<Args>(args)...));
  }
  // Helper that will just return the base query engine if we are not a proc.
  template <typename EngineT, typename... Args>
  static QueryEngine* FromContext(OptimizationContext& ctx, FunctionBase* f,
                                  Args... args) {
    if (f->IsProc()) {
      return FromContext<EngineT>(ctx, f->AsProcOrDie(),
                                  std::forward<Args>(args)...);
    }
    return ctx.SharedQueryEngine<EngineT>(f, std::forward<Args>(args)...);
  }

  explicit CachedStateElementQueryEngine(QueryEngine* to_specialize)
      : ManualGivenQueryEngine({}, {to_specialize}) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    if (function_ == f) {
      return ReachedFixpoint::Unchanged;
    }
    // Cache the function pointer so we can recompute the state element bounds
    // when requested.
    function_ = f;
    return ManualGivenQueryEngine::Populate(f);
  }

  // Fully recomputes the state element bounds and updates the internal query
  // engine. Prior to this being called this query engine functions simply as a
  // forwarding query engine.
  absl::Status Compute();
  // Compute using the given populated proc-state info.
  absl::Status ComputeWithStateInfo(ProcStateRangeQueryEngine& proc_state_info);

 private:
  FunctionBase* function_ = nullptr;
};

}  // namespace xls

#endif  // XLS_PASSES_CACHED_STATE_ELEMENT_QUERY_ENGINE_H_
